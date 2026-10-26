"""Time the fixture's LitGPT LoRA training workload after untimed warmup.

Use requirements-training-reference.txt. Run separately from Go benchmarks.
"""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from make_training_fixture import (
    ROOT, export_adapters, export_weights, make_model, make_optimizer, provenance,
)


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()


def new_run(fixture, device):
    model = make_model(fixture["config"], fixture["lora"])
    if export_weights(model) != fixture["weights"] or export_adapters(model) != fixture["adapters"]:
        raise ValueError("Benchmark inputs differ from the checked-in fixture")
    model = model.to(device)
    model.lm_head.linear.weight = model.transformer.wte.weight
    return model, make_optimizer(model, fixture["optimizer"])


def step(model, optimizer, fixture, device, verify=False):
    optimizer.zero_grad(set_to_none=True)
    token_count = sum(len(e["targets"]) - e["loss_start"] for e in fixture["examples"])
    loss_sum = 0.0
    for example in fixture["examples"]:
        # Include array construction/transfer and scalar loss reads as Go Train does.
        ids = torch.tensor([example["inputs"]], dtype=torch.int64, device=device)
        labels = torch.tensor([example["targets"]], dtype=torch.int64, device=device)
        labels[:, :example["loss_start"]] = -100
        logits = model(ids)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100,
        )
        weight = (len(example["targets"]) - example["loss_start"]) / token_count
        (loss * weight).backward()
        loss_sum += loss.item() * weight
        if verify:
            expected = torch.tensor(example["logits"]["values"]).reshape(example["logits"]["shape"])
            torch.testing.assert_close(logits.detach().cpu(), expected, atol=2e-6, rtol=2e-5)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if verify:
        for actual, expected in zip(export_adapters(model, gradients=True), fixture["gradients"], strict=True):
            torch.testing.assert_close(torch.tensor(actual["values"]), torch.tensor(expected["values"]),
                                       atol=2e-6, rtol=2e-4)
        if abs(loss_sum - fixture["loss"]) > 2e-6 + 2e-5 * abs(fixture["loss"]):
            raise ValueError("Reference loss mismatch")
    # Go computes a global finite-gradient norm even with clipping disabled.
    torch.nn.utils.clip_grad_norm_(parameters, math.inf, error_if_nonfinite=True, foreach=False)
    optimizer.step()
    if verify:
        for actual, expected in zip(export_adapters(model), fixture["updated_adapters"], strict=True):
            torch.testing.assert_close(torch.tensor(actual["values"]), torch.tensor(expected["values"]),
                                       atol=2e-6, rtol=2e-5)
    if not math.isfinite(loss_sum):
        raise ValueError("Nonfinite training loss")
    return loss_sum


def benchmark(fixture, device, rounds, warmup, steps):
    model, optimizer = new_run(fixture, device)
    with sdpa_kernel(SDPBackend.MATH):
        step(model, optimizer, fixture, device, verify=True)
    synchronize(device)
    del optimizer, model
    measurements, final_losses = [], []
    for _ in range(rounds):
        model, optimizer = new_run(fixture, device)
        with sdpa_kernel(SDPBackend.MATH):
            for _ in range(warmup):
                step(model, optimizer, fixture, device)
            synchronize(device)
            start = time.perf_counter()
            for _ in range(steps):
                loss = step(model, optimizer, fixture, device)
            synchronize(device)
            measurements.append((time.perf_counter() - start) * 1000 / steps)
        final_losses.append(loss)
        del optimizer, model
    return {"device": device, "ms_per_step": measurements,
            "median_ms_per_step": statistics.median(measurements), "final_losses": final_losses}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.rounds, args.warmup, args.steps) <= 0:
        parser.error("rounds, warmup, and steps must be positive")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        parser.error("disable MPS CPU fallback for device comparisons")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    source = provenance()
    source["device"] = args.device
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    fixture = json.loads((ROOT / "testdata" / "litgpt_training.json").read_text())
    result = benchmark(fixture, args.device, args.rounds, args.warmup, args.steps)
    result.update(source=source, rounds=args.rounds, warmup_steps=args.warmup,
                  measured_steps=args.steps, torch_cpu_threads=torch.get_num_threads(),
                  mps_cpu_fallback=False, vec_lib_maximum_threads=os.environ.get("VECLIB_MAXIMUM_THREADS"))
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")
