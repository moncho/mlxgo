"""Benchmark real Qwen2.5 checkpoint LoRA training with pinned LitGPT/PyTorch.

Uses the same FP32 weights, initial adapters and token stream as the Go benchmark.
Model loading, correctness probes and optimizer construction are outside timing.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
from litgpt.lora import Config, GPT, mark_only_lora_as_trainable
from litgpt.scripts.convert_hf_checkpoint import copy_weights_qwen_2_5
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from make_training_fixture import provenance


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def reset_adapters(model, initial):
    with torch.no_grad():
        for i, block in enumerate(model.transformer.h):
            block.attn.qkv.lora_A.copy_(torch.cat([initial[f"lora.{i*4}"].T,
                                                 initial[f"lora.{i*4+2}"].T]))
            block.attn.qkv.lora_B.copy_(torch.cat([initial[f"lora.{i*4+1}"].T,
                                                 initial[f"lora.{i*4+3}"].T]))


def exported_adapters(model):
    result = {}
    for i, block in enumerate(model.transformer.h):
        qkv = block.attn.qkv
        aq, av = qkv.lora_A.detach().cpu().split(qkv.r)
        bq, bv = qkv.lora_B.detach().cpu().split(qkv.qkv_shapes)
        for j, value in enumerate((aq, bq, av, bv)):
            result[f"lora.{i*4+j}"] = value.T.contiguous()
    return result


def load_model(directory, adapters, length, device):
    hf_config = json.loads((directory / "config.json").read_text())
    with safe_open(adapters, framework="pt") as stream:
        metadata = stream.metadata()
    base_sha = sha256(directory / "model.safetensors")
    if metadata["base_sha256"] != base_sha:
        raise ValueError("Adapters belong to a different base checkpoint")
    if json.loads(metadata["config"]) != {key: hf_config.get(key) for key in json.loads(metadata["config"])}:
        raise ValueError("Adapter and checkpoint configurations differ")
    rank, alpha = int(metadata["rank"]), float(metadata["alpha"])
    if rank != 8 or alpha != 16:
        raise ValueError("This benchmark uses rank-8, alpha-16 adapters")
    config = Config(
        name="mlxgo-qwen-checkpoint", block_size=length,
        n_layer=hf_config["num_hidden_layers"], n_embd=hf_config["hidden_size"],
        n_head=hf_config["num_attention_heads"], n_query_groups=hf_config["num_key_value_heads"],
        vocab_size=hf_config["vocab_size"], padded_vocab_size=hf_config["vocab_size"],
        intermediate_size=hf_config["intermediate_size"], norm_class_name="RMSNorm",
        norm_eps=hf_config["rms_norm_eps"], mlp_class_name="LLaMAMLP",
        parallel_residual=False, bias=False, attn_bias=True,
        rotary_percentage=1.0, rope_base=hf_config["rope_theta"],
        lora_r=rank, lora_alpha=alpha, lora_dropout=0.0,
        lora_query=True, lora_key=False, lora_value=True,
    )
    if not hf_config["tie_word_embeddings"] or hf_config.get("use_sliding_window", False):
        raise ValueError("Benchmark expects tied embeddings and full attention")
    with torch.device("meta"):
        model = GPT(config)
    hf_weights = load_file(directory / "model.safetensors")
    converted = {}
    copy_weights_qwen_2_5(config, {}, converted, hf_weights, dtype=torch.float32)
    del hf_weights
    initial = load_file(adapters)
    if set(initial) != {f"lora.{i}" for i in range(4 * config.n_layer)}:
        raise ValueError("Incomplete adapter checkpoint")
    for i in range(config.n_layer):
        prefix = f"transformer.h.{i}.attn.qkv"
        converted[prefix + ".lora_A"] = torch.cat([initial[f"lora.{i*4}"].T,
                                                   initial[f"lora.{i*4+2}"].T])
        converted[prefix + ".lora_B"] = torch.cat([initial[f"lora.{i*4+1}"].T,
                                                   initial[f"lora.{i*4+3}"].T])
    model.load_state_dict(converted, strict=True, assign=True)
    del converted
    model.cos, model.sin = model.rope_cache(device=torch.device("cpu"))
    model = model.to(device)
    model.lm_head.linear.weight = model.transformer.wte.weight
    mark_only_lora_as_trainable(model)
    model.train()
    initial = {name: value.to(device) for name, value in initial.items()}
    reset_adapters(model, initial)
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Benchmark requires FP32 parameters")
    return model, initial, base_sha, hf_config


def new_optimizer(model):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8,
                            foreach=False, fused=False)


def loss_tensor(model, tokens, device):
    # Include input creation each step; select supervised rows before cross-entropy.
    ids = torch.tensor([tokens[:-1]], dtype=torch.int64, device=device)
    labels = torch.tensor(tokens[1:], dtype=torch.int64, device=device)
    logits = model(ids)
    start = len(labels) // 2
    return torch.nn.functional.cross_entropy(logits[0, start:, :], labels[start:])


def step(model, optimizer, tokens, device):
    optimizer.zero_grad(set_to_none=True)
    loss = loss_tensor(model, tokens, device)
    loss.backward()
    value = loss.item()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                  math.inf, error_if_nonfinite=True, foreach=False)
    optimizer.step()
    if not math.isfinite(value):
        raise ValueError("Nonfinite training loss")
    return value


def run(args):
    source = provenance()
    source.update(device=args.device, attention_backend="native default dispatch")
    model, initial, base_sha, hf_config = load_model(args.model, args.adapters, args.length, args.device)
    tokens = [(i * 73 + 19) % hf_config["vocab_size"] for i in range(args.length + 1)]
    with torch.no_grad():
        initial_loss = loss_tensor(model, tokens, args.device).item()
    optimizer = new_optimizer(model)
    first_step_loss = step(model, optimizer, tokens, args.device)
    with torch.no_grad():
        post_step_loss = loss_tensor(model, tokens, args.device).item()
    updated = exported_adapters(model)
    if args.verify_output:
        save_file(updated, args.verify_output)
    verification = None
    if args.reference:
        reference = json.loads(args.reference.read_text())
        if reference["checkpoint_sha256"] != base_sha or reference["sequence_length"] != args.length:
            raise ValueError("Reference has different checkpoint or sequence length")
        if reference["initial_adapters_sha256"] != sha256(args.adapters):
            raise ValueError("Reference starts from different adapters")
        for actual, expected in ((initial_loss, reference["initial_loss"]),
                                 (post_step_loss, reference["post_one_update_loss"])):
            if abs(actual - expected) > 5e-4:
                raise ValueError(f"Loss parity failed: {actual} vs {expected}")
        expected = load_file(args.reference_adapters)
        if set(expected) != set(updated):
            raise ValueError("Reference adapter names differ")
        errors, failures, entries = [], 0, 0
        actual_updates, expected_updates = [], []
        for name, value in updated.items():
            if value.shape != expected[name].shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid updated adapter: {name}")
            failures += (~torch.isclose(value, expected[name], atol=2e-6, rtol=2e-4)).sum().item()
            entries += value.numel()
            errors.append((value - expected[name]).abs().max().item())
            # B starts at zero; characterize its actual learning update separately
            # from A, whose first step only has decoupled weight decay.
            if int(name.split(".")[1]) % 2 == 1:
                actual_updates.append(value.reshape(-1).double())
                expected_updates.append(expected[name].reshape(-1).double())
        actual_update = torch.cat(actual_updates)
        expected_update = torch.cat(expected_updates)
        verification = {"reference_implementation": reference["implementation"],
                        "reference_device": reference["device"], "loss_atol": 5e-4,
                        "updated_adapters_atol": 2e-6, "updated_adapters_rtol": 2e-4,
                        "initial_loss_absolute_error": abs(initial_loss - reference["initial_loss"]),
                        "post_step_loss_absolute_error": abs(post_step_loss - reference["post_one_update_loss"]),
                        "updated_adapters_max_absolute_error": max(errors),
                        "updated_adapters_entries": entries,
                        "strict_adapter_check_failed_entries": failures,
                        "strict_adapter_check_passed": failures == 0,
                        "loss_checks_passed": True,
                        "b_update_relative_l2_error": ((actual_update - expected_update).norm() / expected_update.norm()).item(),
                        "b_update_cosine_similarity": torch.nn.functional.cosine_similarity(actual_update, expected_update, dim=0).item(),
                        "scope": "One-step numerical characterization; strict mismatches are reported without relaxing the tiny fixture's tolerances."}
        print(f"verification: losses within 5e-4; {failures}/{entries} adapter entries outside strict tolerance", flush=True)
    del optimizer, updated
    measurements, final_losses = [], []
    for round_index in range(args.rounds):
        reset_adapters(model, initial)
        optimizer = new_optimizer(model)
        for _ in range(args.warmup):
            step(model, optimizer, tokens, args.device)
        sync(args.device)
        start = time.perf_counter()
        for _ in range(args.steps):
            loss = step(model, optimizer, tokens, args.device)
        sync(args.device)
        measurements.append((time.perf_counter() - start) * 1000 / args.steps)
        final_losses.append(loss)
        print(f"round {round_index+1}: {measurements[-1]:.3f} ms/step; loss {loss:.8f}", flush=True)
        del optimizer
    median = statistics.median(measurements)
    result = {
        "implementation": "LitGPT/PyTorch", "device": args.device, "dtype": "float32",
        "model": args.model.name, "model_sha256": base_sha, "config": hf_config,
        "adapter_sha256": sha256(args.adapters),
        "base_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "sequence_length": args.length, "batch_size": 1, "loss_start": args.length // 2,
        "token_formula": "ids[i] = (i * 73 + 19) % vocab_size; shifted next-token labels",
        "lora_rank": 8, "lora_alpha": 16, "learning_rate": 1e-4, "weight_decay": 0.01,
        "rounds": args.rounds, "warmup_steps": args.warmup, "measured_steps": args.steps,
        "ms_per_step": measurements, "median_ms_per_step": median,
        "input_tokens_per_second": args.length * 1000 / median,
        "supervised_tokens_per_second": (args.length - args.length // 2) * 1000 / median,
        "initial_loss": initial_loss, "first_step_loss": first_step_loss, "post_step_loss": post_step_loss,
        "final_losses": final_losses, "source": source,
        "torch_cpu_threads": torch.get_num_threads(), "mps_cpu_fallback": False,
        "vec_lib_maximum_threads": os.environ.get("VECLIB_MAXIMUM_THREADS"),
        "torch_compile": False, "gradient_checkpointing": False,
        "verification": verification,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapters", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--threads", type=int, default=5)
    parser.add_argument("--verify-output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-adapters", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.reference) != bool(args.reference_adapters):
        parser.error("provide both --reference and --reference-adapters")
    if min(args.length, args.rounds, args.warmup, args.steps, args.threads) <= 0:
        parser.error("length, rounds, warmup, steps and threads must be positive")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        parser.error("disable MPS CPU fallback")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS unavailable")
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float32)
    run(args)
