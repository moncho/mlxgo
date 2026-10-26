"""Generate a tiny FP32 LoRA training reference with pinned LitGPT/PyTorch.

Install requirements-training-reference.txt in an isolated Python 3.12 venv.
This runs on CPU without downloading a pretrained model or importing MLX.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform

import torch
from litgpt.lora import Config, GPT, mark_only_lora_as_trainable
from torch.nn.attention import SDPBackend, sdpa_kernel


ROOT = Path(__file__).resolve().parent


def provenance():
    requirements = ROOT / "requirements-training-reference.txt"
    pins = [line.strip() for line in requirements.read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    versions = {}
    for pin in pins:
        if "==" in pin:
            name, expected = pin.split("==")
            actual = importlib.metadata.version(name)
            if actual != expected:
                raise RuntimeError(f"{name}: expected {expected}, found {actual}")
            versions[name] = actual
    litgpt_pin = next(pin for pin in pins if pin.startswith("litgpt @ "))
    revision = litgpt_pin.rsplit("@", 1)[1]
    distribution = importlib.metadata.distribution("litgpt")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct_url.get("vcs_info", {}).get("commit_id") != revision:
        raise RuntimeError("Install LitGPT at the commit pinned in the requirements file")
    versions["litgpt"] = distribution.version
    sources = {}
    for name in ("model.py", "lora.py", "config.py"):
        path = distribution.locate_file(f"litgpt/{name}")
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "repository": "https://github.com/Lightning-AI/litgpt",
        "revision": revision,
        "versions": versions,
        "python": platform.python_version(),
        "device": "cpu",
        "dtype": "float32",
        "attention_backend": "math",
        "source_sha256": sources,
    }


def tensor(value):
    value = value.detach().contiguous()
    if value.dtype != torch.float32 or not torch.isfinite(value).all():
        raise ValueError("Fixture tensors must be finite float32")
    return {"shape": list(value.shape), "values": value.reshape(-1).tolist()}


def initialize(model):
    # Binary fractions keyed by parameter name make every exported input explicit.
    # Both LoRA factors are nonzero so one step exercises gradients through both.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            values = []
            for index in range(parameter.numel()):
                digest = hashlib.sha256(f"{name}:{index}".encode()).digest()
                value = (int.from_bytes(digest[:4], "little") % 2049 - 1024) / 4096
                values.append(value + (1 if "norm" in name or ".ln_f." in name else 0))
            parameter.copy_(torch.tensor(values, dtype=torch.float32).reshape(parameter.shape))


def export_weights(model):
    weights = {
        "embed": tensor(model.transformer.wte.weight),
        "norm": tensor(model.transformer.ln_f.weight),
    }
    for index, block in enumerate(model.transformer.h):
        c = model.config
        sizes = (c.n_head * c.head_size, c.n_query_groups * c.head_size,
                 c.n_query_groups * c.head_size)
        q, k, v = block.attn.qkv.linear.weight.split(sizes)
        bq, bk, bv = block.attn.qkv.linear.bias.split(sizes)
        layer = {
            "input_norm": block.norm_1.weight,
            "post_attention_norm": block.norm_2.weight,
            "wq": q.T, "bq": bq, "wk": k.T, "bk": bk, "wv": v.T, "bv": bv,
            "wo": block.attn.proj.linear.weight.T,
            "wgate": block.mlp.fc_1.linear.weight.T,
            "wup": block.mlp.fc_2.linear.weight.T,
            "wdown": block.mlp.proj.linear.weight.T,
        }
        weights.update({f"layers.{index}.{name}": tensor(value) for name, value in layer.items()})
    return weights


def export_adapters(model, gradients=False):
    tensors = []
    for block in model.transformer.h:
        qkv = block.attn.qkv
        a = qkv.lora_A.grad if gradients else qkv.lora_A
        b = qkv.lora_B.grad if gradients else qkv.lora_B
        # LitGPT packs enabled Q/V branches; Go stores separate [in, out] factors.
        aq, av = a.split(qkv.r)
        bq, bv = b.split(qkv.qkv_shapes)
        tensors.extend(tensor(value.T) for value in (aq, bq, av, bv))
    return tensors


def example_loss(model, example):
    ids = torch.tensor([example["inputs"]], dtype=torch.int64)
    labels = torch.tensor([example["targets"]], dtype=torch.int64)
    labels[:, :example["loss_start"]] = -100
    logits = model(ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100,
    )
    return logits, loss


def make_model(config, lora):
    c = Config(
        name="mlxgo-tiny-qwen2", block_size=config["max_position_embeddings"],
        n_layer=config["num_hidden_layers"], n_embd=config["hidden_size"],
        n_head=config["num_attention_heads"], n_query_groups=config["num_key_value_heads"],
        vocab_size=config["vocab_size"], padded_vocab_size=config["vocab_size"],
        intermediate_size=config["intermediate_size"], norm_class_name="RMSNorm",
        norm_eps=config["rms_norm_eps"], mlp_class_name="LLaMAMLP",
        parallel_residual=False, bias=False, attn_bias=True,
        rotary_percentage=1.0, rope_base=config["rope_theta"],
        lora_r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=0.0,
        lora_query=True, lora_key=False, lora_value=True,
    )
    model = GPT(c).cpu().float()
    model.lm_head.linear.weight = model.transformer.wte.weight
    initialize(model)
    mark_only_lora_as_trainable(model)
    model.train()
    return model


def make_optimizer(model, settings):
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=settings["learning_rate"], weight_decay=settings["weight_decay"],
        betas=(settings["beta1"], settings["beta2"]), eps=settings["epsilon"],
        foreach=False, fused=False,
    )


def generate():
    source = provenance()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    config = {
        "architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2",
        "hidden_size": 8, "intermediate_size": 12, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1, "rms_norm_eps": 1e-6,
        "rope_theta": 10000, "tie_word_embeddings": True, "vocab_size": 16,
        "max_position_embeddings": 32, "hidden_act": "silu",
    }
    lora = {"rank": 2, "alpha": 4}
    model = make_model(config, lora)
    examples = [
        {"inputs": [1, 4, 2, 7, 3], "targets": [4, 2, 7, 3, 9], "loss_start": 2},
        {"inputs": [6, 5, 8], "targets": [5, 8, 2], "loss_start": 1},
    ]
    optimizer_config = {
        "learning_rate": 0.01, "weight_decay": 0.1, "beta1": 0.9,
        "beta2": 0.999, "epsilon": 1e-8, "max_grad_norm": 0,
    }
    optimizer = make_optimizer(model, optimizer_config)
    fixture = {
        "format": "mlxgo-litgpt-training-v1", "source": source,
        "config": config, "lora": lora, "optimizer": optimizer_config,
        "weights": export_weights(model), "adapters": export_adapters(model),
    }
    token_count = sum(len(e["targets"]) - e["loss_start"] for e in examples)
    loss_sum = 0.0
    with sdpa_kernel(SDPBackend.MATH):
        for example in examples:
            logits, loss = example_loss(model, example)
            example["logits"], example["loss"] = tensor(logits), loss.item()
            weight = (len(example["targets"]) - example["loss_start"]) / token_count
            # Match our completion-token mean, explicitly, across unequal examples.
            (loss * weight).backward()
            loss_sum += loss.item() * weight
        fixture.update(examples=examples, loss=loss_sum, gradients=export_adapters(model, gradients=True))
        optimizer.step()
        fixture["updated_adapters"] = export_adapters(model)
        with torch.no_grad():
            fixture["post_step_loss"] = sum(
                example_loss(model, e)[1].item() * (len(e["targets"]) - e["loss_start"]) / token_count
                for e in examples
            )
    assert export_weights(model) == fixture["weights"], "Frozen base weights changed"
    assert all(any(abs(v) > 1e-5 for v in g["values"]) for g in fixture["gradients"])
    assert math.isfinite(fixture["post_step_loss"])
    return fixture


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "testdata" / "litgpt_training.json")
    args = parser.parse_args()
    fixture = generate()
    args.output.write_text(json.dumps(fixture, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}: loss {fixture['loss']:.8f} -> {fixture['post_step_loss']:.8f}")
