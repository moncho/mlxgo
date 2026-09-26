"""Real attention oracle: layers 0/2 alone or layer 2 with its layer-3 consumer.

Cache/index-query quantizers are disabled by default, or replaced by independent
CPU formulas with --quantized-caches. Projections remain float32 in both modes;
this is not released BF16/CUDA inference parity.
"""

import argparse
import ast
import base64
from functools import lru_cache
import gzip
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import urllib.request

import torch
from torch import nn
import torch.nn.functional as F

from quant.make_sample_reference import HEADER_SHA, SHARD, decoded_hash, sha
from reference import HASHES, REVISION, sparse_attention
from quant.make_activation_fixture import inplace_cache_quantizers

ROOT = Path(__file__).resolve().parent
CONFIG_SHA = "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879"
PREFILLS = [1, 127, 128, 129]
TOKENS = 131
COMPRESSED_SHARD = "model-00005-of-00048.safetensors"
COMPRESSED_HEADER_SHA = "f921056a11b2bee72e36ea301b533dd4d67cc8ae6fcc4c24f642516a2e3c4f31"
CACHE_MODE = "fp8_fp4_float32_v1"


def attention_namespace(source, kernel_source=None):
    if sha(source) != HASHES["model.py"]:
        raise ValueError("model source checksum mismatch")
    names = {"RMSNorm", "precompute_freqs_cis", "apply_rotary_emb", "get_window_topk_idxs",
             "Attention", "Compressor", "Indexer", "SharedAttentionRuntime", "select_candidate_blocks"}
    nodes = [n for n in ast.parse(source).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError("incomplete reference source")
    ns = dict(torch=torch, nn=nn, F=F, math=math, lru_cache=lru_cache, world_size=1,
              Linear=FloatLinear, ColumnParallelLinear=FloatLinear, RowParallelLinear=FloatLinear,
              fp8_block_size=32, fp4_block_size=32, scale_fmt="ue8m0", scale_dtype=torch.float32,
              act_quant=lambda *a, **kw: None, fp4_act_quant=lambda *a, **kw: None,
              sparse_attn=sparse_attention)
    if kernel_source is not None:
        ns["act_quant"], ns["fp4_act_quant"] = inplace_cache_quantizers(kernel_source)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future]+nodes, type_ignores=[])), "pinned-attention.py", "exec"), ns)
    return ns


def inputs(start, count, dim):
    t = torch.arange(start, start+count)[:, None]
    col = torch.arange(dim)[None, :]
    x = (((col*37 + t*17 + (col%13)*t*7) % 257 - 128).float()/128).unsqueeze(0)
    if start == 0:
        x[:, 0] = 0
    if start <= 2 < start+count:
        x[:, 2-start] *= 8
    return x


class FloatLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, dtype=None):
        super().__init__(in_features, out_features, bias=bias, dtype=torch.float32)


def load_weights(model, input_norm, samples, prefix, layer, shard, header_sha):
    header = json.loads(prefix[8:])
    manifest_bytes = (samples / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["revision"] != REVISION or manifest["shard"] != shard or manifest["header_sha256"] != header_sha or manifest["repository"] != "deepseek-ai/DeepSeek-V4.1-Flash":
        raise ValueError("sample provenance mismatch")
    entries = {t["name"]: t for t in manifest["tensors"]}
    layer_prefix = f"layers.{layer}."
    expected = {n for n in header if n.startswith(layer_prefix+"attn.") or n == layer_prefix+"attn_norm.weight"}
    if len(manifest["tensors"]) != len(expected) or set(entries) != expected:
        raise ValueError("incomplete/duplicate attention samples")

    def read(name):
        t, orig = entries[name], header[name]
        start, end = orig["data_offsets"]
        if t["file"] != name+".bin" or t["dtype"] != orig["dtype"] or t["shape"] != orig["shape"] or t["bytes"] != end-start or t["source_byte_offsets"] != [len(prefix)+start, len(prefix)+end]:
            raise ValueError("tensor layout mismatch: " + name)
        path = samples / t["file"]
        if path.stat().st_size != t["bytes"]:
            raise ValueError("tensor length mismatch")
        b = bytearray(path.read_bytes())
        if sha(b) != t["sha256"]:
            raise ValueError("tensor checksum mismatch")
        return torch.frombuffer(b, dtype=torch.uint8), t

    metadata = []
    for name in sorted(expected):
        if name.endswith(".scale"):
            continue
        raw, t = read(name)
        scales_sha = ""
        if t["dtype"] == "F8_E4M3":
            weight = raw.view(torch.float8_e4m3fn).reshape(t["shape"]).float()
            sr, st = read(name.removesuffix(".weight")+".scale")
            scale = sr.view(torch.float8_e8m0fnu).reshape(st["shape"]).float()
            weight *= scale.repeat_interleave(32, 0).repeat_interleave(32, 1)
            scales_sha = st["sha256"]
            if name.endswith("wo_a.weight"):
                weight = weight.bfloat16().float()
        else:
            weight = raw.view({"BF16": torch.bfloat16, "F32": torch.float32}[t["dtype"]]).reshape(t["shape"]).float()
        if not torch.isfinite(weight).all():
            raise ValueError("nonfinite weight")
        metadata.append(dict(name=name, shape=t["shape"], dtype=t["dtype"], data_sha256=t["sha256"],
                             scales_sha256=scales_sha, decoded_sha256=decoded_hash(weight)))
        if name == layer_prefix+"attn_norm.weight":
            input_norm.weight = nn.Parameter(weight, requires_grad=False)
        else:
            parts = name.removeprefix(layer_prefix+"attn.").split(".")
            target = model
            for part in parts[:-1]:
                target = getattr(target, part)
            setattr(target, parts[-1], nn.Parameter(weight, requires_grad=False))
        print("Loaded", name, flush=True)
    return metadata, sha(manifest_bytes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--model-source", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--layer", type=int, choices=(0, 2), default=0)
    parser.add_argument("--consumer-samples", type=Path, help="layer-3 samples; validate the layer-2/3 attention pair")
    parser.add_argument("--quantized-caches", action="store_true", help="enable CPU cache/index-query quantizers; keep float32 projections")
    parser.add_argument("--kernel-source", type=Path, help="pinned kernel.py, required with --quantized-caches")
    args = parser.parse_args()
    if args.consumer_samples and args.layer != 2:
        parser.error("--consumer-samples requires --layer 2")
    if args.quantized_caches != bool(args.kernel_source):
        parser.error("--quantized-caches and --kernel-source must be used together")
    filename = "quantized-cache-reference.json.gz" if args.quantized_caches else "attention-reference.json.gz"
    out = args.out or (args.consumer_samples or args.samples) / filename
    if out.exists():
        raise FileExistsError(out)
    if torch.__version__.split("+", 1)[0] != "2.14.0" or sys.byteorder != "little":
        raise RuntimeError("Use torch==2.14.0 on a little-endian host")
    torch.set_num_threads(1)
    if args.model_source:
        source = args.model_source.read_bytes()
    else:
        url = f"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/{REVISION}/inference/model.py"
        with urllib.request.urlopen(url, timeout=60) as response:
            source = response.read(1 << 20)
    if sha(source) != HASHES["model.py"]:
        raise ValueError("model source checksum mismatch")
    ns = attention_namespace(source, args.kernel_source.read_bytes() if args.kernel_source else None)

    def save(report):
        if args.quantized_caches:
            report.update(cache_quantization=CACHE_MODE, kernel_sha256=HASHES["kernel.py"])
        write_report(out, report)

    snapshot = json.loads(gzip.decompress((ROOT / "testdata" / "released-checkpoint.metadata.json.gz").read_bytes()))
    shard, header_sha = (SHARD, HEADER_SHA) if args.layer == 0 else (COMPRESSED_SHARD, COMPRESSED_HEADER_SHA)
    prefix = base64.b64decode(snapshot["headers"][shard]["prefix"])
    config_bytes = base64.b64decode(snapshot["config"])
    if snapshot["revision"] != REVISION or sha(prefix) != header_sha or sha(config_bytes) != CONFIG_SHA:
        raise ValueError("audit header/config checksum mismatch")
    c = json.loads(config_bytes)["text_config"]
    if c["compress_ratios"][args.layer] != args.layer:
        raise ValueError("unexpected compression ratio")
    settings = dict(dim=c["hidden_size"], n_heads=c["num_attention_heads"], q_lora_rank=c["q_lora_rank"],
                    o_lora_rank=c["o_lora_rank"], head_dim=c["head_dim"], rope_head_dim=c["qk_rope_head_dim"],
                    o_groups=c["o_groups"], window_size=c["sliding_window"], compress_ratios=[0],
                    norm_eps=c["rms_norm_eps"], n_layers=1, kv_source_layers=[], index_source_layers=[],
                    max_batch_size=1, max_seq_len=TOKENS, original_seq_len=0, rope_theta=c["rope_theta"],
                    rope_factor=c["rope_scaling"]["factor"], beta_fast=c["rope_scaling"]["beta_fast"],
                    beta_slow=c["rope_scaling"]["beta_slow"])
    if args.layer == 2:
        settings.update(n_layers=3, compress_ratios=c["compress_ratios"][:3],
                        kv_source_layers=[2], index_source_layers=[2],
                        compress_rope_theta=c["compress_rope_theta"],
                        original_seq_len=c["rope_scaling"]["original_max_position_embeddings"],
                        index_n_heads=c["index_n_heads"], index_head_dim=c["index_head_dim"],
                        index_topk=c["index_topk"], candidate_source_layer=c["candidate_source_layer_id"],
                        candidate_topk_blocks=c["candidate_topk_blocks"], candidate_block_size=c["candidate_block_size"],
                        max_seq_len=1281)
        ns["shared_attn"] = ns["SharedAttentionRuntime"]()
    # Avoid random allocation of the large linear weights; replace all meta
    # parameters below. Cache/frequency buffers are constructed normally on CPU.
    class MetaLinear(FloatLinear):
        def __init__(self, *a, **kw):
            with torch.device("meta"):
                super().__init__(*a, **kw)
    ns.update(Linear=MetaLinear, ColumnParallelLinear=MetaLinear, RowParallelLinear=MetaLinear)
    model = ns["Attention"](args.layer, SimpleNamespace(**settings))
    input_norm = ns["RMSNorm"](settings["dim"], settings["norm_eps"])
    metadata, manifest_sha = load_weights(model, input_norm, args.samples, prefix, args.layer, shard, header_sha)
    if args.consumer_samples:
        report = shared_reference(ns, model, input_norm, settings, snapshot, args.consumer_samples, manifest_sha)
        save(report)
        return

    x = inputs(0, TOKENS, settings["dim"])
    flat = lambda v: v.detach().float().flatten().tolist()
    with torch.no_grad():
        normalized = input_norm(x)
        full = model(normalized, 0)
        report = dict(schema=1, revision=REVISION, model_sha256=HASHES["model.py"], config_sha256=CONFIG_SHA,
                      torch_version=torch.__version__, manifest_sha256=manifest_sha, config=settings,
                      tensors=metadata, tokens=TOKENS, full_output=flat(full), cases=[])
        for n in ([1, 2, 127, 128, 129] if args.layer == 2 else PREFILLS):
            model.window_kv_cache.zero_()
            if args.layer == 2:
                ns["shared_attn"] = ns["SharedAttentionRuntime"]()
                model.compress_kv_cache.zero_()
                model.indexer.k_cache.zero_()
                model.compressor.kv_state.zero_()
                model.compressor.score_state.fill_(-torch.inf)
            ys = [model(normalized[:, :n], 0)]
            for pos in range(n, n+2):
                ys.append(model(normalized[:, pos:pos+1], pos))
            y = torch.cat(ys, 1)
            end = n+2
            # Convert official ring storage into chronological order for
            # comparison with the Go implementation's chronological cache.
            ids = torch.arange(max(0, end-settings["window_size"]), end) % settings["window_size"]
            cache = model.window_kv_cache[:, ids]
            error = float((y-full[:, :end]).abs().max())
            report["cases"].append(dict(prefill=n, output=flat(y), cache=flat(cache), reference_cached_full_max_error=error))
            if args.layer == 2:
                report["cases"][-1].update(compressed=flat(model.compress_kv_cache[:, :end//2]),
                                          keys=flat(model.indexer.k_cache[:, :end//2]),
                                          pending=flat(normalized[:, end-end%2:end]))
            print(f"prefill={n}, decode=2: reference cached/full max error {error}", flush=True)
        if args.layer == 2:
            # Exercise real top-512 pruning without materializing full long
            # attention. Keys come from the real compressor and key projection.
            long_x = input_norm(inputs(0, 1280, settings["dim"]))
            latent = model.compressor(long_x, 0)
            keys = model.indexer.k_norm(model.indexer.wk(latent))
            ns["apply_rotary_emb"](keys[..., -settings["rope_head_dim"]:], model.freqs_cis[:1280:2])
            ns["fp4_act_quant"](keys, 32, True)
            model.indexer.k_cache[:, :640] = keys
            ns["shared_attn"].index_k = model.indexer.k_cache
            probes = []
            for pos in [1022, 1024, 1280]:
                query_x = input_norm(inputs(pos, 1, settings["dim"]))
                qr = model.q_norm(model.wq_a(query_x))
                ids = model.indexer(query_x, qr, None, pos, 0)
                probes.append(dict(position=pos, indices=ids.flatten().tolist()))
            report.update(layer=2, index_keys=flat(keys), index_cases=probes)
    save(report)


def shared_reference(ns, owner, owner_norm, settings, snapshot, samples, owner_manifest_sha):
    shard = "model-00006-of-00048.safetensors"
    header_sha = "758bbceaef14ca838a4b6dec90cf3946dfa0d7b35e223153587f6354c0fbb0a1"
    prefix = base64.b64decode(snapshot["headers"][shard]["prefix"])
    if sha(prefix) != header_sha:
        raise ValueError("consumer header checksum mismatch")
    settings = dict(settings, n_layers=4, compress_ratios=[0, 0, 2, 2])
    consumer = ns["Attention"](3, SimpleNamespace(**settings))
    consumer_norm = ns["RMSNorm"](settings["dim"], settings["norm_eps"])
    metadata, manifest_sha = load_weights(consumer, consumer_norm, samples, prefix, 3, shard, header_sha)
    if consumer.compressor is not None or consumer.indexer is not None:
        raise ValueError("expected a cache-only consumer")

    def reset():
        ns["shared_attn"] = ns["SharedAttentionRuntime"]()
        owner.window_kv_cache.zero_()
        owner.compress_kv_cache.zero_()
        owner.indexer.k_cache.zero_()
        owner.compressor.kv_state.zero_()
        owner.compressor.score_state.fill_(-torch.inf)
        consumer.window_kv_cache.zero_()

    flat = lambda v: v.detach().float().flatten().tolist()
    with torch.no_grad():
        normalized = owner_norm(inputs(0, TOKENS, settings["dim"]))

        def forward(start, count):
            y = owner(normalized[:, start:start+count], start)
            # Deliberately only attention composition, not a transformer block:
            # no residual/mHC/FFN is inserted between the two attention calls.
            return consumer(consumer_norm(y), start)

        reset()
        full = forward(0, TOKENS)
        report = dict(schema=1, revision=REVISION, model_sha256=HASHES["model.py"], config_sha256=CONFIG_SHA,
                      torch_version=torch.__version__, manifest_sha256=manifest_sha, config=settings,
                      tensors=metadata, tokens=TOKENS, full_output=flat(full), cases=[], layer=3,
                      owner_manifest_sha256=owner_manifest_sha, wiring="norm3(attention2(norm2(x)))")
        for n in [1, 2, 127, 128, 129]:
            reset()
            ys = [forward(0, n)]
            ys.extend(forward(pos, 1) for pos in range(n, n+2))
            y = torch.cat(ys, 1)
            end = n+2
            ids = torch.arange(max(0, end-settings["window_size"]), end) % settings["window_size"]
            error = float((y-full[:, :end]).abs().max())
            report["cases"].append(dict(prefill=n, output=flat(y), cache=flat(consumer.window_kv_cache[:, ids]),
                                        reference_cached_full_max_error=error))
            print(f"shared prefill={n}, decode=2: reference cached/full max error {error}", flush=True)
    return report


def write_report(out, report):
    payload = (json.dumps(report, separators=(",", ":"), allow_nan=False)+"\n").encode()
    with out.open("xb") as f:
        f.write(gzip.compress(payload, mtime=0))
    print(f"Saved {out}; cache quantization: {report.get('cache_quantization', 'disabled')}")


if __name__ == "__main__":
    main()
