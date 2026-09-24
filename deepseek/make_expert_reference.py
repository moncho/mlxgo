"""Generate an unquantized full-expert oracle from real FP4 weights.

Executes the pinned official Expert.forward with ordinary float32 linear
operations in place of quantized kernels. This is not quantized-kernel parity.
"""

import argparse
import ast
import base64
import gzip
import json
from pathlib import Path
import sys
import urllib.request

import torch
import torch.nn.functional as F

from quant.make_sample_reference import CONVERT_SHA, HEADER_SHA, SHARD, decoded_hash, sha
from reference import HASHES, REVISION

ROOT = Path(__file__).resolve().parent
PREFIX = "layers.0.ffn.experts.0."
CONFIG_SHA = "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, help="local pinned model.py; otherwise fetch just this source file")
    parser.add_argument("--out", type=Path, help="new output path; defaults to <samples>/expert-reference.json")
    args = parser.parse_args()
    out = args.out or args.samples / "expert-reference.json"
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
        raise ValueError("model.py checksum mismatch")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "Expert")
    node.bases = []
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "forward"]
    ns = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "pinned-expert.py", "exec"), ns)
    expert = ns["Expert"]()

    snapshot = json.loads(gzip.decompress((ROOT / "testdata" / "released-checkpoint.metadata.json.gz").read_bytes()))
    convert = base64.b64decode(snapshot["conversion_source"])
    prefix = base64.b64decode(snapshot["headers"][SHARD]["prefix"])
    config_bytes = base64.b64decode(snapshot["config"])
    if (snapshot["revision"] != REVISION or sha(convert) != CONVERT_SHA
            or sha(prefix) != HEADER_SHA or sha(config_bytes) != CONFIG_SHA):
        raise ValueError("audit checksum mismatch")
    header = json.loads(prefix[8:])
    config = json.loads(config_bytes)["text_config"]
    dim, inter, limit = config["hidden_size"], config["moe_intermediate_size"], config["swiglu_limit"]
    table = next(n for n in ast.parse(convert).body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "FP4_TABLE" for t in n.targets))
    exec(compile(ast.Module(body=[table], type_ignores=[]), "pinned-fp4-table.py", "exec"), ns)
    manifest_bytes = (args.samples / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest["revision"] != REVISION or manifest["shard"] != SHARD
            or manifest["header_sha256"] != HEADER_SHA
            or manifest["repository"] != "deepseek-ai/DeepSeek-V4.1-Flash"):
        raise ValueError("sample provenance mismatch")
    entries = {t["name"]: t for t in manifest["tensors"]}
    expected = {PREFIX + w + suffix for w in ("w1", "w2", "w3") for suffix in (".weight", ".scale")}
    if len(manifest["tensors"]) != 6 or set(entries) != expected:
        raise ValueError("incorrect/duplicate tensors")

    def read(name):
        t, original = entries[name], header[name]
        start, end = original["data_offsets"]
        if (t["file"] != name + ".bin" or t["shape"] != original["shape"]
                or t["dtype"] != original["dtype"] or t["bytes"] != end - start
                or t["source_byte_offsets"] != [len(prefix) + start, len(prefix) + end]):
            raise ValueError("tensor layout mismatch")
        path = args.samples / t["file"]
        if path.stat().st_size != t["bytes"]:
            raise ValueError("tensor length mismatch")
        b = bytearray(path.read_bytes())
        if sha(b) != t["sha256"]:
            raise ValueError("tensor checksum mismatch")
        return torch.frombuffer(b, dtype=torch.uint8).reshape(t["shape"]), t["sha256"]

    matrices, metadata = {}, []
    for w, rows, cols in (("w1", inter, dim), ("w2", dim, inter), ("w3", inter, dim)):
        packed, data_sha = read(PREFIX + w + ".weight")
        scales, scales_sha = read(PREFIX + w + ".scale")
        unpacked = torch.stack([ns["FP4_TABLE"][(packed & 15).long()], ns["FP4_TABLE"][(packed >> 4).long()]], -1).flatten(-2)
        decoded = unpacked * scales.view(torch.float8_e8m0fnu).float().repeat_interleave(32, 1)
        if list(decoded.shape) != [rows, cols] or not torch.isfinite(decoded).all():
            raise ValueError("invalid expert matrix")
        matrices[w] = decoded
        setattr(expert, w, lambda x, weight=decoded: F.linear(x, weight))
        metadata.append(dict(name=PREFIX+w, rows=rows, cols=cols, data_sha256=data_sha,
                             scales_sha256=scales_sha, decoded_sha256=decoded_hash(decoded)))

    col = torch.arange(dim)
    x = torch.zeros(6, dim)
    x[0] = ((col * 37) % 257 - 128).float() / 128
    x[1] = ((col * 13) % 127 - 63).float() / 64
    x[2], x[3] = x[0] * 64, x[1] * -64
    x[5, 0], x[5, 31], x[5, -1] = 1, -2, 0.5
    gate, up = expert.w1(x), expert.w3(x)
    clipping = dict(gate_above=int((gate > limit).sum()), gate_below_negative_limit=int((gate < -limit).sum()),
                    up_above=int((up > limit).sum()), up_below=int((up < -limit).sum()))
    if min(clipping.values()) == 0:
        raise ValueError("inputs fail to exercise clipping branches")
    report = dict(schema=1, revision=REVISION, torch_version=torch.__version__,
                  model_sha256=HASHES["model.py"], conversion_sha256=CONVERT_SHA,
                  manifest_sha256=sha(manifest_bytes), dim=dim, inter_dim=inter,
                  matrices=metadata, clipping=clipping, cases=[])
    routing = torch.tensor([1, .25, .75, 0, 1, 1.5]).reshape(6, 1)
    for name, cap, weights in (("clipped_unweighted", limit, None), ("clipped_routed", limit, routing), ("unclipped_routed", 0, routing)):
        expert.swiglu_limit = cap
        y = expert.forward(x, weights)
        g = gate.clamp(max=cap) if cap else gate
        u = up.clamp(min=-cap, max=cap) if cap else up
        hidden = F.silu(g) * u
        if weights is not None:
            hidden *= weights
        l1 = hidden.double().abs() @ matrices["w2"].double().abs().T
        if not torch.isfinite(y).all():
            raise ValueError("nonfinite expert reference")
        report["cases"].append(dict(name=name, limit=cap, routing=([1]*6 if weights is None else weights.flatten().tolist()),
                                    output=y.flatten().tolist(), output_l1=l1.flatten().tolist()))
    with out.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"Validated 3 full expert matrices; clipping coverage: {clipping}")
    print(f"Saved {out}; float32 Expert.forward, no activation quantization")


if __name__ == "__main__":
    main()
