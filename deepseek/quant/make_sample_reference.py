"""Validate local released samples and generate independent PyTorch CPU oracles.

No network access or checkpoint execution. Only the checksum-pinned FP4 table,
conversion function and wo_a conversion block are extracted from the source. Output
contains hashes and projection results, not copies of the model weights.
"""

import argparse
import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
REVISION = "df42c109f1defefcbfcedbe7d905718a12266e40"
CONVERT_SHA = "035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76"
HEADER_SHA = "ff66dd94d7eb6ef5cc1457b2ac13b422c14e9891914c786af995edaa4f10a614"
SHARD = "model-00003-of-00048.safetensors"
SPECS = [
    ("layers.0.attn.wkv", 512, 5120, 1, "fp8_block32", "float32"),
    ("layers.0.ffn.experts.0.w1", 2304, 5120, 1, "fp4_row32", "float32"),
    ("layers.0.attn.wo_a", 8192, 4096, 8, "fp8_block32", "bfloat16"),
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def decoded_hash(tensor):
    # Chunk serialization avoids a Python list for the entire 134 MB wo_a.
    h = hashlib.sha256()
    for chunk in tensor.split(32):
        chunk = chunk.float().contiguous().clone()
        chunk[chunk == 0] = 0.0  # Official FP4_TABLE canonicalizes negative zero.
        h.update(bytes(chunk.view(torch.uint8).flatten().tolist()))
    return h.hexdigest()


def inputs(groups, cols):
    # Binary fractions are identical in Go, float32 and float64.
    x = torch.empty(groups, 3, cols, dtype=torch.float32)
    for g in range(groups):
        col = torch.arange(cols)
        x[g, 0] = ((col * 37 + g * 17) % 257 - 128).float() / 128
        x[g, 1] = ((col * 13 + g * 29) % 127 - 63).float() / 64
        x[g, 2] = 0
        x[g, 2, 0], x[g, 2, 31], x[g, 2, -1] = 1, -2, 0.5
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="new output path; default: <samples>/reference.json")
    args = parser.parse_args()
    out = args.out or args.samples / "reference.json"
    if out.exists():
        raise FileExistsError(out)
    if torch.__version__.split("+", 1)[0] != "2.14.0" or sys.byteorder != "little":
        raise RuntimeError("Use torch==2.14.0 on a little-endian host")
    torch.set_num_threads(1)
    snapshot = json.loads(gzip.decompress((ROOT.parent / "testdata" / "released-checkpoint.metadata.json.gz").read_bytes()))
    source = base64.b64decode(snapshot["conversion_source"])
    prefix = base64.b64decode(snapshot["headers"][SHARD]["prefix"])
    if snapshot["revision"] != REVISION or sha(source) != CONVERT_SHA or sha(prefix) != HEADER_SHA:
        raise ValueError("pinned audit source/header mismatch")
    header = json.loads(prefix[8:])
    nodes = [node for node in ast.parse(source).body
             if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FP4_TABLE" for t in node.targets))
             or (isinstance(node, ast.FunctionDef) and node.name == "cast_e2m1fn_to_e4m3fn")]
    if len(nodes) != 2:
        raise ValueError("unexpected reference definitions")
    ns = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "pinned-convert.py", "exec"), ns)
    wo_blocks = [node for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.If) and isinstance(node.test, ast.Call)
                 and isinstance(node.test.func, ast.Attribute) and node.test.func.attr == "endswith"
                 and len(node.test.args) == 1 and isinstance(node.test.args[0], ast.Constant)
                 and node.test.args[0].value == "wo_a.weight"]
    if len(wo_blocks) != 1:
        raise ValueError("unexpected wo_a conversion block")
    wo_code = compile(ast.Module(body=wo_blocks[0].body, type_ignores=[]), "pinned-wo_a-conversion.py", "exec")
    manifest_bytes = (args.samples / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest["revision"] != REVISION or manifest["shard"] != SHARD
            or manifest["header_sha256"] != HEADER_SHA
            or manifest["repository"] != "deepseek-ai/DeepSeek-V4.1-Flash"):
        raise ValueError("sample provenance mismatch")
    entries = {t["name"]: t for t in manifest["tensors"]}
    expected_names = {name + suffix for name, *_ in SPECS for suffix in (".weight", ".scale")}
    if set(entries) != expected_names or len(manifest["tensors"]) != 6:
        raise ValueError("unexpected/duplicate samples")

    def read(name):
        t, original = entries[name], header[name]
        start, end = original["data_offsets"]
        if (t["file"] != name + ".bin" or t["shape"] != original["shape"]
                or t["dtype"] != original["dtype"] or t["bytes"] != end - start
                or t["source_byte_offsets"] != [len(prefix) + start, len(prefix) + end]):
            raise ValueError("sample layout mismatch: " + name)
        path = args.samples / t["file"]
        if path.stat().st_size != t["bytes"]:
            raise ValueError("sample length mismatch: " + name)
        b = bytearray(path.read_bytes())
        if sha(b) != t["sha256"]:
            raise ValueError("sample checksum mismatch: " + name)
        return torch.frombuffer(b, dtype=torch.uint8).reshape(t["shape"]), t["sha256"]

    report = dict(schema=1, revision=REVISION, torch_version=torch.__version__,
                  conversion_sha256=CONVERT_SHA, manifest_sha256=sha(manifest_bytes),
                  zero_policy="canonicalize signed zero for decoded hashes", cases=[])
    for name, rows, cols, groups, fmt, rounding in SPECS:
        packed, data_sha = read(name + ".weight")
        scales, scales_sha = read(name + ".scale")
        expanded = scales.view(torch.float8_e8m0fnu).float().repeat_interleave(32, 1)
        converter = None
        if fmt == "fp4_row32":
            lo, hi = packed & 15, packed >> 4
            unpacked = torch.stack([ns["FP4_TABLE"][lo.long()], ns["FP4_TABLE"][hi.long()]], -1).flatten(-2)
            decoded = unpacked * expanded
            converted, converted_scales = ns["cast_e2m1fn_to_e4m3fn"](packed.view(torch.int8), scales.view(torch.float8_e8m0fnu))
            via_fp8 = converted.float() * converted_scales.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
            if not torch.isfinite(via_fp8).all():
                raise ValueError("nonfinite official FP4 conversion")
            converter = dict(mismatches=int((via_fp8 != decoded).sum()),
                             max_abs_error=float((via_fp8 - decoded).abs().max()),
                             decoded_sha256=decoded_hash(via_fp8))
        else:
            decoded = packed.view(torch.float8_e4m3fn).float() * expanded.repeat_interleave(32, 0)
        if list(decoded.shape) != [rows, cols] or not torch.isfinite(decoded).all():
            raise ValueError("invalid decoded weights: " + name)
        float32_sha = decoded_hash(decoded)
        if rounding == "bfloat16":
            state = {name + ".weight": packed.view(torch.float8_e4m3fn),
                     name + ".scale": scales.view(torch.float8_e8m0fnu)}
            exec(wo_code, {"name": name + ".weight", "i": 0, "state_dicts": [state]})
            official = state[name + ".weight"].float()
            if not torch.equal(official, decoded.bfloat16().float()):
                raise ValueError("official wo_a conversion differs from expanded block scaling")
            decoded = official
        weight = decoded.reshape(groups, rows // groups, cols)
        x = inputs(groups, cols)
        # Float64 accumulation gives an independent, tighter comparison target
        # than requiring different float32 GEMM implementations to match bits.
        projection = x.double() @ weight.double().transpose(1, 2)
        l1 = x.double().abs() @ weight.double().abs().transpose(1, 2)
        fp32_projection = x @ weight.transpose(1, 2)
        case = dict(name=name, rows=rows, cols=cols, groups=groups, format=fmt, rounding=rounding,
                    data_sha256=data_sha, scales_sha256=scales_sha, float32_sha256=float32_sha,
                    decoded_sha256=decoded_hash(decoded), projection=projection.flatten().tolist(),
                    projection_l1=l1.flatten().tolist(),
                    torch_float32_max_abs_error=float((fp32_projection.double() - projection).abs().max()),
                    official_fp4_conversion=converter)
        report["cases"].append(case)
        print(f"{name}: {rows * cols:,} weights; official FP4 conversion: {converter}", flush=True)
    with out.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
