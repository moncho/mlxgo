"""Generate small decoding oracles with PyTorch CPU, never released weights.

FP8 and E8M0 use actual PyTorch dtype conversions. FP4's CPU cast is unsupported
in PyTorch 2.14.0: use the SHA-verified official converter's table and function.
OCP's signed zero is tested separately; the official FP4_TABLE discards it.
No CUDA/TileLang GEMM, activation quantization or checkpoint loading is tested.
"""
import argparse
import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path
import random
import struct

import torch

ROOT = Path(__file__).resolve().parent
REVISION = "df42c109f1defefcbfcedbe7d905718a12266e40"
CONVERT_SHA = "035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=ROOT / "testdata" / "decoding.json.gz")
args = parser.parse_args()
if torch.__version__.split("+", 1)[0] != "2.14.0":
    raise RuntimeError("Use the pinned torch==2.14.0 CPU reference")
torch.set_num_threads(1)
snapshot = json.loads(gzip.decompress((ROOT.parent / "testdata" / "released-checkpoint.metadata.json.gz").read_bytes()))
source = base64.b64decode(snapshot["conversion_source"])
assert snapshot["revision"] == REVISION
assert hashlib.sha256(source).hexdigest() == CONVERT_SHA
nodes = [node for node in ast.parse(source).body
         if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FP4_TABLE" for t in node.targets))
         or (isinstance(node, ast.FunctionDef) and node.name == "cast_e2m1fn_to_e4m3fn")]
assert len(nodes) == 2
ns = {"torch": torch}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "pinned-convert.py", "exec"), ns)


def bits(t):
    return [int(x) & 0xffffffff for x in t.contiguous().float().view(torch.int32).flatten().tolist()]


def canonical_bits(t):
    return [0x7fc00000 if (b & 0x7fffffff) > 0x7f800000 else b for b in bits(t)]


def checksum(t):
    b = canonical_bits(t)
    return hashlib.sha256(struct.pack("<" + "I" * len(b), *b)).hexdigest()


def encoded(t):
    return base64.b64encode(bytes(t.contiguous().view(torch.uint8).flatten().tolist())).decode()


def float_bits(b):
    return torch.tensor(b, dtype=torch.int64).to(torch.int32).view(torch.float32)


byte_values = torch.arange(256, dtype=torch.uint8)
fp8 = byte_values.view(torch.float8_e4m3fn).float()
scale = byte_values.view(torch.float8_e8m0fnu).float()
# OCP E2M1 includes -0; the DeepSeek FP4_TABLE instead uses +0 at index 8.
fp4_ocp = ns["FP4_TABLE"].clone()
fp4_ocp[8] = -0.0
scaled_fp8 = scale[:, None] * fp8[None, :]
scaled_fp4 = scale[:, None] * fp4_ocp[None, :]
# Exhaust every upper 16-bit pattern just below, at and above an RN-even tie.
ties = ((torch.arange(65536, dtype=torch.int64)[:, None] << 16)
        | torch.tensor([0x7fff, 0x8000, 0x8001])[None, :]).flatten()
tie_values = ties.to(torch.int32).view(torch.float32)
rng = random.Random(9217)
rounding_inputs = [0, 0x80000000, 1, 0x80000001, 0x7f7fffff, 0xff7fffff,
                   0x7f800000, 0xff800000, 0x7f800001, 0xff800001,
                   0x3f807fff, 0x3f808000, 0x3f808001, 0x3f818000]
rounding_inputs += [rng.getrandbits(32) for _ in range(1024)]

cases = []


def matrix(name, fmt, rows, cols, data, scales, reference):
    for rounding in ("float32", "bfloat16"):
        out = reference if rounding == "float32" else reference.bfloat16().float()
        assert torch.isfinite(out).all(), name
        case = dict(name=name + "_" + rounding, format=fmt, rows=rows, cols=cols,
                    rounding=rounding, data=encoded(data), scales=encoded(scales),
                    expected_bits=bits(out))
        if name != "subnormal_rows":
            x = torch.zeros(2, cols)
            x[0, 0], x[0, -1], x[1, 31] = 1, .5, -2
            case.update(input_bits=bits(x), projection_bits=bits(x @ out.T))
        cases.append(case)


def fp8_case(name, rows, cols, row_scales=False, tiny=False):
    data = ((torch.arange(rows * cols) * 29 + 3) % 256).to(torch.uint8).reshape(rows, cols)
    data[data == 127] = 126
    data[data == 255] = 254
    # Include both signs of zero and the smallest positive/negative subnormal.
    data.flatten()[:4] = torch.tensor([0, 128, 1, 129], dtype=torch.uint8)
    shape = (rows if row_scales else (rows + 31) // 32, (cols + 31) // 32)
    codes = torch.tensor([0] if tiny else [121, 128, 124, 133, 126, 129], dtype=torch.uint8)
    scales = codes[torch.arange(shape[0] * shape[1]) % len(codes)].reshape(shape)
    expanded = scales.view(torch.float8_e8m0fnu).float().repeat_interleave(32, -1)
    if not row_scales:
        expanded = expanded.repeat_interleave(32, 0)
    result = data.view(torch.float8_e4m3fn).float() * expanded[:rows, :cols]
    matrix(name, "fp8_row32" if row_scales else "fp8_block32", rows, cols, data, scales, result)


fp8_case("rectangular_blocks", 64, 96)
fp8_case("partial_edge_blocks", 33, 35)
fp8_case("engram_rows", 3, 64, row_scales=True)
fp8_case("subnormal_rows", 1, 32, row_scales=True, tiny=True)

rows, cols = 32, 64
packed = (torch.arange(rows * cols // 2) % 256).to(torch.uint8).reshape(rows, cols // 2)
scales = (125 + (torch.arange(rows * cols // 32) % 3)).to(torch.uint8).reshape(rows, cols // 32)
lo, hi = packed & 15, packed >> 4
unpacked = torch.stack([ns["FP4_TABLE"][lo.long()], ns["FP4_TABLE"][hi.long()]], -1).flatten(-2)
decoded = unpacked * scales.view(torch.float8_e8m0fnu).float().repeat_interleave(32, -1)
matrix("all_packed_bytes", "fp4_row32", rows, cols, packed, scales, decoded)

# Execute the unmodified official converter only on well-conditioned scales.
# Its advertised losslessness is NOT assumed for arbitrary exponent spreads.
converted, converted_scales = ns["cast_e2m1fn_to_e4m3fn"](packed.view(torch.int8), scales.view(torch.float8_e8m0fnu))
expanded = converted_scales.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
via_fp8 = converted.float() * expanded
assert torch.equal(via_fp8, decoded)
matrix("official_fp4_to_fp8", "fp8_block32", rows, cols, converted, converted_scales, via_fp8)

fixture = dict(
    torch_version=torch.__version__, source_revision=REVISION, conversion_sha256=CONVERT_SHA,
    reference="PyTorch CPU dtype casts and pinned DeepSeek FP4 conversion, not CUDA/TileLang kernels",
    fp4_signed_zero="OCP negative zero preserved; official FP4_TABLE uses positive zero",
    fp8_bits=bits(fp8), scale_bits=bits(scale), fp4_bits=bits(fp4_ocp),
    fp8_scaled_sha256=checksum(scaled_fp8), fp8_scaled_bf16_sha256=checksum(scaled_fp8.bfloat16().float()),
    fp4_scaled_sha256=checksum(scaled_fp4), fp4_scaled_bf16_sha256=checksum(scaled_fp4.bfloat16().float()),
    bf16_ties_sha256=checksum(tie_values.bfloat16().float()),
    rounding_inputs=rounding_inputs, rounding_outputs=bits(float_bits(rounding_inputs).bfloat16().float()),
    cases=cases,
)
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_bytes(gzip.compress((json.dumps(fixture, indent=2) + "\n").encode(), mtime=0))
print(f"Wrote {len(cases)} matrix cases, exhaustive code/scale checks and 196608 BF16 tie cases to {args.out}")
