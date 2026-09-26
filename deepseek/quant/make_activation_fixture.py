"""CPU activation-quantization oracle, not a CUDA/TileLang execution.

Execute the pinned bit-scale helpers with a small PyTorch adapter; translate
the two quantizer bodies into tensor operations. FP8 casts use PyTorch, while
FP4 CPU casts are unavailable and use an independent nearest-even table search.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reference import HASHES, REVISION, load_sources

ROOT = Path(__file__).resolve().parent


class TensorLanguage:
    @staticmethod
    def reinterpret(dtype, x):
        if dtype == "uint32":
            return x.float().view(torch.int32).long() & 0xffffffff
        return x.to(torch.int32).view(torch.float32)

    @staticmethod
    def Cast(dtype, x):
        assert dtype == "int32"
        return x.to(torch.int32)

    if_then_else = staticmethod(torch.where)


def bits(x):
    return [int(v) & 0xffffffff for v in x.contiguous().float().view(torch.int32).flatten().tolist()]


def encoded(x):
    return base64.b64encode(bytes(x.contiguous().view(torch.uint8).flatten().tolist())).decode()


def activation_quantizer(source):
    """Return an independent CPU encoder and reconstruction, not CUDA execution."""
    if isinstance(source, str):
        source = source.encode()
    if hashlib.sha256(source).hexdigest() != HASHES["kernel.py"]:
        raise ValueError("kernel source checksum mismatch")
    names = {"fast_log2_ceil", "fast_pow2", "fast_round_scale"}
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == 3
    ns = {"T": TensorLanguage}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "pinned-scale-helpers.py", "exec"), ns)
    fp4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
    even_first = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])

    def encode(x, fmt):
        x = x.contiguous().float()
        rows, cols = x.shape
        group = 16 if fmt == "fp4_cache16" else 32
        a = x.reshape(rows, -1, group)
        amax = a.abs().amax(-1, keepdim=True)
        if fmt == "fp8_activation32":
            scale = ns["fast_round_scale"](amax.clamp_min(1e-4), 1/448)
            stored_scale = scale.to(torch.float8_e8m0fnu)
            q = (a / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            data = q.reshape(rows, cols)
            y = q.float()*scale
        else:
            if fmt == "fp4_cache16":
                stored_scale = (amax.clamp_min(6*2**-9)/6).to(torch.float8_e4m3fn)
                scale = stored_scale.float()
            else:
                scale = ns["fast_round_scale"](amax.clamp_min(6*2**-126), 1/6)
                stored_scale = scale.to(torch.float8_e8m0fnu)
            v = (a/scale).clamp(-6,6)
            distance = (v.abs()[...,None] - fp4[even_first]).abs()
            codes = even_first[distance.argmin(-1)].to(torch.uint8)
            values = fp4[codes.long()]
            values = torch.copysign(values,v)
            codes |= torch.signbit(v).to(torch.uint8)*8
            codes = codes.reshape(rows,cols)
            data = codes[:,0::2] | (codes[:,1::2]<<4)
            y = values*scale
        return data, stored_scale.reshape(rows, -1), y.reshape(rows, cols)
    return encode


def inplace_cache_quantizers(source):
    encode = activation_quantizer(source)

    def apply(x, fmt):
        assert x.dtype == torch.float32
        _, _, y = encode(x.reshape(-1, x.shape[-1]), fmt)
        x.copy_(y.reshape_as(x))

    def fp8(x, block_size, scale_fmt, scale_dtype, inplace):
        assert block_size == 32 and scale_fmt == "ue8m0" and inplace
        apply(x, "fp8_activation32")

    def fp4(x, block_size, inplace, scale_dtype=None):
        assert inplace
        if scale_dtype == torch.float8_e4m3fn:
            assert block_size == 16
            apply(x, "fp4_cache16")
        else:
            assert block_size == 32
            apply(x, "fp4_index32")
    return fp8, fp4


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path)
    p.add_argument("--attention-reference", type=Path, help="optional real layer-2 float32 attention oracle")
    p.add_argument("--out", type=Path, default=ROOT / "testdata" / "activation.json.gz")
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if torch.__version__.split("+", 1)[0] != "2.14.0" or sys.byteorder != "little":
        raise RuntimeError("Use torch==2.14.0 on a little-endian host")
    torch.set_num_threads(1)
    encode = activation_quantizer(load_sources(args.source_dir)["kernel.py"])
    cases = []
    fp4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)

    def quantize(name, x, fmt):
        data, stored_scale, y = encode(x, fmt)
        rows, cols = x.shape
        assert torch.isfinite(x).all() and torch.isfinite(y).all() and torch.isfinite(y.bfloat16()).all(), name
        cases.append(dict(name=name, format=fmt, rows=rows, cols=cols, input_bits=bits(x),
                          data=encoded(data), scales=encoded(stored_scale), expected_bits=bits(y),
                          bf16_bits=bits(y.bfloat16().float())))

    def anchored(values, group, anchor):
        rows = (values.numel()+group-2)//(group-1)
        x = torch.zeros(rows,group)
        for i,v in enumerate(values):
            x[i//(group-1),i%(group-1)] = v
        x[:,-1] = anchor
        return x

    formats = ["fp8_activation32", "fp4_index32", "fp4_cache16"]
    for fmt in formats:
        group = 16 if fmt == "fp4_cache16" else 32
        z = torch.zeros(2,group*2)
        z[:,1::2] = -0.
        quantize(fmt+"_zeros",z,fmt)
        x = ((torch.arange(3*64).reshape(3,64)*37)%257-128).float()/13
        x *= torch.tensor([1/128,1,8])[:,None]
        quantize(fmt+"_float32",x,fmt)
        quantize(fmt+"_bf16_input",x.bfloat16().float(),fmt)
        tiny = torch.tensor([0.,-0.,2**-149,-2**-149,2**-126,-2**-126]*((group+5)//6))[:group]
        quantize(fmt+"_tiny",tiny[None,:],fmt)
        levels = torch.arange(127,dtype=torch.uint8).view(torch.float8_e4m3fn).float() if fmt == "fp8_activation32" else fp4
        mid = (levels[:-1]+levels[1:])/2
        v = torch.stack([torch.nextafter(mid,torch.full_like(mid,-torch.inf)),mid,
                         torch.nextafter(mid,torch.full_like(mid,torch.inf))]).flatten()
        v = torch.cat([v,-v])
        quantize(fmt+"_value_ties",anchored(v,group,448 if fmt=="fp8_activation32" else 6),fmt)
        if fmt == "fp4_cache16":
            levels = torch.arange(1,127,dtype=torch.uint8).view(torch.float8_e4m3fn).float()
            maxima = (levels[:-1]+levels[1:])*3
        else:
            exponents = [-120,-40,-20,-10,-1,0,1,10,80]
            maxima = torch.tensor([(448 if fmt=="fp8_activation32" else 6)*2.**e for e in exponents])
        peaks = torch.stack([torch.nextafter(maxima,torch.zeros_like(maxima)),maxima,
                             torch.nextafter(maxima,torch.full_like(maxima,torch.inf))]).flatten()
        rows = peaks[:,None]*torch.linspace(-1,1,group)[None,:]
        quantize(fmt+"_scale_boundaries",rows,fmt)
    quantize("fp4_cache16_scale_saturation",torch.tensor([2688.,2784.,1e10])[:,None]*torch.linspace(-1,1,16)[None,:],"fp4_cache16")

    provenance = {}
    if args.attention_reference:
        raw = args.attention_reference.read_bytes()
        r = json.loads(gzip.decompress(raw))
        if r["revision"] != REVISION or r["model_sha256"] != HASHES["model.py"] or r["layer"] != 2 or r["tokens"] != 131:
            raise ValueError("unexpected real attention reference")
        last = r["cases"][-1]
        for name,key,cols,fmt in [("window","cache",512,"fp8_activation32"),
                                  ("compressed","compressed",512,"fp4_cache16"),
                                  ("index_keys","keys",128,"fp4_index32"),
                                  ("linear_input","pending",5120,"fp8_activation32")]:
            x = torch.tensor(last[key]).reshape(-1,cols)
            quantize("real_"+name,x,fmt)
            quantize("real_"+name+"_bf16_input",x.bfloat16().float(),fmt)
        provenance = dict(attention_reference_sha256=hashlib.sha256(raw).hexdigest(),
                          attention_manifest_sha256=r["manifest_sha256"])
    report = dict(schema=1, torch_version=torch.__version__, source_revision=REVISION,
                  kernel_sha256=HASHES["kernel.py"], reference="CPU mathematical translation; not CUDA parity",
                  **provenance, cases=cases)
    args.out.write_bytes(gzip.compress((json.dumps(report,separators=(",", ":"))+"\n").encode(),mtime=0))
    print(f"Saved {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
