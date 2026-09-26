"""Independent float64 projection oracle for the local real layer-0 wq_b sample.

No network access. Checkpoint bytes remain in the sample directory; only the
generator is committed. This is weight-only FP8, not activation-quantized GEMM.
"""

import argparse
import base64
import gzip
import json
from pathlib import Path
import sys

import torch

from make_sample_reference import (
    HEADER_SHA, REVISION, ROOT, SHARD, decoded_hash, inputs, sha,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or args.samples / "query-projection-reference.json.gz"
    if out.exists():
        raise FileExistsError(out)
    if torch.__version__.split("+", 1)[0] != "2.14.0" or sys.byteorder != "little":
        raise RuntimeError("Use torch==2.14.0 on a little-endian host")
    torch.set_num_threads(1)
    snapshot = json.loads(gzip.decompress((ROOT.parent / "testdata" / "released-checkpoint.metadata.json.gz").read_bytes()))
    prefix = base64.b64decode(snapshot["headers"][SHARD]["prefix"])
    if snapshot["revision"] != REVISION or sha(prefix) != HEADER_SHA:
        raise ValueError("pinned header mismatch")
    header = json.loads(prefix[8:])
    manifest_bytes = (args.samples / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest["revision"] != REVISION or manifest["shard"] != SHARD
            or manifest["header_sha256"] != HEADER_SHA
            or manifest["repository"] != "deepseek-ai/DeepSeek-V4.1-Flash"):
        raise ValueError("sample provenance mismatch")
    entries = {t["name"]: t for t in manifest["tensors"]}
    if len(entries) != len(manifest["tensors"]):
        raise ValueError("duplicate tensors")
    name = "layers.0.attn.wq_b"
    rows, cols = 32768, 1280

    def read(suffix, shape, dtype):
        full = name + suffix
        t, original = entries[full], header[full]
        start, end = original["data_offsets"]
        if (t["file"] != full + ".bin" or t["shape"] != shape
                or t["shape"] != original["shape"] or t["dtype"] != dtype
                or t["dtype"] != original["dtype"] or t["bytes"] != end - start
                or t["source_byte_offsets"] != [len(prefix)+start, len(prefix)+end]):
            raise ValueError("sample layout mismatch: " + full)
        path = args.samples / t["file"]
        if path.stat().st_size != t["bytes"]:
            raise ValueError("sample length mismatch")
        raw = bytearray(path.read_bytes())
        if sha(raw) != t["sha256"]:
            raise ValueError("sample checksum mismatch")
        return torch.frombuffer(raw, dtype=torch.uint8).reshape(shape), t["sha256"]

    raw, data_sha = read(".weight", [rows, cols], "F8_E4M3")
    scales, scales_sha = read(".scale", [rows//32, cols//32], "F8_E8M0")
    w = raw.view(torch.float8_e4m3fn).float()
    w *= scales.view(torch.float8_e8m0fnu).float().repeat_interleave(32, 0).repeat_interleave(32, 1)
    if not torch.isfinite(w).all():
        raise ValueError("nonfinite decoded weights")
    # Three exact binary-fraction input rows, independently reproduced in Go.
    x = inputs(1, cols)[0].double()
    wd = w.double()
    y, l1 = x @ wd.T, x.abs() @ wd.abs().T
    report = dict(schema=1, revision=REVISION, torch_version=torch.__version__,
                  manifest_sha256=sha(manifest_bytes), header_sha256=HEADER_SHA,
                  name=name+".weight", rows=rows, cols=cols,
                  data_sha256=data_sha, scales_sha256=scales_sha,
                  decoded_sha256=decoded_hash(w), input_recipe="sample_three_rows_v1",
                  projection=y.flatten().tolist(), projection_l1=l1.flatten().tolist())
    payload = json.dumps(report, separators=(",", ":"), allow_nan=False).encode()
    compressed = gzip.compress(payload, mtime=0)
    with out.open("xb") as f:
        f.write(compressed)
    print(f"Saved {out}; {rows*cols:,} weights; SHA256 {sha(compressed)}")


if __name__ == "__main__":
    main()
