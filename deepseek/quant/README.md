# Bounded Weight Decoding

`deepseek/quant` is a pure-Go CPU reference decoder for the storage layouts
identified by the [released-checkpoint audit](../CHECKPOINT_AUDIT.md). It does
not load checkpoints, download weights, quantize activations or implement
quantized GPU matrix multiplication. The released DeepSeek model is still
unsupported by `inference.Open`.

## Download Real Validation Samples

From the repository root (no MLX installation or Python required):

```sh
mkdir -p models
go run ./cmd/fetch-deepseek-sample -out models/deepseek-v41-sample
```

This fetches **42,478,080 payload bytes** (42.5 MB), plus a 258,144-byte header,
from the pinned official DeepSeek-V4.1-Flash revision above. It downloads these
three layer-0 tensors and their matching `.scale` tensors from shard 3:

- `layers.0.attn.wkv.weight`: FP8 block32, logical shape `[512,5120]`.
- `layers.0.ffn.experts.0.w1.weight`: FP4 row32, logical shape `[2304,5120]`;
  on-disk I8 shape `[2304,2560]` packs two FP4 values per byte.
- `layers.0.attn.wo_a.weight`: FP8 block32, logical shape `[8192,4096]`, with
  BF16 rounding in the reference conversion.

The output contains six raw `.bin` files and `manifest.json`, not a safetensors
checkpoint. The manifest records storage shapes, absolute source byte offsets
(end-exclusive), lengths, revision, ETag and SHA-256 hashes of the received files.
Those payload hashes support later local integrity checks; they are not upstream
published checksums or proof of numerical correctness.

The downloader checks the header against our audited SHA-256, checks exact HTTP
206 ranges and a stable shard size/ETag, and limits payload requests to 4 MiB.
A server ignoring Range is rejected before its body is read. Existing output
directories are refused; failed downloads are cleaned up, not resumed. Rerun
after a failure. The output's parent directory must already exist.

These are real pretrained parameter samples for decoder validation, **not a
model that can generate text**. No full shard or Engram table is downloaded.

The same downloader accepts `-set expert` for all three layer-0 expert-0
weight/scale pairs (18,800,640 bytes). `-reuse models/deepseek-v41-sample` copies
the already downloaded `w1` pair after checking provenance, layout, length and
SHA-256, so only 12,533,760 new payload bytes are fetched. It never modifies the
reuse directory. See [complete expert validation](../EXPERT_VALIDATION.md).

`-set attention` fetches all layer-0 attention tensors and the input norm
(126,753,280 bytes). Reusing the initial samples reduces new payload to
90,542,080 bytes. This selection has a bounded 128 MiB download allowance,
including its header; the metadata-only audit retains its 64 MiB default.
See [real attention validation](../ATTENTION_VALIDATION.md).

## Validate the Downloaded Samples

Use a Python 3.12 environment with `torch==2.14.0`, separate from application
dependencies. From the repository root:

```sh
python3.12 -m venv /tmp/mlxgo-sample-reference-venv
/tmp/mlxgo-sample-reference-venv/bin/pip install torch==2.14.0
/tmp/mlxgo-sample-reference-venv/bin/python deepseek/quant/make_sample_reference.py \
  --samples models/deepseek-v41-sample

export MLXGO_DEEPSEEK_SAMPLE_DIR="$PWD/models/deepseek-v41-sample"
go test ./deepseek/quant -run TestReleasedSample -v -count=1
go test -tags "mlx mlxruntime" ./deepseek/quant -run TestReleasedSample -v -count=1
```

The Python step runs offline once PyTorch is installed. It checks all six input
hashes and their layouts against the pinned audit header, then produces a local
`reference.json` using PyTorch CPU casts, the unmodified official FP4 converter,
and the official `wo_a` conversion block. It refuses to overwrite output; use
`--out /path/to/new-reference.json` to reproduce the reference separately.

The Go host test decodes every weight in 32-row chunks and compares SHA-256
hashes of the resulting float32 bytes. Signed zero is canonicalized because the
official FP4 table discards negative zero; all nonzero values must match exactly.
`wo_a` is checked both before and after BF16 rounding.

The native test separately decodes whole matrices on the MLX worker, checks
upload/readback (including BF16 storage for `wo_a`), and runs two dense and one
sparse input through every matrix on both CPU and GPU. `wo_a` preserves all
eight independent groups. Float32 projections are compared with PyTorch float64
accumulation using `abs_error <= 1e-5 + 2e-6 * sum(abs(x_i*w_i))` to account for
different accumulation orders without making cancellation-heavy outputs flaky.
These are decoded-weight float32 operations, not quantized GEMM or activation
quantization tests.

These tests skip unless `MLXGO_DEEPSEEK_SAMPLE_DIR` is set. Once opted in,
missing/corrupt files are failures, not skips. Ordinary CI remains offline and
does not fetch released weights or install PyTorch. Weights and generated
reference data stay under the ignored `models/` directory. See
[the recorded validation results](REAL_WEIGHT_VALIDATION.md).

## API

```go
err := quant.Decode(dst, data, scales, rows, cols, quant.FP8Block32, quant.Float32)
```

For files or other streams, the reusable reader avoids retaining the full
encoded matrix:

```go
values, err := quant.ReadMatrix(weightReader, scaleReader, rows, cols,
    quant.FP4Row32, quant.Float32, 64<<20)
```

The budget covers the returned float32 buffer plus 32-row input scratch buffers
and is checked before allocating or reading. It does **not** include reader
internals, Go allocator overhead, other retained matrices or native MLX copies.
An oversized matrix returns `ErrMemoryBudget`; the reader also rejects truncated,
trailing or nonfinite data and returns no partial result. Readers must contain
exactly one raw tensor each. Caller-supplied dimensions and layout must come
from validated metadata, and callers own integrity checks; `io.TeeReader` can
hash both input streams while decoding. This is not a safetensors loader.

The caller supplies all buffers. `dst` must contain exactly `rows*cols` float32
slots. `data` is a packed row-major matrix and `scales` contains E8M0 bytes.
Dimensions must be positive and their product must fit in an `int`.

| Format | Data bytes | Scale shape |
| --- | --- | --- |
| `FP8Block32` | `rows*cols`, E4M3FN | `[ceil(rows/32),ceil(cols/32)]` |
| `FP8Row32` | `rows*cols`, E4M3FN | `[rows,cols/32]` |
| `FP4Row32` | `rows*cols/2`, E2M1 low nibble then high nibble | `[rows,cols/32]` |

Row-scaled formats require columns divisible by 32. FP8 block edges may be
partial; that generic edge behavior is tested but is not needed by the audited
release. `quant.BFloat16` rounds the scaled result to nearest-even BF16 and
returns the widened float32 value. This models the reference `wo_a` conversion
and Engram row lookup; it is not activation quantization.

No output-sized allocation occurs inside `Decode`. It validates lengths,
encodings and results in a first pass before writing in a second pass. Invalid
calls leave `dst` unchanged. NaN weights/scales and output overflow return
`ErrNonFinite`; finite underflow and signed zeros are retained. Input buffers
must remain immutable during the call, and concurrent calls must use separate
destinations. Successful-call allocation and race tests enforce these contracts.

Decode bounded chunks, not entire large tables. FP8 block chunks must begin on
the original 32-row/column boundaries, with corresponding scale blocks. Row
chunks can start at any row for the row-scaled layouts. Column tiles must be
repacked into contiguous row-major buffers. Chunking does not change total
model memory requirements if callers retain all decoded output.

Scalar helpers `DecodeE4M3`, `DecodeE8M0`, `DecodeE2M1` and `RoundBFloat16` expose
the same arithmetic. Unlike matrix decoding, scalar helpers return NaN/infinity
where appropriate. E8M0 byte 0 means **2^-127**, not zero; byte 255 means NaN.
`DecodeE2M1` reads only the low nibble and preserves the sign of zero.

## Numerical Evidence

The checked-in [fixture](testdata/decoding.json.gz) was generated with
`torch==2.14.0` on CPU. Ordinary Go tests need no Python, network or MLX runtime.

- Every E4M3FN and E8M0 byte and every E2M1 nibble is checked.
- Every FP8/E8M0 and FP4/E8M0 code pair is checked in float32 and BF16, including
  subnormals, NaNs and overflow. NaN payload bits are deliberately unspecified.
- BF16 checks include 196,608 values immediately below, at and above every
  upper-16-bit rounding boundary, plus explicit edge cases and random bit patterns.
- Twelve matrix fixtures cover rectangular 32x32 blocks, partial edge blocks,
  Engram-style rows, subnormal scales, every packed FP4 byte, and the official
  FP4-to-FP8 conversion on well-conditioned scales. Chunked and whole decoding
  agree bitwise. Bad lengths, shapes, NaNs and overflow preserve the destination.
- Native tests upload ordinary decoded weights on CPU/GPU, round-trip BF16
  storage, and compare small linear projections with PyTorch. They do not claim
  Metal subnormal behavior, quantized-GEMM equivalence or full-model parity.
  Host decoding also runs on the MLX worker thread with bitwise agreement,
  including the subnormal fixtures.

**FP4 reference distinction:** PyTorch 2.14.0 does not implement a CPU cast from
`float4_e2m1fn_x2` to float32. The generator instead executes the table and
conversion function extracted from the checksum-verified pinned DeepSeek
`convert.py` stored in the existing audit snapshot. That table turns negative
zero into positive zero. The decoder preserves the OCP sign bit, so comparison
with the official FP4 table is numerical for zero and bitwise for finite nonzero
values. Signed-zero correctness is tested separately against the OCP encoding.
The converter's claimed losslessness is not assumed for arbitrary scale spreads.

References: [OCP microscaling formats](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf),
[PyTorch dtypes](https://docs.pytorch.org/docs/main/tensor_attributes.html),
and the pinned [DeepSeek converter](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/inference/convert.py)
and [Engram lookup](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/inference/model.py).
Upstream reference source is covered by [DeepSeek's MIT license](../THIRD_PARTY_LICENSE).

## Verification

```sh
go test ./deepseek/quant
go test -race ./deepseek/quant
go test -tags "mlx mlxruntime" ./deepseek/quant
```

To regenerate the fixture, use a separate Python 3.12 environment with
`torch==2.14.0`, then run `python deepseek/quant/make_fixture.py`. No model
weights, CUDA or network fetches are needed by the generator. It verifies the
saved official source checksum before executing only the selected definitions.
