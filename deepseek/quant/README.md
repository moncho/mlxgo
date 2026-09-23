# Bounded Weight Decoding

`deepseek/quant` is a pure-Go CPU reference decoder for the storage layouts
identified by the [released-checkpoint audit](../CHECKPOINT_AUDIT.md). It does
not load checkpoints, download weights, quantize activations or implement
quantized GPU matrix multiplication. The released DeepSeek model is still
unsupported by `inference.Open`.

## API

```go
err := quant.Decode(dst, data, scales, rows, cols, quant.FP8Block32, quant.Float32)
```

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
