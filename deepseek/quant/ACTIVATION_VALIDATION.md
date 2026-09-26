# Activation and Cache Quantization Reference

Validated on 2026-09-24 on Apple M3 Pro with Go 1.27.0, PyTorch 2.14.0 and
MLX 0.32.0 / mlx-c 0.6.0_3. This adds bounded pure-Go CPU quantization and
dequantization, not Metal kernels or complete quantized attention.
The device-graph extension was validated on 2026-09-25 in the same environment.

## Contract

Three explicit `ActivationFormat` values keep the different scale rules apart:

| Format | Group | Scale computation |
| --- | ---: | --- |
| `FP8Activation32` | 32 | floor absmax at 1e-4, multiply by float32(1/448), ceil to power of two, store E8M0 |
| `FP4Index32` | 32 | floor absmax at 6*2^-126, multiply by float32(1/6), ceil to power of two, store E8M0 |
| `FP4Cache16` | 16 | floor absmax at 6*2^-9, divide by 6, nearest-even E4M3 scale |

Value casts use saturating nearest-even rounding and preserve signed zero.
The E4M3 scale cast also saturates at 448. All-zero groups have nonzero scale
bytes: 105 for FP8/E8M0, 1 for FP4/E8M0, and 1 for FP4/E4M3. FP4 values pack
low nibble first; scale bytes never share storage with the value bytes.

`ActivationLayout` returns exact sizes without allocation. Both quantization
functions use caller-owned buffers and perform validation before writing.
`QuantizeActivation` accepts float32 inputs as supplied; it does not implicitly
round them to BF16. `DequantizeActivation` optionally rounds reconstructed values
to BF16. Callers modeling the pinned kernel's BF16 inputs must round those inputs
first, independently of the output rounding choice.

Dimensions must be positive, nonoverflowing, and row widths divisible by the
group size. Invalid lengths/formats, nonfinite input, invalid scales, NaN codes
and float32 reconstruction overflow return errors without partial output.
Rejecting nonfinite results is a library safety policy, not an assertion about
how upstream CUDA reports such inputs. Successful calls allocate zero buffers;
callers can process independent row-aligned chunks. Inputs must stay immutable;
output buffers must not overlap or be shared by concurrent calls.

## Reference Method

Pinned release: `df42c109f1defefcbfcedbe7d905718a12266e40`.
Pinned `kernel.py` SHA-256:
`1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455`.

The generator verifies the source checksum and executes its unmodified
`fast_log2_ceil`, `fast_pow2` and `fast_round_scale` functions through a small
PyTorch tensor adapter. The quantizer bodies are explicit CPU mathematical
translations. FP8 uses actual PyTorch E4M3/E8M0 casts. PyTorch 2.14.0 lacks an
FP4 CPU cast, so the oracle selects from the E2M1 table by distance, putting
even codes first at ties. Go instead uses an ordered midpoint search.

These are independent implementations of the intended formulas, **not a run
of the CUDA/TileLang quantizers**. Hardware cast/denormal behavior and quantized
GEMM accumulation parity are not established by these checks.

## Coverage and Results

The checked-in synthetic fixture contains 19 cases and 10,240 values. It covers
zeros, both zero signs, normal and BF16-rounded inputs, tiny/subnormal inputs,
every adjacent positive value midpoint with both signs and neighboring float32
values, power-of-two scale boundaries, E4M3 scale ties and scale saturation.
All 6,224 value bytes, 525 scale bytes and reconstructed float32/BF16 bits match
exactly. Row-by-row chunking agrees with whole-matrix encoding.

An optional 8-case extension adds 224,512 values from the existing real-weight
layer-2 oracle: window KV, compressed KV, index keys and one pending normalized
linear input, each with raw-float32 and BF16-rounded variants. All 182,912 value
bytes, 9,096 scale bytes and reconstructions match exactly. These tensors derive
from released weights driven by deterministic inputs; they are not activations
captured from a complete pretrained model run or quantized upstream execution.

Error tests check unchanged outputs, including errors in later groups. Eight
goroutines with independent outputs pass under the race detector. Successful
encode/decode calls measure zero allocations. A five-second fuzz run completed
400,081 inputs without failure.

Native CPU/GPU smoke tests execute the host functions on the MLX worker and
upload reconstructed values through BF16 storage, requiring exact readback.
Denormal-focused cases are excluded from native storage checks, but tested
exactly on the host. This does not move the quantizer onto the GPU. Full stub
and native runtime suites pass with the real-weight/activation checks enabled.

## MLX Device Graphs

`QuantizeActivationArray` and `DequantizeActivationArray` implement the same
three formats using MLX array operations on the selected CPU or GPU. Encoded
values and scales are actual uint8 arrays, with low-nibble-first FP4 packing.
Input and output data never round-trip through Go. Shape/type validation uses
metadata only; evaluation remains lazy. Returned handles are independent of
input/intermediate handles, and the usual caller-owned `Close` contract applies.

The graph uses bounded binary searches through rounding thresholds rather than
an `[elements, 127]` expansion. E8M0 scales use an exact 48-bit integer product,
float32 nearest-even rounding, then ceiling to a power of two. This prevents
compiled fast-math from changing power-boundary rounding. Subnormal inputs are
normalized from their integer significands; E8M0 reconstruction adjusts exponent
bits directly. Both avoid backend flush-to-zero behavior. BF16 output rounding
also uses integer bits and preserves signed zero.

Unlike the eager host functions, lazy graphs cannot return a data-dependent Go
error without synchronization. Nonfinite input or float32 reconstruction overflow
therefore sets that group's scale byte to 255. Decoding invalid scales, NaN value
codes, or overflowing reconstructions produces canonical NaNs. Shape, format,
dtype and closed-handle errors still return synchronously. These APIs explicitly
stop gradients; no quantization-aware-training estimator is implied.

Exact device checks cover all 19 synthetic cases on CPU and GPU, including tiny
and subnormal cases with no exclusions. The optional 8 real-derived cases also
match packed bytes, scales and float32/BF16 reconstruction bits. Compiled encoding
matches all synthetic cases as well. Additional tests cover randomized exponents,
eight concurrent callers, input closure before evaluation, invalid shapes/types,
nonfinite groups, all value codes with representative valid/invalid scale bytes,
and zero gradients. Ordinary runtime CI runs these tests without downloading
weights; the real-derived extension remains opt-in.

```sh
go test -tags "mlx mlxruntime" ./deepseek/quant -run TestActivationArray -count=1
go test -race -tags "mlx mlxruntime" ./deepseek/quant -run TestActivationArray -count=1
```

These are composed reference operations, **not custom fused kernels, quantized
GEMM, CUDA execution parity, or complete quantized attention**. Full-width
intermediates and dispatch overhead remain; packed retained storage is not a
claim of reduced peak memory or improved throughput.

## Reproduce

Use the separate Python 3.12 / torch==2.14.0 environment from `README.md`:

```sh
/tmp/mlxgo-sample-reference-venv/bin/python deepseek/quant/make_activation_fixture.py \
  --out /tmp/activation-reproduced.json.gz
go test ./deepseek/quant -run Activation -count=1
go test -race ./deepseek/quant
```

The generator refuses to overwrite files. `--source-dir /path/to/inference`
uses local `model.py` and `kernel.py`, still checksum-verified. Without it, only
those pinned source files are fetched, never weights.

For the optional real-cache fixture, first reproduce the layer-2 attention
oracle as described in `../COMPRESSED_ATTENTION_VALIDATION.md`, then:

```sh
/tmp/mlxgo-sample-reference-venv/bin/python deepseek/quant/make_activation_fixture.py \
  --attention-reference models/deepseek-v41-attention2/attention-reference.json.gz \
  --out models/deepseek-v41-attention2/activation-reference.json.gz
export MLXGO_DEEPSEEK_ACTIVATION_REFERENCE="$PWD/models/deepseek-v41-attention2/activation-reference.json.gz"
go test ./deepseek/quant -run TestRealCacheQuantization -count=1
go test -tags "mlx mlxruntime" ./deepseek/quant -run TestActivationReconstructionInMLX -count=1
```

Both fixtures regenerated byte-for-byte. Synthetic gzip SHA-256:
`c5d39c874f5c20c439738ae49dcf50a0d6e1f74c351e695d9eb31c2d70ee1113`.
Optional real-cache gzip SHA-256:
`1ef93cdad43d9f321256535d99045d0c8e30396da1c65752e1386d3eb0f6488e`.
Real-derived data remains in ignored `models/`; only the 59,588-byte synthetic
fixture belongs in source control. Ordinary tests require neither Python nor
network access.

## Remaining Work

Attention sessions now offer explicit [packed cache integration](../QUANTIZED_CACHE_VALIDATION.md),
with separate quantized reference comparisons. Their default remains float32.
Quantized projections and BF16/CUDA parity remain future work. Packed storage
alone does not make the released full model fit on this machine.
