# Released Layer-0 Attention Validation

Validated on 2026-09-24 with Apple M3 Pro, Homebrew MLX 0.32.0 / mlx-c 0.6.0_3
and PyTorch 2.14.0 CPU reference arithmetic. Uses the real DeepSeek-V4.1-Flash
layer-0 weights, including its input norm, with deterministic test activations.
It does not use synthetic weights or claim full-model inference support.

## Coverage

Layer 0 is pure sliding-window attention: 64 heads of dimension 512, 64 rotary
dimensions, eight output groups, query rank 1280, output rank 1024 per group,
and a 128-token window. It has no compressed KV or sparse indexer branch.

The Go test executes the existing `Session.attention` path with real input
normalization, query projection/normalization, KV projection/normalization,
forward RoPE, causal window selection, attention sink, inverse RoPE, grouped
output projection and final projection. It also checks the chronological cache.

The oracle executes checksum-pinned upstream `Attention`, `RMSNorm`, rotary and
window-index code. Quantized linear operations use decoded float32 weights;
the KV quantizer is disabled. The CUDA sparse-attention kernel uses the existing
PyTorch mathematical translation, not a CUDA run. `wo_a` retains the official
BF16 weight rounding before widening to float32. Activations stay float32.

## Reproduce

From the repository root, with the Python 3.12 / torch==2.14.0 environment
described in `quant/README.md`:

```sh
go run ./cmd/fetch-deepseek-sample -set attention \
  -reuse models/deepseek-v41-sample -out models/deepseek-v41-attention0

/tmp/mlxgo-sample-reference-venv/bin/python deepseek/make_attention_reference.py \
  --samples models/deepseek-v41-attention0

export MLXGO_DEEPSEEK_ATTENTION_DIR="$PWD/models/deepseek-v41-attention0"
go test ./deepseek -run TestReleasedAttentionWeights -v -count=1
go test -tags "mlx mlxruntime" ./deepseek -run TestReleasedAttentionForward -v -count=1
```

The downloader saves 14 raw tensors: five FP8 matrix/scale pairs, three BF16 norm
vectors and the float32 sink vector. New payload is 90,542,080 bytes with reuse,
126,753,280 bytes without. The new output directory must not already exist.
Existing samples are not changed. The reference generator downloads only the
pinned `model.py` source; `--model-source /path/to/model.py` runs offline and
still checks its SHA-256. Use `--out /path/to/new.json.gz` for separate regeneration.

Matrices use `quant.ReadMatrix` with a 192 MiB numeric-buffer budget per matrix.
Decoded weights alone total 506,490,112 bytes (about 483 MiB); temporary arrays,
MLX copies, test data and Go allocation overhead require additional memory.
This is not a full-checkpoint loader or a process-wide memory limit.

## Results

All **126,622,528** decoded weight values match their reference hashes with
signed zero normalized. Native upload/readback preserves them on CPU and GPU.

Full prefill covers 131 tokens. Four independent schedules prefill 1, 127, 128
or 129 tokens and then decode two individual tokens. Other sessions are
interleaved between steps to check isolation. The oracle's ring cache is
converted to chronological order for comparison with the Go cache.

| Comparison | CPU max absolute error | GPU max absolute error |
| --- | ---: | ---: |
| Full prefill vs reference | 2.2649765e-6 | 4.7683716e-6 |
| Cached schedules vs reference | 2.5033951e-6 | 6.4373016e-6 |
| Go cached vs full prefill | 8.5830688e-6 | 7.8678131e-6 |
| Final chronological KV cache | 2.0563602e-6 | 3.0994415e-6 |

Per device, 2,682,880 outputs and 198,144 cache values are checked against the
reference, plus 2,012,160 cached/full comparisons. The zero first input must
produce exactly zero, guarding against accidental future-token visibility.
Cache length and final position must match exactly. Fixed tolerances are
`2e-4*(1+abs(reference))` for outputs and `5e-5*(1+abs(reference))` for cache
values. The PyTorch oracle also has small cached/full rounding differences.

## CPU Failure Found and Fixed

The original grouped projection used a rank-5 broadcasted matrix-vector product:
`[1,tokens,groups,1,width] @ [groups,width,rank]`. At released dimensions with
131 tokens it produced a CPU bus error or invalid values. Stage-by-stage checks
isolated it to this operation; the GPU path passed before the change.

The attention path now arranges tokens as matrix rows and groups as the batch
axis: `[groups,tokens,width] @ [groups,width,rank]`, then restores token order.
This is mathematically equivalent and avoids the failing CPU execution path.
It does not change the general-purpose `mlx.Matmul` wrapper or repair the native
backend itself.

`TestGroupedAttentionProjectionRealShape` uses synthetic sparse weights at the
same released dimensions, so ordinary native CI needs no network or checkpoint.
The old expression was temporarily substituted into this regression and
reproduced the CPU bus error. The fixed path matches all 1,073,152 expected
outputs exactly on each device. The test remains on the fixed implementation.

## Provenance and Limits

Revision: `df42c109f1defefcbfcedbe7d905718a12266e40`.
Local manifest SHA-256:
`fac2770e930da2690f0fbf3c88b15039ce588e65e4875cd1cfc8ea1770bff70c`.
Reference gzip SHA-256:
`c827944eab158ea8b307415488a26058fdce4a60d4dea7d318f810b059f229e0`.
The reference was regenerated and matched byte-for-byte. These are locally
measured integrity hashes, not upstream-published tensor checksums.

The full stub and native runtime suites pass with all three real-weight sample
sets enabled. Real tests are opt-in; after setting the environment variable,
missing/corrupt files fail rather than skip. Weights and generated references
remain under ignored `models/` directories.

This layer-0 check does not validate released compressed-attention layers, their indexers,
quantized KV caches, dynamic activation quantization, mHC/block integration,
full-model logits or text generation. Those remain separate milestones.

Layer 2 is now covered separately by the
[compressed-attention validation](COMPRESSED_ATTENTION_VALIDATION.md), still
with float32 arithmetic and activation/cache quantizers disabled.
