# Packed Grouped Output Projection

`ModelOptions.FP8AttentionOA` explicitly selects packed `layers.N.attn.wo_a.weight`
for a layer. It is independent of the KV/query options. Float32 remains the
default. This is a weight-only inference path, not released-checkpoint loading,
activation-quantized GEMM, BF16/CUDA parity or pretrained generation.

## Layout And Conversion

The real layer-0 matrix has shape `[8192,4096]`, split into eight contiguous
groups of `[1024,4096]`. Each group consumes only its corresponding attention
heads. The implementation uploads eight independent packed matrices, projects
each group's inputs and concatenates results in the original group order. It
does not treat the tensor as one ordinary dense projection. No float32 weight
duplicate is retained; validation uses bounded scratch space.

The pinned upstream conversion materializes `wo_a` as BF16 before execution.
For this sample, all 33,554,432 weights are unchanged by that conversion:
both raw decoded float32 and official BF16-widened hashes are
`f543cf52cc28521dc6d128f13af21073678cd3af2c8b6ace695d291c7ccbf6dc`.

This is verified for every weight, not inferred from a few outputs. The public
constructor rejects any value whose float32 bits change under nearest-even
BF16 rounding, including underflow, plus nonfinite values and overflow. It
does not approximate a rounding-changing checkpoint or silently choose a
float32 fallback. The restriction is deliberate: other samples must satisfy
the same guard before using this option. Native accumulation/underflow still
follow MLX; this does not establish bit-exact upstream arithmetic.

Group boundaries must align with 32x32 scale blocks: `ORank` and the per-group
input width must be positive multiples of 32. Ownership, copied caller buffers,
constructor cleanup and model-close behavior match the KV/query options.

## Independent Reference

The existing [real-weight oracle](quant/REAL_WEIGHT_VALIDATION.md) already
executes the checksum-pinned upstream `wo_a` conversion and computes grouped
float64 projections in PyTorch 2.14.0. No new weight download or oracle is needed.
Tests verify its revision, source/manifest/payload hashes, shapes and decoded
weight hashes. Three group-specific binary-fraction input rows repeat over
batches of 1, 3, 32 and 128 tokens. Repetition exercises batch kernels, not
additional input diversity.

The original projection budget remains `abs(error) <= 1e-5 + 2e-6 * L1`, where
L1 is the sum of absolute products from the independent reference.

| Backend | Maximum absolute error | Maximum error / max(1, L1) |
| --- | ---: | ---: |
| CPU | 2.757e-7 | 6.982e-9 |
| GPU | 3.949e-7 | 1.130e-8 |

## Real Attention

Output-only and combined KV/query/output selections pass the unchanged real
layer-0 attention oracle: full 131-token prefill, 1/127/128/129-token prefills
with two decode steps, sliding-window rollover, chronological caches, causality,
cached/full agreement and interleaved sessions. Caches remain float32 for these
real-weight checks. Error budgets remain `2e-4` for outputs and `5e-5` for caches,
scaled by `1 + abs(reference)`.

| Mode | Backend | Worst scaled output error | Worst scaled cache error |
| --- | --- | ---: | ---: |
| Output only | CPU | 1.976e-6 | 1.809e-6 |
| Output only | GPU | 2.642e-6 | 1.925e-6 |
| KV + query + output | CPU | 2.886e-6 | 1.822e-6 |
| KV + query + output | GPU | 2.660e-6 | 1.839e-6 |

Download-free tests exercise all seven nonempty option combinations, ordinary
and quantized caches, mixed packed/float32 layers, ownership, malformed input,
BF16 rejection and eight concurrent sessions. Synthetic logit differences from
decoded float32 were below `4.77e-7`. Additional group-isolation tests ensure an
inactive group stays zero and lazy outputs survive closing the model. CI runs
the model/group tests under the race detector. Real compressed-layer and
packed-weight/quantized-cache combinations still require separate validation.

## Whole-Attention Measurements

Apple M3 Pro, MLX 0.32.0, mlx-c 0.6.0_3, Go 1.27.0. Medians of three 300ms
runs with three warmups per case and no concurrent inference benchmark. The
CPU packed prefill cases exceed 300ms, so each has one timed iteration per run.
These are directional component measurements, not full-model tokens/second.

Timing includes input RMSNorm, all query/KV/output projections, RoPE, attention,
cache updates, evaluation and session cleanup within one `mlx.Batch`. Validation,
uploads and Go readback are excluded. Prefill starts empty and consumes 128
tokens; decode clones a pre-evaluated 128-token cache and consumes one token
including rollover. The template prefill is excluded, cloning is timed equally.
Inputs are deterministic synthetic activations applied to real weights.

| Backend | Workload | Float32 | Packed KV + query | Packed output | All three packed |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU | 128-token prefill | 76.502 ms | 972.998 ms | 747.875 ms | 1634.639 ms |
| CPU | Decode after 128 tokens | 5.920 ms | 10.986 ms | 10.077 ms | 15.240 ms |
| GPU | 128-token prefill | 12.482 ms | 12.765 ms | 13.077 ms | 13.596 ms |
| GPU | Decode after 128 tokens | 4.501 ms | 3.460 ms | 3.781 ms | 2.797 ms |

All-three GPU decode took **38% less time than float32** (1.61x throughput),
or 19% less than KV/query packing alone in this run. Its range was 2.708-2.889 ms,
versus 4.483-4.543 ms for float32 and 3.436-3.625 ms for packed KV/query.
GPU prefill took about 9% longer than float32. CPU all-three prefill was about
21.4x slower and decode 2.6x slower. Keep selection explicit; do not generalize
the decode gain to training, prefill or whole-model performance.

### Memory

`wo_a` alone falls from 134,217,728 float32 bytes to 34,603,008 packed bytes,
including expanded row scales: **99,614,720 bytes (99.61 MB) saved**. Total
attention weight payload is 506,490,112 bytes for float32, 374,189,312 for packed
KV/query, 406,875,392 for packed output only, and **274,574,592 for all three**.
The combined saving is **231.92 MB (45.79%)**. These are array payloads, not RSS.

Median peak active MLX allocator bytes, in decimal MB:

| Backend | Workload | Float32 | Packed KV + query | Packed output | All three packed |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU | 128-token prefill | 699.920 | 566.571 | 588.247 | 455.946 |
| CPU | Decode after 128 tokens | 511.186 | 378.885 | 411.465 | 279.214 |
| GPU | 128-token prefill | 691.599 | 559.298 | 625.539 | 493.238 |
| GPU | Decode after 128 tokens | 511.652 | 379.351 | 412.072 | 279.771 |

The grouped GPU path retains more temporary storage, so its prefill peak does
not fall by the full weight-payload saving. Post-cleanup steady snapshots also
vary with native resource release: output-only GPU prefill ranged from
409.509 to 433.364 MB; all-three ranged from 277.208 to 301.063 MB. Do not
interpret a single steady reading as a leak or a stable retained-weight count.
Peak counters reset after warmup and subtract active allocations present before
loading the sample. They include weights, inputs, caches and temporaries, but
exclude idle allocator cache, Go buffers, loading-time peaks and process RSS.
Counters are process-wide; no other model was intentionally active per case.

## Reproduce

Use the existing local samples and references from the
[sample](quant/REAL_WEIGHT_VALIDATION.md) and [attention](ATTENTION_VALIDATION.md)
reports. These tests never download weights.

```sh
export MLXGO_DEEPSEEK_SAMPLE_DIR="$PWD/models/deepseek-v41-sample"
export MLXGO_DEEPSEEK_ATTENTION_DIR="$PWD/models/deepseek-v41-attention0"
go test -tags 'mlx mlxruntime' ./deepseek \
  -run '^TestReleasedFP8Output(Projection|AttentionForward)$' -v
go test -race -tags 'mlx mlxruntime' ./deepseek -run '^Test(FP8Model|FP8Output)'
go test -tags 'mlx mlxruntime' ./deepseek -run '^$' \
  -bench '^BenchmarkReleasedAttentionFP8$/(cpu|gpu)/(prefill128|decode128)/(float32|kv_qb|oa|kv_qb_oa)$' \
  -benchtime=300ms -count=3
```

Specs and downloaded weights remain outside committed source changes.
