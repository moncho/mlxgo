# Native Packed FP4 Expert Execution

The library now exposes `mlx.MXFP4Matmul`, `quant.NewFP4Linear` and
`deepseek.NewFP4Expert`. `ModelOptions.FP4Experts` explicitly selects routed or
shared experts by `ExpertID{Layer, Index}`; `Index: -1` means shared. All three
selected parameter names (`w1`, `w2`, `w3`) must be omitted from the float32 map.
Selected and ordinary experts may coexist, with independent FP8 attention and
quantized-cache options. Defaults and checkpoint loaders are unchanged.

## Storage And Execution

The checkpoint's row-group32 E2M1 layout matches MLX's weight-only `mxfp4`
mode: eight nibbles per UInt32 word, earlier elements in lower bits, with
UInt8 E8M0 scale codes. The adapter copies bytes without requantization or
FP4-to-FP8 expansion and validates finite decoded values in 32-value chunks.
No full float32 matrix is built or retained. Dimensions must be positive
multiples of 32. See the pinned [MLX operation](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/ops.cpp)
and [CPU implementation](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/cpu/quantized.cpp).

Activations remain float32. SwiGLU clips gate only from above and up on both
sides; routing is applied before the down projection. The ordinary and packed
expert paths share that activation implementation. This is not DeepSeek's
activation-quantized GEMM or BF16 arithmetic. Native underflow, accumulation and
signed-zero behavior follow MLX; no bitwise equivalence for every exponent is claimed.

Constructors copy caller buffers and clean up partial allocations on error.
Copies share handles, Close is idempotent, lazy outputs survive close, and
concurrent calls serialize on the MLX worker. The APIs are inference-only.

## Independent Real-Weight Checks

The existing pinned PyTorch 2.14.0 references are reused, with manifest, source,
matrix and scale hashes checked before upload. No new weights are downloaded.

The standalone real `w1` projection (`[2304,5120]`) passes its unchanged
`1e-5 + 2e-6 * L1` error budget on CPU/GPU for batches 1, 3, 32 and 128.
Observed errors were zero for these exact binary-fraction input rows. The
three canonical rows repeat for larger batches; this tests kernel batching,
not additional activation diversity or universal exactness.

The [complete expert oracle](EXPERT_VALIDATION.md) checks all three real
matrices, six distinct inputs including large activations and a zero input,
clipped/unclipped execution and nonuniform/zero routing. The unchanged full
expert budget is `3e-5 + 2e-6 * L1` per output.

| Backend | Case | Maximum absolute error | Maximum error / max(1, L1) |
| --- | --- | ---: | ---: |
| CPU | Clipped, unweighted | 3.357e-4 | 3.384e-7 |
| CPU | Clipped, routed | 2.442e-4 | 3.384e-7 |
| CPU | Unclipped, routed | 1.075e-2 | 3.384e-7 |
| GPU | Clipped, unweighted | 2.747e-4 | 3.436e-7 |
| GPU | Clipped, routed | 2.289e-4 | 3.436e-7 |
| GPU | Unclipped, routed | 1.075e-2 | 3.436e-7 |

Zero-input and zero-routing rows remain exactly zero. Absolute errors on the
large unclipped inputs should not be read as the small projection errors;
nonlinear propagation and cancellation are included in these full-expert checks.

Download-free tests cover all nibble values, distinct low/high nibble ordering,
row scales, strided inputs, compiled projections/experts, copied buffers,
lazy lifetime, invalid dimensions/dtypes/scales, partial-construction cleanup,
shared/routed and mixed FP4/float32 models, FP8 attention, packed caches,
compressed sharing, and eight concurrent sessions. CI runs native tests and
race checks. Public native/stub signatures remain enforced by the parity test.

## Measured Performance

Apple M3 Pro, MLX 0.32.0, mlx-c 0.6.0_3, Go 1.27. Medians of three
300 ms benchmark runs after warmup; the slow CPU batch-128 packed case has
only one timed iteration per run. These measurements cover one real expert's
three projections, clipping at 10, routing at 0.75, evaluation and output close
inside a worker batch. Upload, validation and input construction are excluded.
The canonical six input rows repeat for batch 128. This is an expert benchmark,
not whole-model throughput or a realistic distribution of routed token batches.

| Backend | Tokens | Float32 (ms) | Packed FP4 (ms) |
| --- | ---: | ---: | ---: |
| CPU | 1 | 1.545 | 15.728 |
| CPU | 128 | 10.235 | 2002.160 |
| GPU | 1 | 1.294 | 0.368 |
| GPU | 128 | 2.968 | 2.794 |

All three float32 matrices occupy 141,557,760 bytes. Packed data and scales
occupy 18,800,640 bytes, saving 122,757,120 bytes (86.7%) without a resident
float32 copy. Measured peak active native allocation, including inputs,
intermediates and outputs, was:

| Backend | Tokens | Float32 (MB) | Packed FP4 (MB) |
| --- | ---: | ---: | ---: |
| CPU | 1 | 141.672 | 18.989 |
| CPU | 128 | 150.340 | 27.657 |
| GPU | 1 | 141.651 | 18.918 |
| GPU | 128 | 155.059 | 27.608 |

MB is decimal. These are process-global MLX allocator counters, not process
RSS or total device memory; lazy evaluation and allocator reuse can affect
them. The structural weight-byte counts above are deterministic.

On this machine, GPU single-token execution is about 3.5 times faster, while
batch 128 is only about 6% faster. CPU execution is about 10 times slower for
one token and 196 times slower at batch 128. Packing is therefore opt-in;
reduced storage does not imply faster execution on every backend or workload.
The benchmark uploads routing directly as float32 because the existing
`mlx.Full(..., Float32)` helper constructs a float64 scalar that GPU rejects;
that separate helper issue is not fixed by this change.

## Reproduce

```sh
export MLXGO_DEEPSEEK_SAMPLE_DIR="$PWD/models/deepseek-v41-sample"
export MLXGO_DEEPSEEK_EXPERT_DIR="$PWD/models/deepseek-v41-expert0"
go test -tags 'mlx mlxruntime' . -run '^TestRuntimeMXFP'
go test -tags 'mlx mlxruntime' ./deepseek/quant ./deepseek \
  -run '^(TestReleasedFP4|TestFP4)' -v
go test -race -tags 'mlx mlxruntime' ./deepseek/quant ./deepseek -run '^TestFP4'
go test -tags 'mlx mlxruntime' ./deepseek -run '^$' \
  -bench '^BenchmarkReleasedFP4Expert$' -benchtime=300ms -count=3
```

Full released-model loading, pretrained tokenizer integration, efficient sparse
expert dispatch and activation-quantized kernels remain separate work. In
particular, the model still evaluates every expert and masks contributions;
packed storage does not fix that scaling cost or prove whole-model feasibility.
