# Sparse Expert Dispatch And Prepared Packed Loading

## Scope

`SessionOptions.SparseExperts` is an opt-in inference scheduler for ordinary,
FP4 and mixed experts. Default sessions and the differentiable `MoE` remain
dense. `SparseMoE` exposes the scheduler for standalone float32 experts.

Routing still uses the existing scores, correction bias, normalization, scale,
and deterministic tie ordering. The scheduler reads only expert indices to Go,
groups selected token rows, executes nonempty expert batches, restores the
original token/top-k order, and sums the contributions. Routing weights remain
on device and are applied before the down projection. Every token still goes
through the shared expert. There is no full tokens-by-experts output mask.

This is not fused GPU dispatch, weight offloading, a training path, or a
`Compile`-compatible operation. The host scheduling boundary synchronizes once
per layer. All expert weights remain resident. Sparse evaluation can change
floating-point summation order; no bitwise equality is claimed.

`deepseek.Load` and `inference.Options.DeepSeek` now read explicit selections
from prepared U8 safetensors bundles. See the [layout and Go API](README.md#sparse-execution-and-packed-loading).
The loader reuses the existing single-file/indexed reader, validates the exact
tensor inventory and a configurable payload limit before allocation, visits
shards in stable order, and retains no decoded float32 duplicate for packed
weights. Loading can temporarily hold shard arrays, packed Go buffers and
native packed copies; its byte limit is not a peak RSS limit.

This does **not** add full released-checkpoint loading, native F8 safetensors
decoding, pretrained tokenization, activation-quantized GEMM, or a packed-bundle
CLI. The Go API removes manual byte reads; selections remain explicit.

## Correctness

Download-free CPU/GPU tests cover:

- Token order restoration, unused/empty expert batches, top-k 1/2/all,
  repeated expert selections across tokens, and rejection of duplicates within
  one token or out-of-range indices.
- Zero and nonuniform routing weights, the shared expert, all three router
  score functions, normalization and deterministic score ties.
- Float32, FP4 and mixed expert execution against the existing dense path,
  with a scaled `2e-5 * (1+abs(reference))` component budget.
- Prepared single-file and three-shard bundles with mixed FP4 experts and
  FP8 KV/query/grouped-output projections. Full prefill and cached rollout
  are compared with manual construction, including compressed sharing,
  packed/unpacked caches, rollover and eight concurrent sessions.
- Common-loader generation against a decoded baseline, both single-file and
  sharded; option mutations after opening do not change the loaded model.
- Incorrect shapes/dtypes, missing/extra tensors, duplicate selections,
  nonfinite scale values, byte-limit failures and wrong-architecture options.
- Contiguous, transposed, broadcast, empty, wrong-dtype and closed UInt8
  readback; returned Go bytes are independent copies.

The existing real-weight projection, expert and combined attention suites
remain regression checks. Sparse routing tests and benchmarks use reduced
synthetic weights, not a full released expert bank or real model logits.

## Measurements

Apple M3 Pro, MLX 0.32.0, mlx-c 0.6.0_3, Go 1.27. Medians of three 300 ms
runs after three warmups, with no concurrent inference workload. Each timed
step includes the router, index evaluation/readback for sparse execution,
expert scheduling, all three expert projections, clipping, routing weights,
shared expert, output evaluation and cleanup inside a worker batch.

The fixture uses 16 routed experts plus one shared expert, dimension and
intermediate width 128, top-k 2, sigmoid scores and normalized weights. Inputs
and weights are deterministic synthetic data. One token selects two experts;
128 tokens select eleven different experts across the batch. Mixed storage
packs even-numbered routed experts and leaves the shared expert float32.

Milliseconds per complete router-plus-expert call:

| Backend | Storage | Tokens | Dense | Sparse |
| --- | --- | ---: | ---: | ---: |
| CPU | Float32 | 1 | 0.798 | 0.209 |
| CPU | Float32 | 128 | 1.202 | 0.645 |
| CPU | Mixed | 1 | 0.857 | 0.225 |
| CPU | Mixed | 128 | 23.781 | 3.951 |
| CPU | FP4 | 1 | 0.918 | 0.251 |
| CPU | FP4 | 128 | 49.828 | 9.116 |
| GPU | Float32 | 1 | 1.134 | 0.597 |
| GPU | Float32 | 128 | 1.539 | 1.098 |
| GPU | Mixed | 1 | 1.220 | 0.602 |
| GPU | Mixed | 128 | 1.590 | 1.147 |
| GPU | FP4 | 1 | 1.433 | 0.653 |
| GPU | FP4 | 128 | 1.662 | 1.121 |

Median peak active MLX allocation at 128 tokens (decimal MB):

| Backend | Storage | Dense | Sparse |
| --- | --- | ---: | ---: |
| CPU | Float32 | 5.335 | 4.374 |
| CPU | Mixed | 5.610 | 4.670 |
| CPU | FP4 | 5.900 | 4.974 |
| GPU | Float32 | 14.683 | 6.270 |
| GPU | Mixed | 18.037 | 6.882 |
| GPU | FP4 | 21.899 | 7.677 |

These are absolute process-wide native allocator peaks, not RSS or isolated
session memory. The fixture retains decoded reference weights even in packed
modes; compare dense versus sparse **within a storage mode**, not the weight
storage efficiency between modes. A production model does not retain those
reference duplicates. The benchmark emits single-token peaks too; they are
dominated by persistent fixture weights and differ little between dispatch modes.

This reduced workload favors sparse dispatch, but the gains are not a claim
about large-model throughput. CPU packed prefill is still much slower than
float32 despite avoiding unused work. Larger top-k, small expert counts and
dense batches can make synchronization/gather overhead outweigh savings.
Defaults therefore remain dense. No pretrained text quality, full-model
feasibility, or long-context throughput follows from these measurements.

## Reproduce

```sh
go test ./...
go test -tags 'mlx mlxruntime' . ./deepseek ./inference \
  -run 'Test(RuntimeUInt8|Sparse|PackedLoad|PackedDeepSeek|LoadSelections)'
go test -race -tags 'mlx mlxruntime' . ./deepseek ./inference \
  -run 'Test(RuntimeUInt8|Sparse|PackedLoad|PackedDeepSeek)'
go test -tags 'mlx mlxruntime' ./deepseek -run '^$' \
  -bench '^BenchmarkSparseDispatch$' -benchtime=300ms -count=3
```

CI runs native runtime tests and the sparse/model-loader race regressions.
The real-weight suites remain opt-in using the existing local sample paths;
no downloads are performed by these tests.
