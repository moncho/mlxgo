# Device-side expert dispatch prototype

`DeviceMoE` runs the existing router, selected routed experts and shared expert
without reading routing indices into Go. It uses MLX `gather_mm` for float32
weights and `gather_qmm` in MXFP4 mode for packed weights. This is a standalone
inference component. The later opt-in session integration is documented in
[device decode validation](DEVICE_DECODE_VALIDATION.md); the measurements below
are the original standalone component benchmark.

## API and scope

```go
bank, err := deepseek.NewDeviceMoE(experts, shared)
if err != nil { return err }
defer bank.Close()

y, err := bank.Forward(x, routerWeight, routerBias, routerConfig, limit)
if err != nil { return err }
defer y.Close()
return mlx.Eval(y)
```

`NewFP4DeviceMoE(experts, shared, dim, inter)` accepts packed `FP4ExpertWeights`
instead. The bank is homogeneous float32 or FP4, including the shared expert;
mixed banks are not supported. Input activations remain float32. Routed weights
are stacked and evaluated once during construction, and caller-owned inputs may
then be closed. Packed construction never retains a decoded matrix. Source and
destination packed buffers coexist temporarily while each projection is stacked.

`Forward` returns a caller-owned lazy `[tokens, dim]` array. It preserves router
scores, selection bias, deterministic tie ordering, normalization, scaling,
asymmetric SwiGLU clipping and routing amplitudes before the down projection.
The shared expert processes every token. Native calls and close operations use
the existing MLX worker; separate goroutines may reuse the bank. Closing a bank
invalidates its copies while existing lazy outputs retain their dependencies.

This implementation uses unsorted one-row gathers for token/expert pairs. It
does not group large token batches per expert, offload weights, add expert
parallelism or introduce custom Metal kernels. The differentiable `MoE` and
host-scheduled `SparseMoE` remain numerical/performance references.

## Validation

The focused tests compare CPU and GPU outputs with the existing dense and
sparse paths, using the existing scaled `2e-5 * (1 + abs(reference))` budget.
They cover non-square expert projections, all router scores, ties, top-k one,
two and all, nonuniform routing, normalization on/off, clipping and the shared
expert. Packed output is compared with both decoded float32 weights and the
existing packed expert implementation.

The ownership tests close constructor inputs, modify the original packed byte
slices, reuse and concurrently call banks, evaluate lazy outputs after close,
and check copied-handle invalidation. Compiled calls with different inputs
exercise device-resident routing. Rejected constructors check owned-handle
counts, including failures after earlier projections have been allocated.

Deliberate mutations confirmed that tests detect wrong expert selection,
incorrect packed transpose, closed optional indices, discarded routing
amplitudes, a host read during compilation, a missing shared expert and leaked
partial constructor allocations. All mutations were restored.

## Measurement method

Measurements use an Apple M3 Pro with 18 GiB unified memory, macOS 27.0.1,
Go 1.27.0, MLX 0.32.0 and mlx-c 0.6.0_3. The benchmark runs on GPU and measures
the complete router-plus-expert call, including dispatch, all three projections,
activation clipping, weighting, shared expert, reduction, output evaluation and
cleanup inside one worker batch. Construction and loading are excluded.

Fixtures have 16 routed experts and one shared expert, with deterministic
synthetic inputs/weights. Float32 weights are decoded from the same FP4 values.
Dimensions are 128/128 and 512/1024, with 1/128 tokens and top-k 2/8. These are
component workloads, not full-model throughput or pretrained-quality evidence.
The measured shapes are also included in GPU numerical comparisons.

Each benchmark leaf owns only its selected implementation. In particular,
packed runs do not retain decoded reference weights, and float32 bank runs close
the source arrays before measurement. Three warmups precede each timed run.
Results use medians of three 300 ms benchmark runs. The benchmark visits dense,
sparse, then device for each shape. The recorded run was interrupted: its 35
complete groups were retained, one incomplete group was rerun, and the final 12
groups were measured in a continuation. The raw output records both continuation
commands. These short, ordered measurements describe observed performance; they
do not establish statistical significance. No other native test or benchmark
from this task runs alongside the measurement. The default stream is explicitly synchronized before the
baseline, after warmup and after timing, before reading native counters. These
fences are outside the timed loop. A focused repeat confirmed identical retained
allocation across nine samples after adding these fences; peak allocations can
still vary between runs, so the report uses medians.

`peak-active-bytes` subtracts the pre-existing active MLX allocation from the
peak. `steady-active-bytes` measures retained native allocation after warmup;
`working-peak-bytes` is peak minus steady. These counters exclude Go buffers,
allocator cache, construction peaks and process RSS. `weight-bytes` is the
logical payload size, not an allocator measurement.

## Results

Observed median latency in milliseconds per complete MoE call. Speedup is
`sparse / device`; values below 1 mean the device path is slower. Dim/inter
are the input and intermediate projection widths. Each case has 16 routed
experts plus one shared expert.

| Dim / inter | Tokens | Top-k | Weights | Dense ms | Sparse ms | Device ms | vs sparse |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 128 / 128 | 1 | 2 | float32 | 1.040 | 0.520 | 0.339 | 1.54× |
| 128 / 128 | 1 | 2 | fp4 | 1.192 | 0.531 | 0.360 | 1.47× |
| 128 / 128 | 1 | 8 | float32 | 1.159 | 0.840 | 0.394 | 2.13× |
| 128 / 128 | 1 | 8 | fp4 | 1.270 | 0.865 | 0.409 | 2.11× |
| 128 / 128 | 128 | 2 | float32 | 1.337 | 1.101 | 0.522 | 2.11× |
| 128 / 128 | 128 | 2 | fp4 | 1.435 | 1.120 | 0.636 | 1.76× |
| 128 / 128 | 128 | 8 | float32 | 1.458 | 1.456 | 0.989 | 1.47× |
| 128 / 128 | 128 | 8 | fp4 | 1.534 | 1.524 | 1.420 | 1.07× |
| 512 / 1024 | 1 | 2 | float32 | 1.501 | 0.689 | 0.501 | 1.38× |
| 512 / 1024 | 1 | 2 | fp4 | 1.185 | 0.558 | 0.384 | 1.45× |
| 512 / 1024 | 1 | 8 | float32 | 1.566 | 1.262 | 0.902 | 1.40× |
| 512 / 1024 | 1 | 8 | fp4 | 1.328 | 0.900 | 0.487 | 1.85× |
| 512 / 1024 | 128 | 2 | float32 | 4.621 | 2.629 | 3.392 | 0.78× |
| 512 / 1024 | 128 | 2 | fp4 | 4.756 | 2.426 | 2.034 | 1.19× |
| 512 / 1024 | 128 | 8 | float32 | 4.677 | 3.962 | 13.882 | 0.29× |
| 512 / 1024 | 128 | 8 | fp4 | 4.744 | 3.714 | 6.061 | 0.61× |

The device path is 1.38–2.13× faster than the existing sparse path for every
one-token case measured. At 128/128 dimensions and 128 tokens, the observed
gain is 1.07–2.11×; the 1.07× result is small enough to warrant longer,
interleaved measurements before treating it as a reliable advantage.

At 512/1024 dimensions and 128 tokens, results diverge. Top-k 2 FP4 is 1.19×
faster, but float32 takes 29% longer. With top-k 8, FP4 takes 63% longer and
float32 takes 3.50× as long as the existing sparse path. This prototype is a
candidate for decode dispatch, not a general replacement for sparse dispatch.
A next experiment would group selected token/expert pairs on device before
matrix multiplication. The current one-row gathers and the existing sparse
path use different batching; these timings do not isolate the individual
costs of synchronization, scheduling, memory traffic or kernels.

Retained native allocation is identical between sparse and device dispatch
in all measured cases. Peak allocation is workload-dependent. For the larger
128-token workloads, median baseline-subtracted active peaks are:

| Weights | Top-k | Sparse peak MiB | Device peak MiB |
| --- | ---: | ---: | ---: |
| float32 | 2 | 111.80 | 114.48 |
| fp4 | 2 | 57.71 | 34.49 |
| float32 | 8 | 129.46 | 118.96 |
| fp4 | 8 | 117.12 | 36.00 |

The top-k 8 FP4 device path has a lower measured peak (36.00 versus 117.12 MiB)
while also being slower. The float32 top-k 2 device peak is higher. Memory and
latency should therefore be evaluated separately; neither is a universal win.

[Raw measurements and source hashes](testdata/device_moe_m3pro.txt) contain all
144 samples (48 cases, three samples each), including allocation metrics and
continuation commands. Numerical comparisons, the full native runtime suite,
stub tests, vet and focused race checks passed. Model defaults remain unchanged;
the subsequent session integration requires explicit `DeviceDecode` selection.

## Reproduce

```sh
go test ./...
go vet ./...
go test -tags 'mlx mlxruntime' . -run '^TestRuntimeGather' -count=1
go test -tags 'mlx mlxruntime' ./deepseek -run '^TestDeviceMoE' -count=1
go test -race -tags 'mlx mlxruntime' . ./deepseek \
  -run 'Test(RuntimeGather|DeviceMoE)' -count=1
go test -tags 'mlx mlxruntime' ./deepseek -run '^$' \
  -bench '^BenchmarkDeviceMoE$' -benchtime=300ms -count=3
```

Native execution requires Metal access. Run benchmark commands sequentially,
without concurrent MLX work affecting the allocator counters.
