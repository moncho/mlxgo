# Opt-in device decode

`SessionOptions{DeviceDecode: true}` selects device-side expert dispatch after
the first successful session step. The initial prompt uses the existing sparse
path, including a one-token prompt. Default sessions and explicit
`SparseExperts` sessions retain their execution behavior.

```go
session, err := model.NewSessionWithOptions(deepseek.SessionOptions{
    DeviceDecode: true,
})
// Check err, then defer session.Close().
tokens, err := lm.Greedy(session, promptIDs, 32)
```

For the common loader and generator, set
`inference.Options.DeepSeek.Session.DeviceDecode = true`. `inference.Open`
prepares the banks while validating session options; subsequent generation
calls reuse them. This does not add support for released DeepSeek checkpoints.

## Storage and lifetime

The first opted-in session prepares one model-owned bank per layer. Each layer
must have all float32 or all FP4 experts, including its shared expert. Storage
may differ between layers. Mixed storage within a layer is rejected before
bank allocation. Float32 activations and existing router/clipping semantics
are preserved; cache quantization is independent of expert dispatch.

Preparation stacks routed projections, retains shared projections, and replaces
the old routed weights with independent views into the evaluated bank buffers.
Existing dense and sparse sessions keep working through those views. The
model publishes the replacement only after all layers are ready; a failure
releases temporary banks/views and leaves the original weights in place.
Original and stacked storage coexist temporarily during preparation, but
there is no duplicate resident routed-weight payload afterward. Closing a
session leaves the banks available to other sessions; model close releases them.
Banks evaluate every routed expert during preparation. The sparse loader can
leave unused float32 expert tensors unevaluated, so a bank can retain more
memory than a partially evaluated sparse model even without duplicate storage.

## Validation

CPU/GPU integration checks compare logits with the existing sparse path using
the existing `2e-5 * (1 + abs(reference))` tolerance and require identical greedy
token sequences. They cover float32, FP4 and different storage across layers,
with ordinary and quantized caches. Dispatch checks distinguish initial prefill
from later decode, including a one-token prompt. Lifecycle checks exercise old
dense/sparse sessions, reuse, concurrent sessions and model close. Rejection and
late preparation failures check cleanup and retry. Synchronized native counters
check that preparation and subsequent generation do not retain a second weight
copy, after evaluating every fallback expert view. The generation benchmark
exposed a lazy indexed-gather copy in the original float32 view helper; it was
replaced with a slice, and the regression test explicitly evaluates the views.

Packed-view tests separately check varying matrices/scales, zero-copy native
allocation, independent ownership, lazy output lifetime and partial-failure
cleanup. Deliberate mutations verify that these assertions detect wrong entry
selection, shallow ownership, leaks and copied storage.

The full native runtime suite, stub tests, native vet, and focused race checks
passed. Integration mutations also detected wrong routing amplitudes, incorrect
prefill/decode selection, omitted model cleanup, ignored preparation failures,
retained original weights and failed-preparation leaks. All mutations were
restored before the recorded measurements.

## Generation measurement method

Measured on 2026-10-21 with an Apple M3 Pro, 18 GiB unified memory, macOS 27.0.1,
Go 1.27.0, MLX 0.32.0 and mlx-c 0.6.0_3.

There is no compatible pretrained MoE checkpoint available for this experiment.
These are deterministic **untrained synthetic complete models**, not pretrained
text-generation or model-quality results. They exercise the actual prepared
bundle loader, `inference.Model.generate`, all model layers and `lm.Stream`
greedy sampling, including token readback and session cleanup.

Each model has four layers, 16 routed experts plus one shared expert per layer,
a 256-token vocabulary, 256-position context, sliding window 64, and compressed
shared attention in layers 1 and 2 (ratio 4). Input/intermediate widths are
128/128 and 512/1024; top-k is 2 or 8. Expert storage is float32 or FP4, decoded
from the same deterministic FP4 values. Other projections and caches are float32.
Prompts contain 16 or 128 token IDs; every call generates 32 tokens without a
stop token. Seed 197 and per-parameter name hashes define the weights.

Sparse and device runs load the same bundle and receive the same prompt. One
model is resident at a time. Three rounds alternate backend order; each measured
generation follows one complete warmup. Load/bank-preparation and warmup times
are recorded separately and excluded from generation latency. Total generation
includes fresh session creation and close. The first token emission measures
time to first token (TTFT); decode throughput is 31 divided by the wall time
between the first and last emissions. This includes sampling and readback,
unlike the generator's forward-only `DecodeSeconds` field.

Every warmup and measured generation must produce the same token IDs across
both backends and rounds. Native stream synchronization occurs outside timing
before memory snapshots. Reported peaks subtract active allocation before model
loading; they include model weights and generation temporaries, but exclude Go
buffers, allocator cache, loading/preparation peaks and process RSS. These are
short, ordered measurements rather than statistical significance estimates.

## Results

Observed medians of three runs per backend. Each total covers one prompt and
32 generated tokens; throughput covers the 31 decode steps after first emission.
Widths are input/intermediate dimensions. All 96 measured generations completed
and matched the paired backend and warmup token sequences.

| Widths | Top-k | Weights | Prompt | Sparse tok/s | Device tok/s | Decode speedup | Sparse total ms | Device total ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128/128 | 2 | float32 | 16 | 113.80 | 124.84 | 1.10× | 282.13 | 258.15 |
| 128/128 | 2 | float32 | 128 | 115.06 | 123.46 | 1.07× | 282.42 | 267.27 |
| 128/128 | 2 | fp4 | 16 | 112.93 | 120.24 | 1.06× | 284.60 | 269.01 |
| 128/128 | 2 | fp4 | 128 | 113.74 | 121.14 | 1.07× | 283.44 | 268.97 |
| 128/128 | 8 | float32 | 16 | 95.13 | 116.06 | 1.22× | 338.23 | 279.75 |
| 128/128 | 8 | float32 | 128 | 96.07 | 114.54 | 1.19× | 341.15 | 286.65 |
| 128/128 | 8 | fp4 | 16 | 95.45 | 117.67 | 1.23× | 336.16 | 279.78 |
| 128/128 | 8 | fp4 | 128 | 91.69 | 115.79 | 1.26× | 353.01 | 286.61 |
| 512/1024 | 2 | float32 | 16 | 114.81 | 122.12 | 1.06× | 284.85 | 270.01 |
| 512/1024 | 2 | float32 | 128 | 115.87 | 126.51 | 1.09× | 287.27 | 265.24 |
| 512/1024 | 2 | fp4 | 16 | 105.93 | 118.74 | 1.12× | 308.85 | 275.89 |
| 512/1024 | 2 | fp4 | 128 | 104.91 | 108.68 | 1.04× | 317.14 | 305.09 |
| 512/1024 | 8 | float32 | 16 | 93.39 | 113.49 | 1.22× | 354.92 | 293.00 |
| 512/1024 | 8 | float32 | 128 | 92.69 | 120.34 | 1.30× | 359.29 | 283.68 |
| 512/1024 | 8 | fp4 | 16 | 89.97 | 113.03 | 1.26× | 364.80 | 291.62 |
| 512/1024 | 8 | fp4 | 128 | 99.35 | 112.95 | 1.14× | 343.97 | 301.99 |

Observed decode throughput improves 3.6–29.8%; total generation time falls
3.8–21.0%. These short runs do not establish significance for the smallest
differences. They support retaining the backend as an opt-in experiment, without
claiming the same gain on a pretrained model or another configuration.

TTFT and peak native allocation remain workload-dependent. The initial prompt
still uses sparse dispatch; there is no consistent TTFT improvement:

| Widths | Top-k | Weights | Prompt | Sparse TTFT ms | Device TTFT ms | Sparse peak MiB | Device peak MiB |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 128/128 | 2 | float32 | 16 | 10.24 | 9.80 | 11.57 | 15.88 |
| 128/128 | 2 | float32 | 128 | 13.45 | 14.10 | 36.93 | 39.62 |
| 128/128 | 2 | fp4 | 16 | 10.07 | 11.45 | 4.49 | 4.40 |
| 128/128 | 2 | fp4 | 128 | 11.60 | 13.62 | 22.44 | 25.71 |
| 128/128 | 8 | float32 | 16 | 12.33 | 12.74 | 17.62 | 17.67 |
| 128/128 | 8 | float32 | 128 | 14.77 | 15.98 | 54.82 | 55.35 |
| 128/128 | 8 | fp4 | 16 | 11.56 | 12.67 | 5.02 | 4.99 |
| 128/128 | 8 | fp4 | 128 | 14.89 | 17.35 | 32.06 | 35.82 |
| 512/1024 | 2 | float32 | 16 | 14.52 | 16.14 | 312.33 | 420.33 |
| 512/1024 | 2 | float32 | 128 | 22.31 | 18.41 | 403.94 | 487.90 |
| 512/1024 | 2 | fp4 | 16 | 13.13 | 14.80 | 62.34 | 62.50 |
| 512/1024 | 2 | fp4 | 128 | 19.82 | 20.44 | 123.05 | 123.04 |
| 512/1024 | 8 | float32 | 16 | 21.92 | 17.78 | 433.19 | 433.06 |
| 512/1024 | 8 | float32 | 128 | 24.83 | 26.07 | 574.42 | 574.42 |
| 512/1024 | 8 | fp4 | 16 | 20.22 | 20.29 | 66.36 | 66.22 |
| 512/1024 | 8 | fp4 | 128 | 32.20 | 27.14 | 193.16 | 186.27 |

For 512/1024 float32 with top-k 2, device banks retain 411.17 MiB, while
sparse execution retains 303.17 MiB for the 16-token prompt case and 327.17 MiB
for the 128-token prompt case, measured after warmup generation. Sparse leaves
some float32 expert tensors lazy;
device preparation evaluates every routed expert. With top-k 8, both retain
411.17 MiB. Large FP4 cases retain approximately 57.39 MiB for either backend.
The post-generation regression test verifies that a fully evaluated model
does not acquire duplicate weight storage. Every benchmark model returned
native active allocation to its pre-load baseline after close.

[Complete raw measurements](testdata/device_decode_m3pro.json) include all
samples, emission timestamps, generated token IDs, separate load/warmup times,
allocation metrics, deterministic configurations and bundle/source SHA-256
hashes. The saved hashes match the source used for these measurements.

## Reproduce

```sh
go test -tags 'mlx mlxruntime' ./deepseek -run '^TestDeviceDecode' -count=1
go test -tags 'mlx mlxruntime' ./deepseek/quant -run 'TestFP4(Bank|Linear)' -count=1
MLX_DEVICE_DECODE_BENCH=1 \
MLX_DEVICE_DECODE_OUT=/tmp/device-decode-new.json \
go test -tags 'mlx mlxruntime' ./inference \
  -run '^TestDeviceDecodeGenerationBenchmark$' -count=1 -v
```

The output path must be new. Optional `MLX_DEVICE_DECODE_ROUNDS` sets 1–20 rounds;
`MLX_DEVICE_DECODE_FILTER` is a regular expression over case names such as
`dim128_inter128/k2/fp4/prompt16`. Run native measurements sequentially with no
other MLX tests or benchmarks from this task active.
