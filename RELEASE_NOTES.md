# v0.2.0

Sampling, streaming, and Qwen cache improvements since v0.1.0, with additional
ownership and profiling diagnostics. This remains a pre-1.0 source release;
model integrations are experimental, not a universal checkpoint runtime.

## Migration From v0.1.0

The breaking change is the Qwen cache constructor:

```go
// v0.1.0
cache := qwen2.NewKVCache(config.NumLayers)
```

```go
// v0.2.0: use the actual base weights' dtype.
dtype, err := weights.Embed.DType()
if err != nil {
    return err
}
cache := qwen2.NewKVCache(config, dtype)
defer cache.Close()
```

Cache buffers grow lazily in 256-position blocks. Keep one cache per sequence
and discard it after an error. Applications using `inference.Open`,
`Model.Generate`, or `qwen2.NewSession` need no constructor changes.

Existing `Generate`, `GenerateTokens`, and `lm.Greedy` signatures and successful
greedy token selection are unchanged. Generation can now return partial output
with an error; always check the error before treating a result as complete.
Prompt-mode `cmd/generate` writes text incrementally, followed by its existing
timing line. Raw-token mode still prints a final token list.

## Added

- `lm.SamplingOptions`, `lm.Sample`, `Model.GenerateWith`, and
  `Model.GenerateTokensWith`: temperature, top-p filtering, and an MLX seed.
  Temperature zero uses the greedy path; top-p zero or one disables filtering.
  Options must be finite and in range. CLI flags are `-temperature`, `-top-p`,
  and `-seed`; an explicit `-top-p` requires positive temperature.
- `lm.Stream` and `Model.Stream`: caller-goroutine callbacks, partial results
  on callback errors, and incremental text decoding that withholds incomplete
  UTF-8 suffixes until completion or the final flush. Text output omits EOS;
  raw-token generation includes a matching stop token.
- Core `CumsumAxis`, `Slice`, and `SliceUpdate` with native/stub API parity.
- `LiveArrays`: process-wide owned, unclosed Go handle count, including
  idempotent/concurrent close and Qwen session cleanup checks.
- `Synchronize`: wait for submitted work on the selected default stream.
  The allocator regression test now synchronizes its measurement window;
  neither 8 MiB assertion was relaxed.

## Improved

- Qwen preallocated KV buffers replace per-step whole-cache concatenation.
  The cache retains the base dtype and sessions evaluate buffers each step.
- Numeric readers use bulk copies after contiguous materialization; boolean
  readers retain element conversion. Returned slices remain Go-owned.
- Empty native output handles are marked closed and removed from live-handle
  accounting on operation failure.
- Homebrew cgo flags are centralized, with custom C/C++ prefix guidance.
- A top-level quickstart and separate training/verification guides make the
  existing supported workflows easier to find.

On the local Apple M3 Pro, the September 27 single-run GPU decode measurements
before/after the cache change were 100.5/101.5 tokens/s at a 48-token budget and
102.2/101.5 at 512 tokens, using the same prompt. These do not demonstrate a
meaningful speedup. They measure forward passes, not end-to-end throughput.

## Installation

After the v0.2.0 tag is published:

```sh
brew install mlx-c
go get github.com/moncho/mlxgo@v0.2.0
go run -tags mlx ./your-app
```

Native support remains Apple Silicon macOS with cgo and the external MLX/mlx-c
libraries. The local validation baseline is Go 1.27.0, macOS 26.7, MLX 0.32.0,
and mlx-c 0.6.0_3. CI covers Go 1.26/1.27 on macOS 15, plus Linux stub builds.
See `COMPATIBILITY.md` for the support boundary and native dependency policy.

## Validation

Local validation on 2026-10-04 passed:

- `make release-check`: stub/native tests and vet, the full native runtime
  suite, and the complete native race suite.
- All five core examples: smoke, linear, MLP, manual-gradient training, and
  autograd training. The real Qwen sampled CLI reproduced identical text in two
  fresh processes with temperature 0.8, top-p 0.9, and seed 42.
- Real Qwen golden verification: 32/32 reference tokens and maximum absolute
  prefill-logit error 0.03125, below the unchanged 0.25 limit.
- Real Qwen common-loader parity and temporary three-shard loading: 32/32
  reference tokens. The 200-token memory gate retained 44.2 -> 45.8 -> 46.0 MiB
  RSS, below the unchanged 64 MiB growth limit.
- DeepSeek's runtime suite with the existing local attention, compressed/shared
  attention, output/query projection, expert, and quantized-cache references
  enabled; the separate quantization suite also used its real activation oracle.

Real adapter-loader parity and saved training-checkpoint artifact comparison
were not enabled for this preparation. Reduced-fixture LoRA/training tests are
part of the runtime suite. Public CI does not download or run these local
real-weight fixtures. Existing numerical thresholds are unchanged.

Updating these files does not publish a tag or GitHub release. The exact
candidate must pass the complete CI matrix before tagging, then the tag
workflow reruns that matrix before publication.

## Known Limits

- Sampling uses MLX's global RNG; interleaved random operations can change a
  seeded sequence. This is not a per-session random generator.
- Native work remains serialized on one OS thread. Stream callbacks run outside
  internal batches, but callers must not wrap them in worker code that blocks
  on another goroutine needing MLX. The default device is process-wide.
- `LiveArrays` is not a native-memory counter. `Synchronize` neither submits
  lazy graphs nor waits for unrelated streams; concurrent callers can still
  affect allocator measurements.
- Qwen support remains the documented standard BF16 architecture and q/v LoRA,
  not arbitrary models or QLoRA. DeepSeek remains experimental component and
  prepared-bundle support, not full released-checkpoint text inference.
- No DeepSeek module split, weight offloading, or quantized training is added.

The release contains Go source, not native libraries or model weights.
`THIRD_PARTY_NOTICES.md` records bundled upstream notices; model and native
dependency licenses apply separately.
