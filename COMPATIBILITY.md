# Compatibility And Stability

## v0.2.0 Support Boundary

| Area | Supported or validated boundary |
| --- | --- |
| Native hardware | Apple Silicon (`darwin/arm64`), CPU and Metal GPU index 0 |
| macOS | Documented minimum 14; release validation on macOS 15 in CI and 26.7 locally. macOS 14 is not separately exercised in CI. |
| Go | Minimum 1.26 from `go.mod`; CI covers the 1.26 and 1.27 lines |
| Native dependencies | Local tested baseline: MLX 0.32.0, mlx-c 0.6.0_3, Homebrew `/opt/homebrew` |
| Build | `CGO_ENABLED=1`, `-tags mlx`; runtime tests additionally need `mlxruntime` |
| Without MLX | Stub APIs compile and return unavailable errors; this is not CPU emulation. Linux stub builds run in CI. |

Homebrew dependencies are not vendored or pinned by the Go module. CI installs
the available `mlx-c` formula and records the actual MLX/mlx-c versions, making
upstream changes visible. That rolling check does not certify every past or
future native release. Pin your native environment to a validated combination
for deployment and rerun the runtime suite after upgrades. Other installation
prefixes, Intel Macs, Linux native execution and additional GPU devices are not
part of this release's supported configuration.

## API Policy

`v0.2.0` remains pre-1.0. It is not a promise of v1 API stability or production
suitability for every workload. Within each patch line (such as `v0.2.x`),
the intent is to preserve exported signatures and defaults, documenting
behavioral corrections and security fixes. Breaking changes target a new
minor version and must be described in the release notes. Do not move published
tags; fixes receive new versions. Applications should pin a module version.

Upgrading from v0.1.0 requires direct cache users to replace
`qwen2.NewKVCache(layers)` with `qwen2.NewKVCache(config, weightsDType)`.
Sessions and the common inference loader make this change internally. The
[release notes](RELEASE_NOTES.md#migration-from-v010) cover this signature and
the generation error/streaming behavior changes.

The root array/autograd/optimizer APIs are the reusable foundation. `qwen2`,
`inference`, `deepseek`, and `deepseek/quant` remain experimental and narrow:
support for an architecture name does not imply support for every checkpoint,
quantization scheme, attention variant, tokenizer or training setup in its family.
The test suite checks exported native/stub signature parity.

## Operational Limits

- Own and close arrays explicitly. Copies share close state; returned lazy
  outputs retain native dependencies, but closing a Go handle invalidates all
  copies of that handle.
- Native work is confined to one OS-thread worker. Independent callers may use
  it concurrently; this does not mean native dispatch runs in parallel. `Batch`
  amortizes dispatch overhead. Never wait inside a callback or batch for another
  goroutine doing MLX work; that forms a deadlock.
- Default-device selection is process-wide, not per session. Select it before
  starting a workload; independent concurrent requests cannot choose unrelated
  per-request defaults. Backend dtype support still applies: Float64 is CPU-only.
- `Full` converts its float64 argument to the requested dtype on CPU using MLX
  before broadcasting. Its input can only express values representable by Go
  float64; use typed constructors for exact integers outside that range.
- Qwen inference and LoRA support the documented standard BF16 configuration,
  not arbitrary models or QLoRA. Qwen training/model state must follow the
  documented ownership and no-concurrent-mutation rules.
- DeepSeek sparse execution and prepared packed loading are explicit options.
  Sparse execution is not compatible with autograd or `Compile`, and all expert
  weights remain resident. The prepared loader's byte limit is not an RSS limit.
- CPU/GPU numerical results may differ. Quantized validation uses documented
  tolerances; component benchmarks are not whole-model throughput claims.
- Sampling seeds MLX's process-wide RNG. Reproducibility assumes no interleaved
  random operations from other callers; a seed is not per-session isolation.
- Stream callbacks run on the caller outside internal batches. Do not wrap
  streaming in a worker callback or `Batch` if the consumer waits for MLX work
  on another goroutine. Callback errors return partial output, not a completion.
- `LiveArrays` counts owned, unclosed Go handles, not native graph buffers or
  bytes. `Synchronize` drains the selected default stream, not all streams;
  submit lazy work first with `Eval` or `AsyncEval` before profiling.

## Validation

Release tags run the same CI matrix as development: stub tests/vet, native
build/vet, CPU/GPU runtime tests, race regressions and smoke programs on Go
1.26/1.27. Native tests need Metal access; sandboxed/headless processes can fail
MLX initialization even before selecting CPU.

Real-weight tests are opt-in because model weights are not included in the
repository. A green public CI run therefore does not mean those tests ran.
The v0.2.0 validation record is maintained in `RELEASE_NOTES.md`. Historical
v0.1.0 validation on 2026-09-27 included `make release-check` and the real
Qwen and DeepSeek sample tests. Later cache validation matched all 32 Qwen
reference tokens with maximum prefill-logit error 0.03125 (limit 0.25), and
the 200-token RSS gate passed without changing its 64 MiB growth limit.
These are bounded regression checks, not a general model-quality guarantee.

No test suite proves the absence of bugs. Report failures with module version,
Go version, macOS/CPU model, MLX/mlx-c versions, build tags, selected device and
a minimal reproducer. Avoid including private data, credentials or model weights.
