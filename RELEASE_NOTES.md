# v0.1.0

First versioned release of mlxgo, Go bindings for MLX via mlx-c on Apple Silicon.
This is a pre-1.0 library release, not a stable universal model runtime.

## Included

- CPU/Metal arrays, shape and math operations, autograd, optimizers, compilation,
  safetensors I/O, explicit ownership, dedicated-thread execution and batching.
- Qwen2.5-0.5B inference and q/v LoRA fine-tuning, including resumable training,
  adapter validation, a Go tokenizer and local single-file/sharded loading.
- Experimental DeepSeek components with independently checked real-weight
  samples, packed FP8/FP4 projections, quantized caches, opt-in sparse expert
  scheduling and prepared packed-bundle loading.
- Honest stub builds, native/stub API parity tests, CPU/GPU runtime regressions,
  race checks and runnable smoke examples.

## Stabilization

- Fix `Full(..., Float32)` and other GPU-compatible fill dtypes by converting
  the scalar on CPU before broadcasting on the selected device. Integer fills
  do not lose precision through a float32 intermediate. Float64 output remains
  CPU-only. Tests cover dtypes, large integers, special floating values,
  scalar/empty shapes, errors, concurrent callers, compilation and autograd.
- Test Go 1.26 and 1.27 in CI, record native dependency versions, and require
  the complete CI matrix before publishing a GitHub release.
- Document the supported environment, pre-1.0 API policy and known limitations
  in `COMPATIBILITY.md` and the maintainer procedure in `RELEASING.md`.
- Include upstream license/provenance notices for the reduced Qwen tokenizer
  fixture alongside the existing DeepSeek reference notice.

## Installation

```sh
brew install mlx-c
go get github.com/moncho/mlxgo@v0.1.0
go run -tags mlx ./your-app
```

Native execution requires an Apple Silicon Mac, cgo, and the Homebrew mlx-c
headers/libraries under `/opt/homebrew`. The local validation baseline is
Go 1.27.0, macOS 26.7, MLX 0.32.0 and mlx-c 0.6.0_3. CI also exercises macOS 15
and the Go 1.26/1.27 lines. Homebrew dependencies are external, not vendored.

## Limits

- APIs are pre-1.0; higher-level model and quantization packages are experimental.
- This is not a loader for arbitrary model architectures or the full released
  DeepSeek-V4.1 checkpoint. DeepSeek samples do not produce pretrained text.
- Native calls share a worker thread. Callbacks and `Batch` bodies must not wait
  for another goroutine that needs MLX. Sparse scheduling performs host readback
  and is inference-only; dense execution remains the default.
- Packed weights do not imply faster CPU execution, quantized training,
  weight offloading or a guarantee that a checkpoint fits available memory.

The release distributes Go source, not native libraries, model weights or
platform binaries. Individual model licenses and native dependencies apply
separately from this repository's MIT license; see `THIRD_PARTY_NOTICES.md`.
