# MLX From Go

## Quickstart

On an Apple Silicon Mac with Go and the Hugging Face CLI (`hf`) installed,
run from this repository's root:

```sh
brew install mlx-c
hf download Qwen/Qwen2.5-0.5B-Instruct config.json model.safetensors tokenizer.json tokenizer_config.json \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --local-dir models/Qwen2.5-0.5B-Instruct
go run -tags mlx ./cmd/generate -prompt "Explain why the sky is blue in one sentence."
```

Or use the same downloaded model from Go (build with `-tags mlx`):

```go
import (
    "fmt"
    mlx "github.com/moncho/mlxgo"
    "github.com/moncho/mlxgo/inference"
)

func generate() error {
    if err := mlx.SetDefaultGPU(); err != nil { return err }
    model, err := inference.Open("models/Qwen2.5-0.5B-Instruct", inference.Options{})
    if err != nil { return err }
    defer model.Close()
    result, err := model.Generate("Explain why the sky is blue.", 64)
    if err != nil { return err }
    fmt.Println(result.Text)
    return nil
}
```

## Overview

Go bindings for Apple's MLX through the official MLX C bridge, with array,
autograd and optimizer APIs, Qwen2.5-0.5B inference, and LoRA fine-tuning.
The higher-level model packages are experimental and deliberately narrow.

This source tree targets **v0.2.0**, still a pre-1.0 library. Check
[published releases](https://github.com/moncho/mlxgo/releases) for availability. See
[compatibility and API stability](COMPATIBILITY.md),
[release notes](RELEASE_NOTES.md), and the [release procedure](RELEASING.md).

The [DeepSeek experimental package](deepseek/README.md) runs a reduced float32
text backbone with sparse attention, MoE, residual mixing and per-session caches.
It shares an inference interface and greedy decoder (`lm`) with Qwen. It does
**not** yet load or run released DeepSeek-V4.1-Flash checkpoints.
The reduced model also supports float32 Engram with prepared token-history
hashes; its reference tests cover lookup/gating gradients and incremental state.

MLX itself does not expose an official Go API. The supported native path is:

```text
Go -> cgo -> mlx-c -> MLX
```

## Requirements

- Apple Silicon Mac
- macOS 14 or newer
- Go 1.26 or 1.27 with cgo enabled
- Homebrew `mlx-c`

Install the native dependencies:

```sh
brew install mlx-c
```

Install a published Go module version (this command requires the v0.2.0 tag):

```sh
go get github.com/moncho/mlxgo@v0.2.0
```

The cgo binding uses Homebrew's Apple Silicon paths:

- headers: `/opt/homebrew/include`
- library: `/opt/homebrew/lib/libmlxc.dylib`

The local tested native baseline is MLX 0.32.0 with mlx-c 0.6.0_3. Homebrew
dependencies are not bundled or pinned by `go get`; review the compatibility
matrix and rerun runtime tests after native upgrades.

### Non-Homebrew Installs

Homebrew's `mlx-c` does not provide a pkg-config file. Default flags are declared
once in `cgo_flags_mlx.go`. For a compatible installation in another prefix,
set both C and C++ include paths and the linker search path:

```sh
prefix=/path/to/mlx-install
CGO_CFLAGS="-I$prefix/include" CGO_CXXFLAGS="-I$prefix/include" \
CGO_LDFLAGS="-L$prefix/lib -Wl,-rpath,$prefix/lib" \
  go run -tags mlx ./cmd/smoke
```

Environment flags are combined with the in-source defaults (before them in the
tested Go toolchain); missing default directories are harmless. Keep headers
and libraries from the same compatible installation, including MLX's runtime
dependencies. This does not extend the tested platform/version support matrix.

## Run The Smoke Test

The real MLX bindings are behind the `mlx` build tag so that normal Go tooling
still works before the native library is installed.

```sh
CGO_ENABLED=1 go run -tags mlx ./cmd/smoke
```

Expected behavior: the program creates two MLX float32 arrays, adds them, forces
evaluation, and prints the resulting Go slice.

Operations use GPU index 0 by default. Call `SetDefaultCPU` before creating or
running arrays when you want CPU execution instead. `SetDefaultDevice` currently
supports CPU or GPU index 0.

Run the linear-regression loss example:

```sh
CGO_ENABLED=1 go run -tags mlx ./cmd/linear
```

The examples use the higher-level helpers where possible:

```go
predictions, err := mlx.Linear(features, weights, bias)
loss, err := mlx.MSELoss(predictions, labels)
nextParams, err := mlx.SGDWithLearningRate(params, grads, learningRate)
```

Run the small MLP classifier example:

```sh
CGO_ENABLED=1 go run -tags mlx ./cmd/mlp
```

Run the manual-gradient linear-regression training example:

```sh
CGO_ENABLED=1 go run -tags mlx ./cmd/train-linear
```

Run the autograd linear-regression training example:

```sh
CGO_ENABLED=1 go run -tags mlx ./cmd/autograd-linear
```

## Fine-Tune A Tiny MLP

```sh
CGO_ENABLED=1 go run -tags mlx ./cmd/finetune-mlp
# To run on Metal:
CGO_ENABLED=1 go run -tags mlx ./cmd/finetune-mlp -device gpu -out checkpoints/mlp-gpu
```

This self-contained example trains a float32 `1 -> 16 (tanh) -> 1` MLP with
49 parameters. It pretrains on `y = sin(1.5*x)`, saves and closes the model,
then loads the checkpoint and fine-tunes all four parameter tensors on
`y = 0.6*sin(1.5*x) + 0.5`. Both phases use autograd and full-batch SGD,
with each training step wrapped in `Batch`.

There are 64 training inputs in `[-1, 1]` and 63 separate held-out inputs at
their midpoints. A fixed initialization seed makes runs reproducible. The
command prints training loss and held-out MSE before and after adaptation;
it exits with an error unless source and adapted MSE are at most `0.02`,
adaptation reduces target MSE by more than 90%, and reloading the final
checkpoint preserves predictions within `1e-6`.

Weights and architecture metadata are saved to `pretrained.safetensors` and
`finetuned.safetensors` under `-out` (default `checkpoints/finetune-mlp`, ignored
by Git). Existing files at those paths are overwritten. The example defaults
to CPU; the runtime test exercises both CPU and GPU. This demonstrates toy
model fine-tuning, not a pretrained LLM or LoRA training pipeline.

In sandboxed or headless macOS processes, MLX may abort with `No Metal device
available` during library initialization. In that case, run the smoke command
from a normal Terminal session with Metal access.

## Qwen Inference And LoRA

`cmd/generate` uses the [common local loader](inference/README.md). It selects
the architecture from `config.json`; `-device gpu` (default) or `-device cpu`
selects the backend. Applications can use `inference.Open` and `Model.Generate`
without importing an architecture package. Experimental DeepSeek bundles use
the same command with `-tokens`, but still cannot generate pretrained text.

Download the public bf16 checkpoint once (about 1 GB) with the Hugging Face CLI:

```sh
hf download Qwen/Qwen2.5-0.5B-Instruct config.json model.safetensors tokenizer.json tokenizer_config.json \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --local-dir models/Qwen2.5-0.5B-Instruct
go run -tags mlx ./cmd/generate -prompt "Explain why the sky is blue in one sentence."
```

Generation runs entirely in Go through MLX, with GPU by default and optional
CPU selection. It uses the Qwen chat
template, greedy decoding by default, tied embeddings, and a preallocated KV cache.
Cache buffers grow in blocks of 256 positions in the weights' dtype. Direct
cache users construct a cache with `qwen2.NewKVCache(config, dtype)`.
For sampling, add `-temperature 0.8 -top-p 0.9 -seed 42`. Temperature zero
preserves greedy decoding; top-p 0 or 1 disables nucleus filtering. The CLI
requires a positive temperature when top-p is supplied. `lm.Sample`,
`inference.Model.GenerateWith`, and `GenerateTokensWith` accept
`lm.SamplingOptions`. Sampling seeds MLX's global RNG; reproducibility assumes
no interleaved random operations from other callers.
Prompt-mode generation streams text as tokens arrive. `inference.Model.Stream`
accepts a text callback; `lm.Stream` exposes token callbacks. They run on the
calling goroutine outside the worker batch. Callback errors stop generation and
return partial output; incomplete UTF-8 suffixes are withheld until complete
or flushed at the end. Do not invoke streaming inside a worker callback or Batch
if its consumer needs MLX work on another goroutine.
Only the standard Qwen2 architecture with SiLU, full attention, unscaled RoPE,
tied embeddings, zero dropout, and bf16 safetensors weights is supported.
Weights can be a single file or an indexed set of shards; see the
[checkpoint loader](checkpoint/README.md) for validation and storage limits.
Configuration and tensor mismatches return errors. The tokenizer is pure Go
and supports Qwen2's byte-level BPE, NFC normalization, and added tokens.

Fine-tune q/v LoRA adapters in all 24 layers, then generate with them:

```sh
go run -tags mlx ./cmd/finetune-qwen
go run -tags mlx ./cmd/generate \
  -adapters checkpoints/qwen-lora.safetensors \
  -prompt "Please state amber's assigned code."
```

The included synthetic dataset teaches two code assignments using eight
training prompts and two different validation prompts. It is an end-to-end
engineering demonstration, not a general capability benchmark. The command
requires validation loss to improve and checks that reloading adapters
reproduces it. It saves only float32 LoRA factors and metadata; bf16 base
weights remain frozen. The default rank is 4, alpha is 4, batch size is 2,
learning rate is 0.001, global gradient norm limit is 1, and training lasts
30 AdamW steps.

For your own data, use JSONL records in separate training and validation files:

```json
{"prompt":"What is the code for amber?","completion":"The code for amber is 7."}
```

```sh
go run -tags mlx ./cmd/finetune-qwen -train train.jsonl -valid valid.jsonl \
  -steps 100 -rank 8 -batch-size 2 -learning-rate 0.0001 \
  -max-length 128 -out checkpoints/custom-lora.safetensors
```

Prompt tokens are masked out of the loss. Gradients are accumulated across
variable-length examples and averaged by the number of completion tokens.
Global L2 gradient clipping is applied after averaging, before AdamW; use
`-max-grad-norm 0` to disable clipping. Nonfinite gradient norms are rejected
before updating parameters or optimizer state. The library exposes this as
`mlx.ClipGradNorm`; `qwen2.TrainOptions.MaxGradNorm` defaults to zero (disabled).
Examples are processed individually, so padding is unnecessary. Overlong
examples are rejected rather than truncated. Keep validation examples separate
from training. Memory use grows with sequence length and vocabulary logits;
this implementation has no activation checkpointing or quantization.

Adapters are bound to the exact base checkpoint SHA256. Single-file bundles
retain their existing hash; sharded bundles hash the tensor routing and shard
contents. Repacking or renaming shards changes this identity. Loading adapters
against another checkpoint or incompatible configuration fails. Each training call
starts fresh AdamW moments unless `ResumeFrom` is supplied; adapter-only files
are not full optimizer checkpoints.
Models, caches, adapters and optimizers must not be mutated or closed while in
use. A cache belongs to one generation, and must be discarded after an error.

See [resumable training and the extraction benchmark](docs/training.md) and
[reference verification](docs/verification.md) for the detailed workflows.

## Test

Default stub build:

```sh
go test ./...
```

Native compile and validation tests:

```sh
go test -tags mlx ./...
```

Runtime MLX tests, from a normal Terminal session with Metal access:

```sh
CGO_ENABLED=1 go test -tags "mlx mlxruntime" ./...
```

The same commands are available through `make`:

```sh
make test
make test-native
make test-runtime
make vet
make vet-native
make test-race
make test-race-native
make smoke
make linear
make mlp
make train-linear
make autograd-linear
make finetune-mlp
```

## API Covered

- Constructors: `NewFloat32`, `NewFloat64`, `NewInt32`, `NewInt64`, `Arange`,
  `ArangeDType`, `Zeros`, `Ones`, `Full`, `ZerosLike`, `OnesLike`, scalar
  constructors
- Introspection: `Shape`, `Size`, `DType`, shared close state across copied
  `Array` values
- Data copies: `Float32Data`, `Float64Data`, `Int32Data`, `Int64Data`,
  `UInt8Data`, `UInt32Data`, `UInt64Data`, `BoolData`
- Elementwise ops: `Add`, `Subtract`, `Multiply`, `Divide`, `Maximum`,
  `Minimum`, `Power`, `Clip`, `Abs`, `Exp`, `Log`, `Negative`, `Square`,
  `Sqrt`, `Sigmoid`, `Tanh`, `Sin`, `Cos`, `ReLU`, `StopGradient`
- Matrix/reduction ops: `Matmul`, `Sum`, `SumAxis`, `SumAxes`, `Mean`,
  `MeanAxis`, `MeanAxes`, `LogSumExp`, `LogSumExpAxis`, `LogSumExpAxes`,
  `AddMM`, `CumsumAxis`
- Device/stream control: `SetDefaultGPU`, `SetDefaultCPU`, `SetDefaultDevice`,
  `Batch`, `Synchronize`
- Shape/type ops: `Reshape`, `Transpose`, `TransposeAxes`, `BroadcastTo`,
  `ExpandDims`, `ExpandDimsAxes`, `Squeeze`, `SqueezeAxis`, `SqueezeAxes`,
  `Flatten`, `AsType`, `Contiguous`, `Slice`, `SliceUpdate`
- Model ops: `Softmax`, `SoftmaxAxis`, `SoftmaxAxes`, `Argmax`, `ArgmaxAxis`,
  `Argmin`, `ArgminAxis`, `Equal`, `Greater`, `GreaterEqual`, `Less`,
  `LessEqual`, `Where`, `Take`, `TakeAxis`, `TakeAlongAxis`, `Gather`,
  `GatherSlices`, `Concatenate`, `ConcatenateAxis`, `Stack`, `StackAxis`
- IO: `Load`, `Save`, `LoadSafetensors`, `SaveSafetensors`
- Random: `RandomSeed`, `RandomKey`, `RandomNormal`, `RandomUniform`,
  `RandomRandint`, `RandomBernoulli`, `RandomCategorical`
- Transforms: `Closure`, `NewClosure`, `Compile`, `ValueAndGrad`, `NewValueAndGrad`,
  `Eval`, `AsyncEval`
- NN/loss helpers: `Linear`, `LinearNoBias`, `SiLU`, `RMSNorm`, `RoPE`,
  `ScaledDotProductAttention`, `MSELoss`, `LogSoftmaxAxis`,
  `SoftmaxCrossEntropyAxis`, `CrossEntropyAxis`
- Optimizers/utilities: `SGD`, `SGDWithLearningRate`, `NewAdamW`, `CloseArrays`
- Weight-only quantized matmul: `MXFP8Matmul`, `MXFP4Matmul` (packed UInt32
  weights and UInt8 E8M0 scales; floating inputs). See the experimental
  [DeepSeek adapters and validation](deepseek/README.md).

## Memory Profiling

`mlx.LiveArrays()` counts owned Go array handles that have not been closed;
copies sharing one handle count once, borrowed handles do not count, and the
stub build returns zero. Compare before/after counts when testing cleanup.
The snapshot is process-wide, not a native-buffer or byte count; lazy graphs
can retain buffers after their Go handles close. No finalizers are installed.

`mlx.GetMemoryUsage()` reports process-wide MLX allocator active, cached and
peak bytes, not Go heap size or process RSS. Evaluate pending work and call
`mlx.Synchronize()` before a measurement window to drain previously submitted
work on the selected device's default stream. Synchronization does not evaluate
unsubmitted lazy graphs or wait for unrelated streams, and other callers can
still affect these process-wide counters. `mlx.ResetPeakMemory()` resets the global peak statistic without
freeing allocations; use it only during coordinated profiling, not within
ordinary inference calls. The [packed attention report](deepseek/FP8_ATTENTION_VALIDATION.md)
shows whole-attention timing and allocator measurements on real weight samples.
The [combined quantization report](deepseek/COMBINED_QUANTIZATION_VALIDATION.md)
and [FP4 expert report](deepseek/FP4_EXPERT_VALIDATION.md) cover the newer paths.
The [sparse expert report](deepseek/SPARSE_EXPERT_VALIDATION.md) measures optional
token-to-expert dispatch and documents prepared packed checkpoint loading.

## Development Notes

- MLX computation is lazy. Call `Eval` or a data-copy method such as
  `Float32Data` before reading results.
- File loading uses a CPU stream because MLX's load primitive has no GPU
  implementation. Loaded arrays can feed GPU operations without changing the
  selected device.
- The wrapper defaults to GPU index 0. Call `SetDefaultCPU` when you want CPU
  execution, or `SetDefaultDevice` to choose CPU/GPU index 0 explicitly. This
  does not bypass MLX's Metal initialization requirement in sandboxed processes
  that cannot enumerate a Metal device.
- Native MLX calls run on a dedicated OS thread. This keeps MLX stream affinity
  stable across lazy graph construction and evaluation, so callers can use
  ordinary Go goroutines; the native calls themselves are serialized.
- Use `Batch` to amortize dispatcher overhead across a sequence of MLX calls,
  such as a full training step.
- Data-copy methods call `Contiguous` internally before touching MLX's raw data
  pointers, so transposed and broadcasted views copy back correctly. Numeric
  readers bulk-copy into owned Go slices; bool readers convert each element.
- The native build installs an MLX error handler during package initialization.
  MLX operation failures should return Go errors with MLX's diagnostic text
  instead of aborting the process.
- Close arrays explicitly with `defer arr.Close()` when you allocate them.
  Copies share close state, so closing one copy prevents later use through other
  copies.
- Helper functions close their own intermediate arrays, but they do not close
  inputs or returned arrays. Call `CloseArrays` for returned parameter, value, or
  gradient slices.
- Keep Go slices alive until after cgo calls return. The current constructors use
  `mlx_array_new_data`, which copies the input buffer.
- `NewValueAndGrad` wraps MLX's closure-based autograd. Callback inputs are
  temporary handles managed by the wrapper. Callback outputs are transferred to
  MLX, so return freshly created arrays rather than arrays you intend to keep
  using after the callback.
- Closure and value-and-gradient callbacks run on the MLX worker thread. Do not
  delegate MLX work from a callback to another goroutine, and do not block a
  callback on anything that needs to call MLX.
- Expand the wrapper a few operations at a time. MLX C's API is broad, and a
  typed Go surface is easier to maintain than a generated one-to-one binding at
  the start.
