# MLX From Go

This workspace is a small Go starter for using Apple's MLX through the official
MLX C bridge. It includes a low-level array wrapper plus a small helper layer
for common model code such as linear layers, losses, and SGD updates.

MLX itself does not expose an official Go API. The supported native path is:

```text
Go -> cgo -> mlx-c -> MLX
```

## Requirements

- Apple Silicon Mac
- macOS 14 or newer
- Go with cgo enabled
- Homebrew `mlx-c`

Install the native dependencies:

```sh
brew install mlx-c
```

Install the Go module:

```sh
go get github.com/moncho/mlxgo
```

The cgo binding in this starter uses Homebrew's Apple Silicon paths:

- headers: `/opt/homebrew/include`
- library: `/opt/homebrew/lib/libmlxc.dylib`

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
  `UInt32Data`, `UInt64Data`, `BoolData`
- Elementwise ops: `Add`, `Subtract`, `Multiply`, `Divide`, `Maximum`,
  `Minimum`, `Power`, `Clip`, `Abs`, `Exp`, `Log`, `Negative`, `Square`,
  `Sqrt`, `Sigmoid`, `Tanh`, `Sin`, `Cos`, `ReLU`, `StopGradient`
- Matrix/reduction ops: `Matmul`, `Sum`, `SumAxis`, `SumAxes`, `Mean`,
  `MeanAxis`, `MeanAxes`, `LogSumExp`, `LogSumExpAxis`, `LogSumExpAxes`,
  `AddMM`
- Device/stream control: `SetDefaultGPU`, `SetDefaultCPU`, `SetDefaultDevice`,
  `Batch`
- Shape/type ops: `Reshape`, `Transpose`, `TransposeAxes`, `BroadcastTo`,
  `ExpandDims`, `ExpandDimsAxes`, `Squeeze`, `SqueezeAxis`, `SqueezeAxes`,
  `Flatten`, `AsType`, `Contiguous`
- Model ops: `Softmax`, `SoftmaxAxis`, `SoftmaxAxes`, `Argmax`, `ArgmaxAxis`,
  `Argmin`, `ArgminAxis`, `Equal`, `Greater`, `GreaterEqual`, `Less`,
  `LessEqual`, `Where`, `Take`, `TakeAxis`, `TakeAlongAxis`, `Gather`,
  `GatherSlices`, `Concatenate`, `ConcatenateAxis`, `Stack`, `StackAxis`
- IO: `Load`, `Save`, `LoadSafetensors`, `SaveSafetensors`
- Random: `RandomSeed`, `RandomKey`, `RandomNormal`, `RandomUniform`,
  `RandomRandint`, `RandomBernoulli`, `RandomCategorical`
- Transforms: `Closure`, `NewClosure`, `ValueAndGrad`, `NewValueAndGrad`,
  `Eval`, `AsyncEval`
- NN/loss helpers: `Linear`, `LinearNoBias`, `MSELoss`, `LogSoftmaxAxis`,
  `SoftmaxCrossEntropyAxis`, `CrossEntropyAxis`
- Optimizers/utilities: `SGD`, `SGDWithLearningRate`, `CloseArrays`

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
  pointers, so transposed and broadcasted views copy back correctly.
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
