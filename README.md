# MLX From Go

Go bindings for Apple's MLX through the official MLX C bridge, with array,
autograd and optimizer APIs, Qwen2.5-0.5B inference, and LoRA fine-tuning.
The higher-level model packages are experimental and deliberately narrow.

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

## Qwen Inference And LoRA

Download the public bf16 checkpoint once (about 1 GB) with the Hugging Face CLI:

```sh
hf download Qwen/Qwen2.5-0.5B-Instruct config.json model.safetensors tokenizer.json tokenizer_config.json \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --local-dir models/Qwen2.5-0.5B-Instruct
go run -tags mlx ./cmd/generate -prompt "Explain why the sky is blue in one sentence."
```

Generation runs entirely in Go through MLX on the GPU. It uses the Qwen chat
template, greedy decoding, tied embeddings, and a concatenating KV cache.
Only the standard Qwen2 architecture with SiLU, full attention, unscaled RoPE,
tied embeddings, zero dropout, and a single bf16 safetensors file is supported.
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

Adapters are bound to the exact base checkpoint SHA256. Loading them against
another checkpoint or incompatible configuration fails. Each training call
starts fresh AdamW moments; adapter files are not full optimizer checkpoints.
Models, caches, adapters and optimizers must not be mutated or closed while in
use. A cache belongs to one generation, and must be discarded after an error.

### Extraction Benchmark

The [ticket extraction benchmark](benchmarks/tickets/README.md) compares the
base model with trained adapters on held-out support-ticket wording. It includes
192 training, 48 validation, and 96 test examples, plus eight unrelated retention
prompts. Reports contain every prediction, strict JSON/schema and field scores,
timings, process peak RSS, and checkpoint/dataset hashes. The data is synthetic;
this is a reproducible application-shaped experiment, not a real-world accuracy
claim. Run `go run ./cmd/bench-qwen -mode prepare` without MLX to recreate the data.

### Reference Verification

CI runs 119 Hugging Face tokenizer cases, small-transformer cache comparisons,
LoRA finite-difference gradients, learning/frozen-weight/checkpoint tests, and
AdamW checks against a scalar reference. It never downloads model weights.

The real-checkpoint tests are opt-in:

```sh
MLXGO_QWEN2_DIR="$PWD/models/Qwen2.5-0.5B-Instruct" \
  go test -tags "mlx mlxruntime" ./qwen2 -run TestQwenGolden -v
MLXGO_QWEN2_DIR="$PWD/models/Qwen2.5-0.5B-Instruct" MLXGO_QWEN2_MEMORY=1 \
  go test -tags "mlx mlxruntime" ./qwen2 -run TestQwenMemory -v
```

Golden verification requires all 32 reference tokens to match and maximum
absolute prefill-logit error of at most 0.25. These limits are fixed in the
test, not chosen from the Go output. The memory test warms up for 200 tokens,
then checks retained RSS after two more 200-token runs against a 64 MiB growth
limit. This detects regression in retained memory, not all possible leaks.

Fixtures were generated with the pinned versions in
`qwen2/requirements-reference.txt`, using the upstream
[mlx-lm Qwen2 model](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen2.py)
and its built-in `ConcatenateKVCache`. The model uses fused bias projections
and compiled SwiGLU to match reference bf16 rounding. Different MLX versions,
kernel choices, or cache layouts can change greedy decisions near tied logits.
The C++ attention shim supports both mlx-c 0.6.0 and the newer `force_fused`
signature without a version-dependent function-pointer cast.

To deliberately regenerate fixtures on a Metal-capable Mac:

```sh
python3.12 -m venv .venv-qwen
.venv-qwen/bin/pip install -r qwen2/requirements-reference.txt
.venv-qwen/bin/python qwen2/make_fixture.py
```

The checked-in tokenizer vocabulary is a reduced fixture for the reference
corpus; applications must load the full tokenizer from the model directory.

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
- Transforms: `Closure`, `NewClosure`, `Compile`, `ValueAndGrad`, `NewValueAndGrad`,
  `Eval`, `AsyncEval`
- NN/loss helpers: `Linear`, `LinearNoBias`, `SiLU`, `RMSNorm`, `RoPE`,
  `ScaledDotProductAttention`, `MSELoss`, `LogSoftmaxAxis`,
  `SoftmaxCrossEntropyAxis`, `CrossEntropyAxis`
- Optimizers/utilities: `SGD`, `SGDWithLearningRate`, `NewAdamW`, `CloseArrays`

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
