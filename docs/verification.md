# Verification

Run commands from the repository root.

### Reference Verification

CI runs 119 Hugging Face tokenizer cases, small-transformer cache comparisons,
LoRA finite-difference gradients, learning/frozen-weight/checkpoint tests, and
AdamW checks against a scalar reference. The runtime suite also compares a tiny
LoRA training step against saved LitGPT/PyTorch results on CPU and GPU. It never
downloads model weights or runs Python.

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

### LitGPT Training Reference

```sh
CGO_ENABLED=1 go test -tags "mlx mlxruntime" ./qwen2 -run '^TestLitGPTTrainingReference$' -v
```

`qwen2/testdata/litgpt_training.json` contains synthetic float32 weights for a
two-layer Qwen2 model with grouped-query attention, tied embeddings, and rank-2
query/value LoRA adapters. Both adapter factors start nonzero and alpha/r is 2.
Two examples have different lengths, masked prompts, and different completion
token counts. The test checks every logit and adapter gradient, per-example and
token-weighted losses, every adapter value after one real `Train`/AdamW update,
post-step loss, and unchanged base weights.

The generator executes upstream `litgpt.lora.GPT`, PyTorch autograd and AdamW on
CPU with float32 math attention. It exports tensors in Go's layout: projections
are transposed, and LitGPT's packed QKV and query/value adapter tensors are split.
It explicitly ties the output head and weights the loss by completion tokens.
This is a numerical reference for our training objective, rather than a run of
LitGPT's training CLI with its default batching and learning-rate schedule.

The fixture records the LitGPT commit, installed dependency versions, Python
version, and hashes of the upstream model/config/LoRA files. The generator
verifies the installed versions and commit against
`qwen2/requirements-training-reference.txt`. To regenerate with Python 3.12:

```sh
training_reference_env="$(mktemp -d)"
python3.12 -m venv "$training_reference_env"
"$training_reference_env/bin/python" -m pip install -r qwen2/requirements-training-reference.txt
"$training_reference_env/bin/python" qwen2/make_training_fixture.py
```

Generation needs no pretrained checkpoint or Metal device. Use `--output PATH`
to compare regenerated data without replacing the checked-in fixture. Acceptance
limits stay in the Go test: absolute error 2e-6 plus relative error 2e-5 for
logits/losses/updates, or 2e-4 for gradients. This tiny FP32 check complements the
real-checkpoint bf16 inference fixture; it does not cover checkpoint conversion,
quantization, or longer training trajectories.

### LitGPT Training Runtime

The runtime benchmarks use the same fixture, two examples per optimizer step,
and completion-token weighting. They include forward/backward computation,
gradient accumulation and finite-gradient checks, AdamW, input array creation,
and scalar loss reads. Model construction is excluded. Each round resets the
model, runs 20 untimed warmup steps, then times 100 steps with the same optimizer;
device synchronization brackets the timed interval.

Run these commands sequentially on a Metal-capable Mac, using the isolated
Python environment created above:

```sh
VECLIB_MAXIMUM_THREADS=1 CGO_ENABLED=1 \
  go test -tags "mlx mlxruntime" ./qwen2 -run '^$' \
  -bench '^BenchmarkLitGPTTrainingRuntime$' -benchtime=100x -count=5
PYTORCH_ENABLE_MPS_FALLBACK=0 VECLIB_MAXIMUM_THREADS=1 \
  "$training_reference_env/bin/python" qwen2/benchmark_training_reference.py \
  --device cpu --output /tmp/litgpt-cpu.json
PYTORCH_ENABLE_MPS_FALLBACK=0 VECLIB_MAXIMUM_THREADS=1 \
  "$training_reference_env/bin/python" qwen2/benchmark_training_reference.py \
  --device mps --output /tmp/litgpt-mps.json
```

Keep `-benchtime=100x`: an adaptive Go benchmark changes the number of training
steps and the resulting model trajectory. Python defaults to the same five
rounds, 20 warmup steps, and 100 measured steps. It checks the initial forward
pass, gradients, and update against the fixture before timing either device.
MPS CPU fallback is disabled; PyTorch uses one CPU thread, float32 math attention,
and unfused AdamW. `VECLIB_MAXIMUM_THREADS=1` is set for both implementations;
this does not assert that every MLX operation uses one thread. Go shuffles the
two examples; Python visits them in fixture order. Both include both examples
in every update. LitGPT keeps packed QKV/LoRA tensors while Go uses separate
projections and adapter factors.

The [M3 Pro results](../benchmarks/litgpt/runtime-m3-pro.json) record all five
measurements, versions, fixture/source hashes, and hardware. This two-layer
fixture has sequence lengths 5 and 3 and hidden width 8. Treat it as a comparison
of tiny training steps, not evidence of full-size model throughput or inference
speed. A representative performance comparison needs realistic model sizes,
sequence lengths, and batch sizes measured separately.

### Qwen2.5 Checkpoint Training Runtime

The checkpoint benchmarks run the pretrained Qwen2.5-0.5B-Instruct model with
512- and 1,024-token sequences, batch size 1, and rank-8 query/value LoRA in every
layer (alpha 16). Both implementations cast the same BF16 checkpoint to FP32
and load identical initial adapters. FP32 aligns precision because Go's mixed
BF16/FP32 LoRA casting differs from upstream LitGPT's implementation.
AdamW uses learning rate 1e-4, weight decay
0.01, default betas/epsilon, and a finite-gradient check without clipping.

The input is a repeated synthetic token stream, with next-token targets and
the first half of target positions masked. This measures training computation,
not model quality on a task. Timing includes input array creation, the full
forward/backward pass, a scalar loss read, the gradient check, and the optimizer
update. Loading, conversion, tokenization, dataset I/O, verification probes, and
checkpoint writing are excluded. Three rounds each start from the shared initial
adapters and a fresh optimizer; three warmup updates precede five timed updates
using that same optimizer. Synchronization brackets the measured interval.

The model checkpoint must already exist locally. Using the isolated Python
environment above, build the Go measurement tool and prepare the shared adapters:

```sh
training_benchmark_dir="$(mktemp -d)"
training_benchmark_model="$PWD/models/Qwen2.5-0.5B-Instruct"
CGO_ENABLED=1 go test -tags "mlx mlxruntime" -c \
  -o "$training_benchmark_dir/qwen-runtime.test" ./qwen2
VECLIB_MAXIMUM_THREADS=5 \
  MLXGO_TRAIN_BENCH_MODEL="$training_benchmark_model" \
  MLXGO_TRAIN_BENCH_DEVICE=gpu MLXGO_TRAIN_BENCH_LENGTH=16 \
  MLXGO_TRAIN_BENCH_ROUNDS=1 MLXGO_TRAIN_BENCH_WARMUP=1 MLXGO_TRAIN_BENCH_STEPS=1 \
  MLXGO_TRAIN_BENCH_SAVE_ADAPTERS="$training_benchmark_dir/initial.safetensors" \
  MLXGO_TRAIN_BENCH_OUTPUT="$training_benchmark_dir/preparation.json" \
  "$training_benchmark_dir/qwen-runtime.test" \
  -test.run '^TestQwenTrainingRuntime$' -test.v -test.timeout 30m
```

Run all measurements sequentially to avoid competing for the same processor and
memory. This loop also exports each implementation's adapters after one untimed
update for numerical comparison:

```sh
for training_length in 512 1024; do
  for training_device in gpu cpu; do
    VECLIB_MAXIMUM_THREADS=5 \
      MLXGO_TRAIN_BENCH_MODEL="$training_benchmark_model" \
      MLXGO_TRAIN_BENCH_DEVICE="$training_device" MLXGO_TRAIN_BENCH_LENGTH="$training_length" \
      MLXGO_TRAIN_BENCH_ROUNDS=3 MLXGO_TRAIN_BENCH_WARMUP=3 MLXGO_TRAIN_BENCH_STEPS=5 \
      MLXGO_TRAIN_BENCH_ADAPTERS="$training_benchmark_dir/initial.safetensors" \
      MLXGO_TRAIN_BENCH_VERIFY_OUT="$training_benchmark_dir/go-$training_device-$training_length-step1.safetensors" \
      MLXGO_TRAIN_BENCH_OUTPUT="$training_benchmark_dir/go-$training_device-$training_length.json" \
      "$training_benchmark_dir/qwen-runtime.test" \
      -test.run '^TestQwenTrainingRuntime$' -test.v -test.timeout 30m
  done
  for training_device in mps cpu; do
    training_reference_device="$training_device"
    if [ "$training_device" = mps ]; then training_reference_device=gpu; fi
    PYTORCH_ENABLE_MPS_FALLBACK=0 VECLIB_MAXIMUM_THREADS=5 \
      "$training_reference_env/bin/python" qwen2/benchmark_training_checkpoint.py \
      --model "$training_benchmark_model" --adapters "$training_benchmark_dir/initial.safetensors" \
      --device "$training_device" --length "$training_length" --threads 5 \
      --rounds 3 --warmup 3 --steps 5 \
      --reference "$training_benchmark_dir/go-$training_reference_device-$training_length.json" \
      --reference-adapters "$training_benchmark_dir/go-$training_reference_device-$training_length-step1.safetensors" \
      --verify-output "$training_benchmark_dir/torch-$training_device-$training_length-step1.safetensors" \
      --output "$training_benchmark_dir/torch-$training_device-$training_length.json"
  done
done
```

PyTorch uses five CPU threads, native attention dispatch, and unfused AdamW,
without `torch.compile`, gradient checkpointing, or MPS CPU fallback. Both
implementations receive `VECLIB_MAXIMUM_THREADS=5`; this does not establish an
exact MLX thread count. The Go path uses its existing compiled SwiGLU and native
attention. These are measurements of the two current implementations, not a
comparison of identically fused operation graphs or the LitGPT training CLI.

The [checkpoint results](../benchmarks/litgpt/runtime-qwen2.5-0.5b-m3-pro.json)
record the checkpoint, shared adapters, settings, raw rounds, and initial/one-step
numerical diagnostics. Those diagnostics are separate from the strict tiny
fixture assertions: close losses alone do not establish that every adapter update
passes the fixture's tensor tolerances. The Python reference check stops on an
initial or post-update loss difference above 5e-4. It compares every adapter
element and records failure counts, maximum error, and the B-factor update's relative L2
error and cosine similarity; strict adapter mismatches are reported without
weakening the existing tiny fixture's tolerances. Because B starts at zero, A's
first update has only weight decay, so its agreement is not a gradient check.
Report throughput as input tokens per
second, with half as many supervised tokens per second for these even lengths.
The short runs on a shared workstation do not establish sustained training
throughput, BF16 or quantized performance, larger-batch performance, or inference
speed. The earlier tiny benchmark used forced PyTorch math attention and one CPU
thread; changes from those results cannot be attributed solely to model size.

