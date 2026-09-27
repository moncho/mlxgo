# Training

Run commands from the repository root.

### Resumable Training

Save a full checkpoint at completed optimizer steps, then resume in a new process:

```sh
go run -tags mlx ./cmd/finetune-qwen -steps 20 \
  -checkpoint checkpoints/training.safetensors -checkpoint-every 5
go run -tags mlx ./cmd/finetune-qwen -steps 10 \
  -resume checkpoints/training.safetensors \
  -checkpoint checkpoints/training.safetensors -checkpoint-every 5
```

`-steps` always means **additional** steps; the second command ends at step 30.
Reported step numbers are absolute. Keep rank, learning rate, gradient limit,
batch size, and training data the same when resuming. Checkpoints also bind
weight decay, seed, model configuration, base-weight SHA256, and the ordered
tokenized inputs/targets/loss masks. A changed dataset or training setting fails
before replacing adapter parameters. Validation data is not part of training
state and may differ. CLI validation loss is compared to the fresh base model,
including on resumed runs, and adapter export/reload is still verified.

The full file stores adapters, float32 AdamW first/second moments, completed
step count, current permutation/cursor/epoch, and seed. `-out` remains a separate
adapter-only inference export; it cannot share a path with the training file.
The same `-resume` and `-checkpoint` path is allowed. `-checkpoint-every 0`
(default) saves only after the final step; periodic saves also always save the
final step. Set `-checkpoint` explicitly to enable saving. An interruption loses
only work since the last completed save; resume from that file after an error.

Writes use a temporary file in the destination directory, flush it, then rename
over the destination. A failed pre-rename write leaves the previous checkpoint
intact. This is atomic replacement, not a backup or a guarantee against power
loss. Save errors are reported after the completed step; in-memory parameters
have already advanced. Abrupt process termination may leave a temporary file.

The library uses `TrainOptions.ResumeFrom`, `CheckpointPath`, and
`CheckpointEvery`. `mlx.AdamW.State` and `mlx.NewAdamWFromState` expose owned
optimizer snapshots for other models. Do not mutate training data, adapters,
or optimizer state while training or saving. Checkpoint files should be trusted.

To preserve the existing sampling sequence, resume replays past Go `math/rand`
permutations from the saved seed, checks the saved permutation, then restores
the cursor. No gradient steps are replayed; shuffle replay costs O(previous
epochs * dataset size). The format currently requires the same Go version.
Bitwise continuation is tested separately on CPU and GPU on the same software
and device; numerical identity across MLX versions or devices is not promised.

Real-checkpoint verification on Qwen2.5-0.5B compared six uninterrupted steps
with three steps plus a fresh-process three-step resume: all 288 adapter and
moment tensors were bit-identical, metadata matched, and both validation losses
were 0.729367. To compare your own two full training artifacts:

```sh
MLXGO_CHECKPOINT_FULL=checkpoints/full.safetensors \
MLXGO_CHECKPOINT_RESUMED=checkpoints/resumed.safetensors \
  go test -tags "mlx mlxruntime" ./qwen2 -run TestTrainingCheckpointFilesEqual -v
```

### Extraction Benchmark

The [ticket extraction benchmark](../benchmarks/tickets/README.md) compares the
base model with trained adapters on held-out support-ticket wording. It includes
192 training, 48 validation, and 96 test examples, plus eight unrelated retention
prompts. Reports contain every prediction, strict JSON/schema and field scores,
timings, process peak RSS, and checkpoint/dataset hashes. The data is synthetic;
this is a reproducible application-shaped experiment, not a real-world accuracy
claim. Run `go run ./cmd/bench-qwen -mode prepare` without MLX to recreate the data.

