# Initial Fixed-Recipe Run

One run on Apple M3 Pro, macOS arm64, Go 1.27.0, using the protocol in
`../README.md`. No hyperparameters or test references were changed after seeing
the test results. Raw predictions, timings, and hashes are in `base.json` and
`tuned.json`. Adapters and base weights are not committed.

| Test metric | Base | LoRA |
|---|---:|---:|
| Exact extraction | 0/96 (0%) | 64/96 (66.7%) |
| Valid JSON | 3/96 | 96/96 |
| Strict schema | 0/96 | 96/96 |
| Correct ticket ID | 0/96 | 96/96 |
| Correct queue | 0/96 | 91/96 |
| Correct priority | 0/96 | 68/96 |
| Token cap reached | 0/96 | 0/96 |
| Mean generation latency | 0.406 s | 0.250 s |
| Output tokens / total generation second | 93.3 | 80.1 |
| Peak process RSS | 641.4 MiB | 1055.5 MiB |
| Unrelated-prompt exact match | 5/8 | 6/8 |

Validation loss: **0.812929 -> 0.025405** after 100 training steps. Reloaded
adapters reproduced the final validation loss exactly. Training settings and
step losses are preserved in `training.txt`.

## Interpretation

Fine-tuning improved adherence to the application's output contract. The base
model often emitted markdown fences and expanded label descriptions instead of
the required enum values. The 0/96 strict score is **not** evidence that it
understood none of the tickets; invalid schemas score zero on every field.

Semantic generalization is incomplete: 28 priority errors and 5 queue errors
remain, with one overlapping error, for 32 failed records. For example, some
outputs label "This does not block anyone on the team" as normal rather than
low priority. These are real failures, not corrected or excluded from scoring.

Lower latency does not establish a faster model. The adapter model emitted
shorter answers (20 tokens each); its output-token rate was lower. RSS increased
by about 414 MiB. These are single-run process measurements, not a dedicated
kernel-throughput or total-GPU-memory benchmark.

The tiny retention set showed no aggregate regression, but eight prompts cannot
establish that general capabilities were preserved. Synthetic template-derived
records are correlated and cannot establish accuracy on real tickets. This
test split has now been examined; use a new independent test set if subsequent
training changes are motivated by these failure examples.

## Inspect

From the repository root, without native MLX:

```sh
go run ./cmd/bench-qwen -mode compare \
  -base benchmarks/tickets/results/base.json \
  -tuned benchmarks/tickets/results/tuned.json
```
