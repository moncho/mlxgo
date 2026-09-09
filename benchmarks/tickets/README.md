# Ticket Extraction Benchmark

This is a reproducible **synthetic** support-ticket extraction experiment, not
evidence of performance on real customer tickets. The corpus is generated from
handwritten templates in `internal/ticketbench/dataset.go`. It contains no real
customer data and requires no external dataset download.

The [initial fixed-recipe results](results/README.md) include raw predictions:
exact extraction improved from 0/96 to 64/96, but held-out priority errors remain.

## Protocol

- Extract `ticket_id`, `queue`, and `priority` using the policy in every prompt.
- Train: 192 records from four rendering families. Validation: 48 records from
  one separate family. Test: 96 records from two other families.
- IDs, full prompts, rendering families, and issue/impact wording are disjoint
  across splits. Label policy and schema are shared. Four metadata variants
  repeat each case: records are correlated, not independent real-world samples.
- Eight unrelated prompts are a small retention smoke test, not a general
  reasoning benchmark. They are never used for training or model selection.
- Fixed training recipe: rank/alpha 8, 100 steps, two accumulated examples per
  step, learning rate 0.0003, weight decay 0, global gradient norm limit 1,
  seed 42, maximum input length 256. This is a single predeclared run; no test-set
  hyperparameter search or checkpoint selection.
- Evaluate base and saved/reloaded adapters in separate processes on the GPU
  with greedy generation, a 96-token cap, and an identical unscored warm-up.
- Compare semantic exact match (all three fields), JSON validity, strict schema
  validity, and individual fields. Field scores are conservative: invalid schema
  scores zero on all fields. JSON key order/whitespace do not matter. Fences,
  extra text/fields, duplicate keys, nulls, and incorrect types fail schema checks.
- Report raw counts and every prediction, token-cap hits, generation time,
  prefill/decode time, and process-lifetime peak RSS. RSS includes model loading
  and warm-up and is **not** total GPU memory. Timing is a single local run, not
  a controlled performance benchmark; longer answers increase total latency.
- Dataset, base checkpoint, tokenizer, and adapter SHA256 values are recorded.
  Evaluation checks dataset hashes and split separation before loading MLX.

## Reproduce

Use the pinned Qwen2.5-0.5B-Instruct checkpoint documented in the root README.
Prepare uses only Go and works without native MLX. It rewrites the generated
dataset deterministically; do not use it on a directory containing custom data.

```sh
go run ./cmd/bench-qwen -mode prepare
go run -tags mlx ./cmd/bench-qwen -out checkpoints/tickets-base.json
go run -tags mlx ./cmd/finetune-qwen \
  -train benchmarks/tickets/train.jsonl -valid benchmarks/tickets/valid.jsonl \
  -steps 100 -batch-size 2 -rank 8 -learning-rate 0.0003 \
  -max-grad-norm 1 -max-length 256 -out checkpoints/tickets-lora.safetensors
go run -tags mlx ./cmd/bench-qwen \
  -adapters checkpoints/tickets-lora.safetensors -out checkpoints/tickets-tuned.json
go run ./cmd/bench-qwen -mode compare
```

Evaluation refuses to overwrite an existing report and removes incomplete
reports on ordinary errors. A worse score is a valid experimental result, not
a command failure. The training command separately checks validation-loss
improvement and exact loss reproduction after adapter reload.

The adapter is an inference artifact, **not a resumable training checkpoint**.
Each training invocation starts fresh optimizer moments and data ordering.
Full training-state checkpoints remain necessary before moving to long runs.
