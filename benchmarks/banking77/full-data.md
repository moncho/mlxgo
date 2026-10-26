# Full-Data Experiment

This extends the [expanded-data experiment](followup.md), using all eligible
official training rows and a stronger classical baseline. Previous adapters,
reports, and datasets are preserved. Results remain subset results, not a full
BANKING77 leaderboard evaluation.

## Dataset

`data-v3` contains 9,376 training, 616 validation, and 770 test messages. Eleven
source-training duplicates/test overlaps are excluded; the original validation
file stays byte-identical. Training excludes the entire official test source.
The new test contains ten messages per intent, excluding IDs and normalized
texts from both inspected test sets (385 and 770 messages). Preparation copies
both predecessor manifests and verifies their selections and source hashes.
Normalization screens case/whitespace duplicates, not semantic duplicates or
possible inclusion in model pretraining.

## Fixed Search

The runner writes `protocol.json` before fitting anything. Three independent
LoRA runs start from the same base and seed, using learning rates 0.0001, 0.0003,
and 0.001. Each is evaluated after one and two passes: 2,344 and 4,688 updates,
batch size 4, q/v rank and alpha 8, seed 42, gradient clipping 1, no weight decay,
and maximum sequence length 1,024. The second pass resumes full optimizer state,
not just adapter weights. Each stage saves a resumable checkpoint every 586
updates and exports adapters; reloaded validation loss must match exactly.

Candidate classification uses the unchanged 77-label prompt and ordinary greedy
decoding with a 24-token cap. Select by validation macro-F1, then correct count,
then fewer passes, then lower learning rate. Validation token loss is diagnostic,
not the selection metric. This evaluates more data, more updates, and a small
search together; it does not isolate their individual causal contributions.

The classical baseline uses lowercased Unicode word unigrams and bigrams,
minimum document frequency 2, sublinear term frequency, smoothed IDF, and
L2-normalized rows. Vocabulary and IDF come only from training. Multinomial
logistic regression minimizes mean cross-entropy plus `||W||^2/(2*C*N)`;
intercepts are unregularized. Gonum v0.17.0 L-BFGS starts at zero, with maximum
300 iterations and gradient infinity-norm threshold 1e-6. Early solver exits
fail rather than silently becoming scores. Search C in {1, 4, 16}, using the
same validation metric, correct-count tie-break, then lower C. Reports record
feature count, solver status, iterations, objective, and gradient norm.

References: [TF-IDF definition](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)
and [Gonum optimization API](https://pkg.go.dev/gonum.org/v1/gonum@v0.17.0/optimize).
This is a Go implementation, not a claim of scikit-learn numerical parity.

After all validation runs, `selection.json` freezes the selected LoRA adapter
and TF-IDF regularization before any test inference. The final comparison uses
full-data Naive Bayes, selected TF-IDF, the previous expanded-data LoRA, selected
full-data LoRA, and that same selected LoRA with constrained decoding. Report
all five; do not change the selected model or decoder after reading test scores.

## Reproduce

From the repository root, using the pinned Qwen checkpoint and existing v2 data:

```sh
go run ./cmd/bench-banking -mode full \
  -parent benchmarks/banking77/data-v2 \
  -data benchmarks/banking77/data-v3 -test-per-class 10
go run ./cmd/bench-banking -mode check -data benchmarks/banking77/data-v3
go run -tags mlx ./cmd/bench-banking -mode search \
  -data benchmarks/banking77/data-v3 -out checkpoints/banking77-v3 \
  -adapters checkpoints/banking77-v2/lora.safetensors \
  -trained-on benchmarks/banking77/data-v2
```

Both preparation and search require new output directories. Search launches
training and evaluations sequentially; each has a separate log. Expect hours
of training, not seconds. On failure it stops, preserving logs and checkpoints;
it does not automatically resume an incomplete experiment or overwrite reports.
The ordinary `finetune-qwen -resume` workflow can recover saved training state.

To run the classical baseline independently without MLX:

```sh
go run ./cmd/bench-banking -system tfidf -logistic-c 16 \
  -data benchmarks/banking77/data-v3 -split valid \
  -out checkpoints/another-tfidf-valid.json
```

Inference timing excludes fitting, model loading, and one warm-up. Peak process
RSS includes fitting/loading and is not total Metal allocation. Single-run
timings are not isolated performance benchmarks. No out-of-domain rejection,
calibrated confidence, or general-capability retention is measured here.
