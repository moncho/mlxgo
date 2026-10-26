# BANKING77 Few-Shot Pilot

This experiment compares a word-unigram multinomial Naive Bayes classifier,
prompted Qwen2.5-0.5B-Instruct, and q/v LoRA fine-tuning of the same Qwen model.
It is a closed-set, English-language intent classifier, not financial advice.
It has no out-of-domain rejection or calibrated confidence. Do not use it to
authorize transactions or make financial decisions.

The [initial fixed-recipe results](results.md) reached 62.34% accuracy after
fine-tuning, versus 53.77% for Naive Bayes and 14.29% for the fixed prompted
base-model setup. Substantial errors remain; this is not deployment validation.

The [expanded-data follow-up](followup.md) keeps validation unchanged, adds
training examples, and compares ordinary and valid-label-constrained decoding
on 770 previously unused test messages. Its protocol separates validation-based
selection from final test reporting.

The [full-data experiment](full-data.md) expands to all eligible training rows,
adds TF-IDF logistic regression, and selects LoRA learning rate and duration
using validation before testing on another untouched subset.

## Data And Attribution

[BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data)
is provided by PolyAI under CC-BY-4.0. Cite Casanueva, Temcinas, Gerz, Henderson,
and Vulic (2020), [Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807).
The preparation command downloads CSVs, category names, and the upstream license
at the pinned revision; it records hashes and preserves the originals locally.
Downloaded data, training artifacts, and local reports are not committed.

The pilot uses all 77 intents with **8 training, 8 validation, and 5 test rows
per intent**: 616 / 616 / 385 records. Train and validation come only from the
official training set; test comes only from the official test set. Rows within
each class are sorted by SHA256 of `42/<source-split>-<five-digit-row-number>`.
The first 8 training rows form training, the next 8 form validation; the first
5 test rows form testing. Case/whitespace-normalized duplicates within source
training and overlaps with the entire source test set are removed from training
before selection. This detects exact normalized matches, not semantic duplicates.
No scores or generated predictions participate in selection.

This is a small, stratified subset, **not a full BANKING77 leaderboard result**.
Public benchmark material may have appeared in the base model's pretraining.
Results do not establish performance on real customers or unrelated domains.

## Fixed Recipe

Declared before test predictions: seed 42; 308 AdamW steps, batch size 4
(1232 examples, two passes through 616 training rows); q/v LoRA rank/alpha 8;
learning rate 0.0003; weight decay 0; global gradient norm limit 1; maximum
training length 1024. Base weights remain BF16 and frozen. No hyperparameter
search or checkpoint selection on test results. Export only the final adapters.
Training reports validation loss and verifies that reloaded adapters reproduce it.

Naive Bayes sees exactly the same 616 training messages, lowercased and split
into Unicode letter/number word tokens. It uses unigram counts, Laplace alpha=1,
empirical class priors, and ignores out-of-vocabulary words. It does not see the
prompt or category-name list as features. It is a simple baseline, not an optimized
classical classifier. No validation/test vocabulary or parameter fitting is used.

Qwen base and LoRA receive identical prompts listing every category, use greedy
generation on GPU, and have a 24-token output cap. Only surrounding whitespace
is ignored: explanations, code fences, or misspelled labels are invalid and
count as wrong. Metrics include accuracy, macro-F1, per-intent precision/recall/F1,
invalid labels, token-cap hits, raw predictions, latency, and process peak RSS.
RSS includes model loading and warm-up, not all Metal allocations. Timing is
one local run; the baseline executes on CPU and Qwen on GPU. Training cost is
separate from inference cost. There is no claim of general capability retention.

## Run

From the repository root, with the pinned Qwen checkpoint described in the root
README already installed:

```sh
go run ./cmd/bench-banking -mode prepare
go run ./cmd/bench-banking -mode check
go run -tags mlx ./cmd/finetune-qwen \
  -train benchmarks/banking77/data/train.jsonl \
  -valid benchmarks/banking77/data/valid.jsonl \
  -steps 308 -batch-size 4 -rank 8 -learning-rate 0.0003 \
  -max-grad-norm 1 -max-length 1024 \
  -checkpoint checkpoints/banking77/training.safetensors -checkpoint-every 77 \
  -out checkpoints/banking77/lora.safetensors
go run ./cmd/bench-banking -system nb -split test -out checkpoints/banking77/nb-test.json
go run -tags mlx ./cmd/bench-banking -system base -split test -out checkpoints/banking77/base-test.json
go run -tags mlx ./cmd/bench-banking -system lora \
  -adapters checkpoints/banking77/lora.safetensors \
  -split test -out checkpoints/banking77/lora-test.json
go run ./cmd/bench-banking -mode compare \
  checkpoints/banking77/nb-test.json checkpoints/banking77/base-test.json \
  checkpoints/banking77/lora-test.json
```

Preparation refuses an existing directory; evaluation refuses existing reports.
Use a fresh directory for another experiment. Validation classification uses
`-split valid`; do not tune on test results. Once inspected, these pilot test
examples must not be treated as an unseen test for changes motivated by their
errors. Full official test evaluation can be prepared separately with
`-test-per-class 40`, but overlaps with this pilot must be acknowledged.

## Classify A New Message

```sh
go run -tags mlx ./cmd/bench-banking -mode classify -system lora \
  -adapters checkpoints/banking77/lora.safetensors \
  -text "My card still has not arrived"
```

Use `-system nb` without `-tags mlx` for the baseline, or `-system base` for
the untuned model. Classification returns `{"intent":"..."}` and rejects
invalid generated labels. Valid labels can still be wrong, especially for
messages outside the 77 supported intents.

For the expanded-data adapters, use their matching dataset and optionally
constrain generation to the 77 allowed labels:

```sh
go run -tags mlx ./cmd/bench-banking -mode classify -system lora \
  -data benchmarks/banking77/data-v2 \
  -adapters checkpoints/banking77-v2/lora.safetensors -decoding labels \
  -text "My card still has not arrived"
```

Constrained decoding guarantees an allowed label on success, not the correct
intent. Omit `-decoding labels` for ordinary greedy generation. The reusable
`lm.NewChoices` API accepts tokenized choices; include EOS in each sequence.
