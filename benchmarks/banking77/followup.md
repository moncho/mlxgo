# Frozen-Validation Follow-Up

Protocol fixed before fresh test predictions. The initial 385 inspected test
messages are diagnostic only and will not be reused for the final comparison.

## Data

Expand the verified v1 source into `data-v2`, keeping all 616 validation records
byte-identical. Retain the original training examples and add deterministically
ordered examples up to 32 per intent, excluding validation and the entire source
test set. `contactless_not_working` has only 35 source-training examples, so it
contributes 27 after reserving validation; all other intents contribute 32.
There are 2459 unique training records. Never oversample to claim 32 unique rows.

Choose 10 previously unused official test records per intent (770 total) by the
same seeded hash ordering, excluding normalized text matches to all v1 test
records. Preserve the original source files and parent manifest, including
their hashes. Only case/whitespace duplicates are screened; this does not prove
semantic independence or absence from model pretraining. Remaining official
test examples stay unused.

## Training And Selection

One new LoRA training run from the original base weights, not continued from
the 8-shot adapters. Keep rank/alpha 8, seed 42, batch size 4, learning rate
0.0003, weight decay 0, gradient limit 1, and maximum length 1024. Run 1230
updates: two passes through 2459 records plus two examples from the next pass.
Save resumable state every 246 steps and export final adapters only.

This is a data-and-training-budget experiment: the larger corpus receives
roughly four times as many optimizer updates at the same number of passes.
It does not isolate data diversity from total compute. There is no learning-rate
search in this iteration. Compare classification accuracy and macro-F1 on the
fixed validation set; validation token loss is a secondary diagnostic.

## Decoding Ablation

Use the same unchanged 77-label prompt and 24-token cap. Compare the original
8-shot and new up-to-32-shot adapters with ordinary greedy generation and with
greedy decoding constrained to a trie of tokenized labels plus EOS. No fuzzy
matching, label substitutions, or correcting invalid strings after generation.
The trie permits EOS only after a complete allowed label and disallows all
other next tokens. It guarantees label membership on successful completion,
not semantic correctness. Greedy constrained decoding does not find the most
likely whole label across all possible token sequences.

Evaluate all four LoRA combinations on validation first. Freeze the comparison
before opening the fresh test set; report all four there, not only the winner.
Run an up-to-32-shot Naive Bayes baseline on the same validation/test splits.
Use validation macro-F1 for selection, then accuracy, then the cheaper variant
as tie-breakers. The public test is for reporting, not further tuning.

## Prepare And Train

```sh
go run ./cmd/bench-banking -mode expand \
  -data benchmarks/banking77/data-v2 -train-per-class 32 -test-per-class 10
go run ./cmd/bench-banking -mode check -data benchmarks/banking77/data-v2
go run -tags mlx ./cmd/finetune-qwen \
  -train benchmarks/banking77/data-v2/train.jsonl \
  -valid benchmarks/banking77/data-v2/valid.jsonl \
  -steps 1230 -batch-size 4 -rank 8 -learning-rate 0.0003 \
  -max-grad-norm 1 -max-length 1024 \
  -checkpoint checkpoints/banking77-v2/training.safetensors -checkpoint-every 246 \
  -out checkpoints/banking77-v2/lora.safetensors
```

The original artifacts under `checkpoints/banking77/` are never overwritten.
Downloaded data, checkpoints, and raw reports remain ignored locally.

## Evaluation

Use `-decoding greedy` or `-decoding labels`. The original adapter must explicitly
declare `-trained-on benchmarks/banking77/data` while evaluation still uses
`-data benchmarks/banking77/data-v2`. The new adapter defaults its declared
training source to the evaluation dataset. Reports record the training-file
hash and count; this declaration is checked against the provided dataset, not
cryptographically embedded in the adapter. The saved training artifacts and
logs provide the separate run provenance.

Example for the old adapter with constrained decoding on validation:

```sh
go run -tags mlx ./cmd/bench-banking -system lora \
  -data benchmarks/banking77/data-v2 -trained-on benchmarks/banking77/data \
  -adapters checkpoints/banking77/lora.safetensors -decoding labels \
  -split valid -out checkpoints/banking77-v2/old-labels-valid.json
```

Repeat for the other three variants using distinct report paths. Use
`-mode ablate -data benchmarks/banking77/data-v2 REPORT...` to recompute and
compare their summaries; it refuses mismatched evaluation data, references,
base weights, tokenizers, model configurations, and token caps. Decoder and
adapter differences are permitted for this explicit ablation mode. The legacy
three-system `compare` mode requires matching Qwen decoding settings.

## Validation Selection (Before Fresh Test)

| Variant | Correct / 616 | Accuracy | Macro-F1 | Invalid |
| --- | ---: | ---: | ---: | ---: |
| Expanded-data Naive Bayes | 440 | 71.43% | 0.7056 | 0 |
| Original LoRA, greedy | 380 | 61.69% | 0.6089 | 12 |
| Original LoRA, constrained | 382 | 62.01% | 0.6082 | 0 |
| Expanded-data LoRA, greedy | 469 | 76.14% | 0.7608 | 7 |
| Expanded-data LoRA, constrained | 469 | 76.14% | 0.7577 | 0 |

The predeclared macro-F1 rule selects **expanded-data LoRA with ordinary greedy
decoding**. The constrained variant is still reported separately: it guarantees
allowed outputs but did not improve validation accuracy, and its macro-F1 is
slightly lower. No model, prompt, dataset, or decoder settings will change for
the fresh-test comparison. That comparison will report all five fixed variants.

New validation token loss is 0.184113 (base 1.227113, old adapter 0.351555).
Reloaded adapters reproduce 0.184113 exactly. The new run took 2181.62 seconds
including validation/reload (about 36 minutes 22 seconds), not just optimization.

## Fresh-Test Results

Completed locally on October 8-9, 2026, following the selection above. All five
variants use the same 770 messages, none of which occurred by source ID or
normalized text in the initial 385-message pilot test. The validation-selected
variant remains expanded-data LoRA with ordinary greedy decoding; test results
were not used to change that choice or tune any settings.

| Variant | Correct / 770 | Accuracy | Macro-F1 | Invalid |
| --- | ---: | ---: | ---: | ---: |
| Expanded-data Naive Bayes | 548 | 71.17% | 0.7061 | 0 |
| Original LoRA, greedy | 465 | 60.39% | 0.5949 | 21 |
| Original LoRA, constrained | 470 | 61.04% | 0.5966 | 0 |
| Expanded-data LoRA, greedy | 581 | 75.45% | 0.7534 | 7 |
| Expanded-data LoRA, constrained | 581 | 75.45% | 0.7516 | 0 |

No run hit the token cap. Expanded-data greedy improves by **15.06 percentage
points** over original greedy: 147 formerly incorrect messages become correct,
while 31 formerly correct messages become incorrect, for 116 fewer errors net.
This measures more training data plus more updates, not a compute-controlled
data-only effect. The new model beats expanded-data Naive Bayes by **4.29 points**,
correcting 120 of its errors but missing 87 messages that Naive Bayes gets right.
Do not use the old 385-message pilot scores as a paired comparison with this set.

Constrained decoding adds five correct answers for the original adapters. For
the new adapters, it changes only the seven invalid outputs into allowed but
incorrect labels: all 581 correct and 189 incorrect decisions stay in those
groups. It fixes format, not the remaining semantic errors. Its lower macro-F1
reflects where those seven false-positive labels go, despite unchanged accuracy.

The selected model still makes **189 errors (24.55%)**: 182 wrong allowed labels
and seven invalid labels. Repeated confusions include `why_verify_identity` to
`verify_my_identity` (7), `pending_top_up` to `topping_up_by_card` (5), and
`card_payment_wrong_exchange_rate` to `wrong_exchange_rate_for_cash_withdrawal`
(4). There are only ten test examples per intent. No out-of-domain rejection,
confidence calibration, or general-capability retention was evaluated. These
results demonstrate task adaptation, not production readiness or a reason to
prefer an LLM regardless of inference cost.

## Runtime And Artifacts

| Variant | Mean inference | Peak process RSS |
| --- | ---: | ---: |
| Expanded-data Naive Bayes | 0.033 ms | 79.5 MiB |
| Original LoRA, greedy | 149.551 ms | 1061.1 MiB |
| Original LoRA, constrained | 140.608 ms | 1062.4 MiB |
| Expanded-data LoRA, greedy | 147.943 ms | 1064.3 MiB |
| Expanded-data LoRA, constrained | 141.429 ms | 1071.7 MiB |

Environment matches the [initial run](results.md): Apple Silicon, macOS 27.0,
Go 1.27.0, MLX 0.32.0, mlx-c 0.6.0_3. Naive Bayes runs on CPU; Qwen runs on GPU.
Test processes ran sequentially after training, with the last evaluation started
after an agent restart. Each number is a single local run, not an isolated speed
comparison. Loading and one unscored warm-up are excluded from inference time,
but included in peak RSS; RSS does not capture all Metal allocation. Training
took 2181.62 seconds including validation/reload, separate from inference cost.

Data and its parent manifest are under `benchmarks/banking77/data-v2/`.
Ignored local outputs under `checkpoints/banking77-v2/` include `training.log`,
the final `lora.safetensors`, resumable `training.safetensors`, and ten raw JSON
reports: `nb-{valid,test}.json` and `{old,new}-{greedy,labels}-{valid,test}.json`.
Each report includes every prediction, per-intent metrics, timings, and hashes.
Original data and adapters remain untouched.

The saved new adapters with constrained decoding returned `card_arrival` for
`My card still has not arrived`. This smoke test is not part of the scored set.

| Artifact | SHA256 |
| --- | --- |
| Expanded manifest | `c46fd01e788865202e30a3a84de8ff8046ece99eeb5e4ba2525086433d3401a5` |
| Frozen validation JSONL | `0e1882104d5bbf3893c07188ee5b34a5cbdb400a13b3613077fbe1525418aeb7` |
| New adapter export | `7db35fc57aad22d705c1a92d6e7a2b9dfb7f293181ac1902e5bb673be17a9dae` |
| New training checkpoint | `14f74dc8af572ac28aded2fca8ce6877afb2ab99f05048b6602cbc794d809300` |

Base weights, tokenizer, and original-adapter identities are recorded in the
[initial results](results.md). Recompute and verify the final comparison:

```sh
go run ./cmd/bench-banking -mode ablate -data benchmarks/banking77/data-v2 \
  checkpoints/banking77-v2/nb-test.json \
  checkpoints/banking77-v2/old-greedy-test.json \
  checkpoints/banking77-v2/old-labels-test.json \
  checkpoints/banking77-v2/new-greedy-test.json \
  checkpoints/banking77-v2/new-labels-test.json
```

These 770 test messages have now been inspected. Changes motivated by their
errors need a further independent evaluation; a retest is not unseen evidence.
