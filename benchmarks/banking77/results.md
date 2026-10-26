# Initial Fixed-Recipe Results

Run on 2026-10-08, using the [declared pilot protocol](README.md): 77 intents,
616 training examples, 616 validation examples, and 385 test examples. No
training setting, prompt, scoring rule, or checkpoint was changed after seeing
test predictions. These are subset results, not full BANKING77 benchmark scores.

## Classification

| Metric | Naive Bayes | Prompted Qwen | LoRA Qwen |
| --- | ---: | ---: | ---: |
| Correct labels | 207/385 | 55/385 | 240/385 |
| Accuracy | 53.77% | 14.29% | 62.34% |
| Macro-F1 | 0.5153 | 0.1105 | 0.6054 |
| Invalid labels | 0 | 73 | 4 |
| Token-cap hits | 0 | 0 | 0 |
| Mean inference time | 0.030 ms | 135.813 ms | 165.898 ms |
| Peak process RSS | 77.8 MiB | 750.5 MiB | 1004.9 MiB |

LoRA improved accuracy by **8.57 percentage points** over this simple baseline
and **48.05 points** over this fixed prompted-Qwen setup. Against Naive Bayes,
LoRA corrected 81 cases but broke 48 cases the baseline got right; both were
correct on 159 and wrong on 97. It is not uniformly better on every message.

The base result is not a claim that Qwen cannot classify intents: this is one
unchanged prompt listing all 77 labels, with strict output scoring. Better
prompting or candidate-selection methods could change its performance. Naive
Bayes is also only one untuned baseline, not the best available non-LLM method.

The tuned model still made **145 errors (37.66%)**: 141 wrong allowed labels and
4 invalid labels. Invalid outputs were `shipping_time` once and
`getting_physical_card` three times. Among the semantic confusions, four
`why_verify_identity` cases became `verify_my_identity`, and four
`compromised_card` cases became `card_swallowed`. These failures are retained
in the scores, not repaired by postprocessing.

This demonstrates useful task adaptation through mlxgo, but **does not justify
deployment**. There are only five test examples per intent, no out-of-domain
rejection, no confidence calibration, and no general-capability retention test.
It also does not establish that the accuracy gain is worth the much greater
inference cost compared with Naive Bayes. That depends on the application.

## Training And Runtime

Validation loss was **1.227113 -> 0.351555** after 308 AdamW steps. The saved and
reloaded adapters reproduced 0.351555 exactly. The run took 1415.27 seconds
(about 23 minutes 35 seconds), including initial/final validation and reload
verification, not just optimization. The adapter export is about 2.1 MiB and
the resumable checkpoint about 6.2 MiB. Base weights were not modified.

Environment: Apple Silicon macOS 27.0 (26A428), Go 1.27.0, MLX 0.32.0, and
mlx-c 0.6.0_3. Qwen runs used GPU; Naive Bayes used CPU. Evaluation processes
ran separately after training exited. Inference timing excludes dataset/model
loading and one unscored warm-up; RSS includes them and is not total Metal
allocation. These are single local measurements, not isolated latency or
throughput benchmarks. Training cost is separate from the inference table.

The new-message smoke test returned `card_arrival` for
`My card still has not arrived` with both the baseline and saved LoRA adapters.
This single demonstration is not included in the test scores.

## Local Artifacts

The downloaded source CSVs, license, prepared JSONL, and manifest are under
`benchmarks/banking77/data/`. Source preparation removed seven normalized
train/test overlaps and four repeated source-training rows before selection.
The pinned CSVs themselves are unchanged. Normalization detects exact
case/whitespace matches only, not semantic duplicates.

Ignored local outputs under `checkpoints/banking77/`:

- `nb-test.json`, `base-test.json`, `lora-test.json`: every prediction, per-intent
  metrics, timings, model/configuration identity, and dataset hashes.
- `training.log`: reported losses, validation/reload result, and process timing.
- `lora.safetensors`: inference adapters.
- `training.safetensors`: final resumable optimizer checkpoint.

SHA256 identities:

| Artifact | SHA256 |
| --- | --- |
| Prepared manifest | `bbb4f8d76202e61300003eed5a950daed7c406bd6eaaf1b8de6aff23ebc8edbe` |
| Base checkpoint | `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe` |
| Tokenizer | `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539` |
| Adapter export | `434ac18423223e409360a040f1e4144f6a5bd16d08cce6bc3937db8e14624117` |
| Training checkpoint | `464bb7846c370943feca53413f101b1e4d0ac755d9b4909d7fae8761b1231e60` |

Recompute summaries and verify report compatibility with:

```sh
go run ./cmd/bench-banking -mode compare \
  checkpoints/banking77/nb-test.json checkpoints/banking77/base-test.json \
  checkpoints/banking77/lora-test.json
```

These test examples have now been inspected. Any subsequent changes motivated
by their errors require a new independent evaluation set; do not present a
retest of these cases as unseen generalization.
