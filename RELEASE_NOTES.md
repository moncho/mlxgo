# v0.3.0

Device-side DeepSeek expert dispatch, constrained generation, and training
verification since v0.2.0. Native builds now require the mlx-c 0.7 API. This
remains a pre-1.0 source release with experimental model integrations.

## Added

- `GatherMM` and `GatherMXFP4`: matrix products selected by device-resident
  index arrays, with matching unavailable stubs for builds without MLX.
- DeepSeek float32 and packed FP4 expert banks, exposed through `DeviceMoE`
  and the opt-in `SessionOptions.DeviceDecode`. The initial prompt uses sparse
  dispatch; subsequent tokens select experts on the device. Banks are prepared
  once per model and shared by sessions. Default session behavior is unchanged.
- `lm.NewChoices` and `Choices.Greedy`: generation restricted to complete
  allowed token sequences. Include EOS in each choice when the model selects
  termination, and check errors before using a returned label. Successful
  generation guarantees membership, not classification accuracy or maximum
  whole-sequence likelihood.
- `cmd/bench-banking`: reproducible BANKING77 preparation, evaluation, and
  validation-based selection for prompted Qwen, LoRA, Naive Bayes, and TF-IDF
  logistic regression. Downloaded datasets and training artifacts are excluded
  from the release. The full-data search remains incomplete; no final test
  result is claimed for it.
- A tiny LitGPT/PyTorch training reference covering logits, masked losses,
  LoRA gradients, an AdamW update, and frozen base weights on CPU and GPU.
  Go tests consume the saved fixture without Python or model downloads.
- Opt-in tiny-model and Qwen2.5-0.5B training benchmarks with pinned reference
  tooling and saved measurements. These do not change the Qwen training API.

## Fixed

- `CumsumAxis` calls `mlx_cumsum_axis` with an unset optional dtype, preserving
  inclusive forward behavior with mlx-c 0.7. Its Go signature is unchanged.
- BANKING77 `compare` rejects report sets other than Naive Bayes, base Qwen,
  and LoRA Qwen instead of printing a zero-value row for a missing classifier.
  Use `ablate` for the supported comparisons across additional variants.
- Native CI race checks include the new DeepSeek bank/session concurrency tests
  and concurrent reuse of `lm.Choices`.
- The Qwen inference fixture is regenerated independently with MLX 0.32.4.
  The native upgrade changes greedy tokens near tied BF16 logits; the pinned
  checkpoint and 32-token/0.25-logit acceptance limits are unchanged.

## Installation

After the v0.3.0 tag is published:

```sh
brew install mlx-c
go get github.com/moncho/mlxgo@v0.3.0
go run -tags mlx ./your-app
```

Native support remains Apple Silicon macOS with cgo and external MLX/mlx-c
headers and libraries. mlx-c 0.6 is no longer supported. The local validation
baseline is Go 1.27.0, macOS 27.0.1, MLX 0.32.4, and mlx-c 0.7.0_1. CI targets
Go 1.26/1.27 on macOS 15, plus Linux stub builds. Homebrew dependencies are not
pinned by the Go module; see [compatibility](COMPATIBILITY.md).

## Validation

Local validation passed on the baseline above:

- `make release-check`: stub/native tests and vet, the complete native runtime
  suite, and the complete native race suite.
- BANKING77 comparison regression tests in default, native, and race builds,
  plus package vet after the report-validation fix.
- Independent upstream Qwen runs on MLX 0.32.0 and 0.32.4 reproduced their
  respective reference sequences. The refreshed 0.32.4 fixture matches all
  32 Go tokens with maximum prefill-logit error 0.03125, below the unchanged
  0.25 limit. Both tokenizer fixtures remain byte-identical.
- Real Qwen common-loader parity and temporary three-shard loading matched
  all 32 reference tokens. The 200-token RSS gate retained 90.2 -> 90.8 ->
  90.8 MiB, below the unchanged 64 MiB growth limit.
- Pinned local DeepSeek attention, compressed/shared attention, output/query
  projection, expert, and quantized-cache checks; the separate quantization
  suite also passed with the real activation reference enabled.
- The five core examples, MLP fine-tuning/reload, DeepSeek quantized-cache
  smoke, and exported float32/Engram bundles through the generation CLI.
- `go mod tidy -diff` and whitespace checks reported no changes or errors.

Real adapter-loader parity and saved training-checkpoint artifact comparison
were not enabled. Performance benchmarks and the unfinished BANKING77 search
were not rerun. Public CI does not download these local real-weight fixtures.

The exact candidate must pass the complete CI matrix before tagging. The tag
workflow reruns that matrix before publishing the source release. Updating
these files does not publish a tag or GitHub release.

## Known Limits

- Device decode is inference-only. Each layer must have wholly float32 or
  wholly FP4 experts, including its shared expert. Banks remain resident and
  are released with the model. This does not add released DeepSeek checkpoint
  support or expert offloading.
- Saved DeepSeek measurements use untrained synthetic models; saved Qwen
  training measurements use the documented FP32 workloads. Both were recorded
  with MLX 0.32.0 and mlx-c 0.6.0_3, not the current release baseline. They are
  historical measurements, not new performance claims for mlx-c 0.7.
- The larger Qwen training comparison records strict adapter-value mismatches
  despite close losses. It is numerical characterization, not a full tensor
  parity pass. The tiny fixture's acceptance limits are unchanged.
- BANKING77 results are bounded subset experiments. Constrained labels do not
  establish task accuracy, and the unfinished full-data search must not be
  presented as a completed evaluation.
- Native work remains serialized on one OS thread. Device selection and the
  sampling RNG remain process-wide. Existing ownership, streaming, and model
  support limits are described in [compatibility](COMPATIBILITY.md).

The release contains Go source and small reference fixtures, not native
libraries, downloaded datasets, or model checkpoints. See
[third-party notices](THIRD_PARTY_NOTICES.md) for bundled reference material.
