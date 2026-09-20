# Experimental DeepSeek Text Backbone

This package runs a reduced, unquantized DeepSeek-V4.1 text backbone with real
prefill and incremental cache paths. **It cannot load the released checkpoint,
generate meaningful text with the included untrained weights, or fine-tune the
released model.** All parameters must currently be float32. Engram is supported
with prepared hash metadata and unquantized tables. Vision, DSpark, production
quantization and pretrained tokenizer integration remain unsupported.

The implementation and tests are pinned to the
[official inference source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/df42c109f1defefcbfcedbe7d905718a12266e40/inference),
revision `df42c109f1defefcbfcedbe7d905718a12266e40`. No released weights are
downloaded by the tests. The source license is in `THIRD_PARTY_LICENSE`.

## Run The Reduced Model

```sh
go run -tags mlx ./cmd/deepseek-smoke -device cpu
go run -tags mlx ./cmd/deepseek-smoke -device gpu
go run -tags mlx ./cmd/deepseek-smoke -device gpu -fixture deepseek/testdata/model_engram.json
```

The example uses checked-in deterministic synthetic weights: 8 layers, 16,488
parameters, vocabulary size 32, and a 32-position context. It prints generated
token IDs, not text. No model download or Python environment is required.
The Engram-enabled fixture has 20,709 parameters and inserts memory contributions
at layers 0, 3 and 6. It is a separate untrained model, not a conversion of the
released weights.

To save these synthetic weights as a local bundle, pass
`-export models/deepseek-synthetic` (the directory must not already exist).
The [common loader and generation command](../inference/README.md) can then
load it by configuration and generate raw token IDs. This does not convert or
add support for released weights.

`NewModel(config, parameters)` validates all parameter names, shapes and dtypes
and retains its own handles. `Config.ParameterShapes` defines the float32
parameter contract. `Model.Forward` computes all logits in a temporary session.
`Model.NewSession` creates independent incremental state. The model is immutable
and may be used by separate sessions concurrently; do not share a session across
goroutines. Sessions borrow the model, which must remain open until they finish.

Both DeepSeek and `qwen2.NewSession` implement `lm.Session`: `Step`, `Position`
and `Close`. The same `lm.Greedy` decoder works with either architecture. Input
IDs, tokenization, checkpoint loading and cache layout stay architecture-specific.
`Step` returns caller-owned `[1,1,vocab] logits for the last consumed token.
The final generated token has not yet been consumed by the session.

DeepSeek accepts a nonempty prompt on the first call and one token per subsequent
call. `ForwardAll` returns logits for every consumed position. Successful calls
evaluate logits and caches before advancing the position. Native failures
invalidate the session; invalid tokens, context overflow and invalid chunk sizes
are rejected before mutation. Close each session and its model when finished.

This is an inference path, not a whole-model training API. The individual
primitive functions below remain differentiable.

## Engram

`EngramHasher` is a pure-Go, per-sequence state machine. It remaps original IDs
through a prepared compressed token map, XORs exact int64 products for n-grams
of length 2 through `MaxNGram`, and places each head in a disjoint prime-sized
bucket range. It retains only the last `MaxNGram-1` compressed IDs. Padding uses
the remapped `PadID`; a false token mask blocks lookback across that position,
including across chunk boundaries. Invalid input leaves history unchanged.
Separate sequences need separate hashers; hashers are not shared concurrently.

`Engram` takes float32 residuals `[batch,seq,streams,dim]`, table indices, an
optional text mask, and `EngramWeights`. It gathers and concatenates table rows,
projects per-stream keys and a shared value, normalizes queries/keys per stream,
then applies the reference's signed-square-root sigmoid gate. Masked positions
pass through unchanged. Inputs are borrowed and the result is caller-owned.
Gradients propagate through the residual, table, projection and normalization
weights; the integer hash is not differentiable. The complete model remains an
inference API, and its public token-only path supplies an all-text mask.

The optional `Config.Engram` contains the token map, compressed vocabulary size,
original pad ID, layers, primes, table sizes, head dimensions and multipliers.
`ParameterShapes` adds `layers.N.engram.{embed.weight,wkv.weight,q_weight,k_weight}`.
The model inserts Engram before each selected transformer block. Session-owned
history is isolated even when several models or sessions are interleaved.
Existing configurations without `engram` are unchanged. The common loader and
`deepseek-smoke -export` preserve this metadata and validate required weights.

**Prepared metadata is deliberate:** the official token map depends on decoded
token strings and tokenizer normalization, while hash multipliers depend on
NumPy's per-layer RNG and the compressed vocabulary size. The Go runtime does
not approximate either with a different tokenizer or RNG. Export their exact
values with the matching tokenizer, and preserve multipliers as JSON integers
(many exceed the exact range of float64). The fixture generator executes the
pinned reference's normalization, prime selection and multiplier generation on
explicit synthetic decoded/raw token strings. This verifies remapping and
hashing, not compatibility with the released tokenizer.

The released Engram table uses FP8 rows and scales. Replacing that storage with
float32 in these tests intentionally excludes dequantization and BF16 rounding.
Mask handling is tested at the component level; it does not add vision support.

## Implemented And Tested

| Component | Coverage |
| --- | --- |
| `Route` | sqrt-softplus, sigmoid and softmax scores; temperature; selection-only correction bias; top-k; normalization and scaling |
| `MoE` | Routed and shared SwiGLU experts; asymmetric activation clamps; routing weights applied before down projection |
| `HyperMix` | Projection normalized over all residual streams; pre/post coefficients; Sinkhorn combination matrix |
| `HyperPre`, `HyperPost` | Residual collapse and expansion, including input/output stream orientation |
| `SparseAttention` | Shared key/value vectors, explicit sparse indices, ignored -1 slots, entirely empty rows, per-head denominator-only sink |
| `CompressComplete` | Complete-group learned softmax pooling before RoPE, RMS normalization, ratio-1 projection path |
| `EngramHasher` | Exact integer hashes, normalization collisions, padding, dead-token boundaries, arbitrary chunking and bounded history |
| `Engram` | Float32 lookup/projection, per-stream normalization, signed square-root floor, masking, forward values and gradients for every weight/input |

All operations use Go MLX bindings, including the added precision-preserving
`Log1p` operation for routing's softplus. They retain the computation
graph and run inside `mlx.Batch`; there are no host reads of routing scores.
CPU/GPU tests compare forward results and input gradients to reference fixtures.
An eight-goroutine build/evaluate test runs on both backends.

Model tests additionally cover:

- Low-rank queries, grouped output projection, tail RoPE and inverse output RoPE.
- Single-pass mHC with the previous sublayer's pre-mix, including final collapse.
- Bounded sliding-window state, partial compression groups, ratio-2 to ratio-1
  source transitions, reindex layers and layers that reuse prior selections.
- Hierarchical candidate blocks, causal compressed-position visibility and top-k.
- Fourteen reference schedules: prefill lengths 1, 2, 3, 4, 5, 8 and 13, with and
  without nontrivial YaRN extrapolation, on CPU and GPU.
- Interleaved sessions from two different models, context limits, validation,
  cleanup, and shared greedy generation versus uncached recomputation.
- The same fourteen schedules with Engram enabled at three layers, including
  CPU/GPU concurrent-session isolation and shared-loader safetensors round-trip.

Each logit is compared with absolute tolerance `2e-5`; observed reference errors
on the development Mac are below `2e-6`. Reference ties in sparse top-k are not
specified identically across backends, so the fixture avoids boundary ties.

The MoE implementation evaluates all experts and masks their contributions. That
is intentional for small correctness tests and differentiability; its cost grows
with **all** experts, not only the selected experts. It is not a production MoE
implementation. Sparse attention materializes gathered keys/values and scores;
it is not a fused sparse kernel. Repeated argmax implements expert top-k; ties
choose the lowest index, unlike PyTorch's unspecified tie ordering. The sparse
indexer uses full sorting, not an optimized partial selection kernel. Window
caches retain a chronological, bounded slice instead of a physical ring; this
preserves attention semantics but copies on updates. Compressed caches append
arrays. These choices target small correctness workloads, not production speed.

Arrays passed to these functions are borrowed. The caller owns every returned
array and must close it. Projection matrices have `[output,input] orientation.
The APIs reject non-float32 arrays instead of silently claiming quantized or
BF16 compatibility.

## Architecture Audit

The released model is not a Qwen-compatible checkpoint. Its configuration and
reference code introduce these requirements beyond memory capacity:

| Area | What remains |
| --- | --- |
| CSA2 attention | Float32 structure and cache coordination implemented; optimized storage and kernels remain |
| Sparse indexer | Float32 scoring, candidate filtering and sharing implemented; quantized scoring and optimized top-k remain |
| Single-pass mHC | Complete block wiring implemented and compared through model logits |
| Engram | Float32 remapping/hash/lookup/gating implemented; released tokenizer metadata, FP8 table loading and rounding remain |
| Checkpoint storage | Sharded names and shapes, mixed BF16/FP8/packed FP4, associated scaling conventions; not interchangeable with generic MLX affine quantization |
| Cache quantization | Separate formats for sliding-window KV, compressed KV and index keys; disabling weight quantization alone does not disable these |
| Text input/output | Official encoding rules and tokenizer integration; no assumption that the existing Qwen chat formatting applies |
| Vision | DeepSeek-ViT, image processing, projection and vision-specific routing bias |
| DSpark | Draft path and its state; the released minimal reference itself does not supply a speculative generation loop |

The architecture has a 40-layer causal encoder/decoder backbone with different
cache ownership patterns across layers. Replacing its attention with ordinary
Qwen attention would test a different model, even if the dimensions matched.
Source: the pinned `model.py`, `kernel.py`, and
[inference configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/inference/config.json).

The reduced-backbone acceptance tests now compare full prefill with incremental
decode and reference logits, with and without float32 Engram. Released
checkpoint storage, tokenizer integration and production quantization still
need implementations and tests before pretrained support can be claimed.

## Reference Method

`make_fixture.py` verifies SHA-256 hashes of the pinned official source before
extracting and executing `Gate.forward`, `Expert.forward`, `MoE.forward`,
`Compressor.forward`, and `Block.hc_*`. Constructors are replaced with small,
deterministic float32 tensors. These are component tests.

`make_model_fixture.py` additionally executes the official `Transformer`,
`Block`, `Attention`, `Indexer` and compressor/cache methods from the verified
source AST. Float32 Linear replaces quantized projections; cache quantization
is disabled; the head returns all positions. Engram is off by default; `--engram`
enables its original constructor/forward and `NgramHashState` with prepared
synthetic tokenizer metadata and a float32 embedding replacement. Vision and
DSpark stay off. The reference cache-publication correction below applies to
both model fixtures.

`make_engram_fixture.py` runs the unmodified pinned `engram.py` for token-map
normalization, prime layout, multiplier generation and incremental hashing.
It also executes `Engram.forward` from `model.py`, replacing only quantized
embedding/projection storage. Fixtures include two masked/unmasked batch rows,
four chunk schedules, gradients for every differentiable argument and gate
edge cases around zero. The downloaded `engram.py` is SHA-256 verified as well.
The same CPU kernel translations are shared through `reference.py`.

### Explicit Reference Correction

The pinned source's `Indexer.forward` publishes `shared_attn.index_k` only when
it produces a new latent. During an incomplete compression group, this can leave
the previous token's later KV owner's cache in that global slot. Thus its own
full-prompt and token-by-token runs disagree for this source-sharing schedule.

The generator first reproduces that failure with the unmodified cache logic.
Then an explicitly recorded pre-forward hook publishes the current owner's
existing `k_cache` on every indexer invocation. It does not change scoring or
selection formulas. The fixture records the adjustment and unpatched error;
its corrected reference verifies full/incremental agreement before writing
fixtures. This is comparison against an **adjusted reference**, not a claim of
bitwise parity with the unmodified release. Go caches are session-local and
always read the current owner, so they do not use that global slot.

Configurations must also provide enough candidate capacity for top-k when the
newest candidate block has only one position. Otherwise the reference can fill
unused top-k slots with masked candidates. Such configurations are rejected.

The CUDA/TileLang sparse-attention and Sinkhorn kernels cannot run on this Mac.
Their fixtures use explicitly identified PyTorch mathematical translations.
Consequently these tests establish formula agreement, **not** CUDA kernel
parity or quantized numerical parity. They omit BF16 accumulation rounding,
FP4/FP8 quantization, released weights and full-sized production-model outputs.

Generate fixtures in a separate Python 3.12 environment with `torch==2.14.0`:

```sh
python deepseek/make_fixture.py
python deepseek/make_model_fixture.py
```

For the Engram fixtures, also install `numpy==2.5.3`, `sympy==1.14.0` and
`tokenizers==0.22.2` in that separate environment:

```sh
python deepseek/make_engram_fixture.py
python deepseek/make_model_fixture.py --engram
```

`--source-dir /path/to/inference` uses already downloaded sources with the same
hash checks. The fixture records the source revision, hashes, seed, dtype,
PyTorch version and limitations. Ordinary tests need no Python or network.

```sh
go test ./deepseek
go test -tags "mlx mlxruntime" ./deepseek
```

The repository's existing recursive native runtime CI job includes these tests.
