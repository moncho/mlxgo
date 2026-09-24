# Released Layer-2 Compressed Attention Validation

Validated on 2026-09-24 on Apple M3 Pro with MLX 0.32.0 / mlx-c 0.6.0_3
against PyTorch 2.14.0 CPU reference arithmetic. All weights are from the pinned
DeepSeek-V4.1-Flash release; inputs are deterministic test activations, not
hidden states obtained from a full pretrained model run.

## Coverage

Layer 2 is the first ratio-2 compressed-attention layer and owns both its
compressed KV cache and sparse indexer. Tests execute the existing
`Session.attention` path, including input normalization, the 128-token sliding
window, learned softmax compression, index-key creation, sparse selection,
attention sink, YaRN tail RoPE, inverse RoPE and grouped output projections.

The oracle executes checksum-pinned upstream `Attention`, `Compressor`,
`Indexer`, `RMSNorm`, rotary and window-index methods. Linear kernels use
decoded float32 weights. All activation/KV quantizers are explicitly disabled;
the CUDA sparse kernel is replaced by the existing PyTorch mathematical
translation. `wo_a` retains the release conversion's BF16 weight rounding.
BF16 compressor/indexer weights widen exactly to float32. No source correction
is needed for this isolated KV owner; reference shared state is reset between
schedules. Source config fields retain the released values; layer count and
allocated context are bounded for this component test.

Full prefill covers 131 tokens. Independent schedules prefill 1, 2, 127, 128
and 129 tokens, then decode two single tokens. They cross incomplete/complete
compression groups and window rollover, with another session interleaved.
Outputs, chronological window cache, compressed KV, index keys and pending
input agree with the reference. Cache lengths/positions match exactly and the
first zero input produces exactly zero, checking future-token masking.

A separate long compressor/indexer probe processes 1,280 inputs into 640 keys
using the real weights on both implementations. All keys are compared; Go's
generated keys then feed the production indexer. Queries at positions 1,022,
1,024 and 1,280 test selection below, at and above the released top-512 cutoff.
All selected index arrays match exactly, including pruning 640 keys to 512.
This is not a complete 1,280-token attention or model run.

## Reproduce

From the repository root, using the separate Python 3.12 / torch==2.14.0
environment described in `quant/README.md`:

```sh
go run ./cmd/fetch-deepseek-sample -set compressed-attention \
  -out models/deepseek-v41-attention2

/tmp/mlxgo-sample-reference-venv/bin/python deepseek/make_attention_reference.py \
  --layer 2 --samples models/deepseek-v41-attention2

export MLXGO_DEEPSEEK_COMPRESSED_ATTENTION_DIR="$PWD/models/deepseek-v41-attention2"
go test ./deepseek -run TestReleasedCompressedAttentionWeights -v -count=1
go test -tags "mlx mlxruntime" ./deepseek \
  -run TestReleasedCompressedAttentionForward -v -count=3
```

`--model-source /path/to/model.py` uses a local source copy and still verifies
its checksum. `--out /path/to/new.json.gz` supports separate regeneration.
The generator never overwrites an existing reference.

The download contains 22 raw tensors totaling 142,947,072 payload bytes from
shard 5, not the full 7.4 GB shard. Pinned header size/hash, exact byte ranges,
stable ETag, per-file lengths/hashes and manifest provenance are checked.
Requests are at most 4 MiB; the total budget is payload plus header size, capped
at 144 MiB. Failed downloads are removed and existing directories are preserved.
Metadata-only auditing keeps its original 64 MiB budget.

Decoded weights total 549,353,216 bytes (about 524 MiB). Native copies,
activations, fixtures and allocator overhead require additional memory.
This is neither a complete-checkpoint loader nor a process-wide memory limit.

## Results

All **137,338,304** decoded values match the reference hashes with signed zero
normalized. Native upload/readback preserves them on both devices.

| Comparison | CPU max absolute error | GPU max absolute error |
| --- | ---: | ---: |
| Full prefill vs reference | 4.0531158e-6 | 7.2717667e-6 |
| Cached schedules vs reference | 4.5299530e-6 | 7.5101852e-6 |
| Go cached vs full prefill | 8.8214874e-6 | 8.5830688e-6 |
| Window KV cache | 2.8312206e-6 | 4.4107437e-6 |
| Compressed KV cache | 2.5480986e-6 | 3.6954880e-6 |
| Short-schedule index keys | 2.9336661e-6 | 2.9541552e-6 |
| Long-probe index keys | 4.7862530e-5 | 4.7892332e-5 |

Per device: 2,703,360 output values are checked against the oracle, plus
2,032,640 cached/full comparisons, 341,632 short-schedule cache/pending values,
81,920 long-probe key values and 1,535 exact selected indices. Pending-input
maximum error is 2.9802322e-8 on both devices. Fixed tolerances remain
`2e-4*(1+abs(reference))` for output and `5e-5*(1+abs(reference))` for caches/keys.
The long-key maximum normalized error is 3.2483224e-5, below that fixed bound.

Three consecutive CPU/GPU runs passed with identical measured maxima. The full
stub and native runtime suites passed with all four real-weight sample sets
enabled. Ordinary CI does not download weights; these tests skip without the
environment variable and fail, rather than skip, when configured files are
missing or invalid. Existing synthetic model tests remain network-independent.

## Provenance and Limits

Release revision: `df42c109f1defefcbfcedbe7d905718a12266e40`.
Local manifest SHA-256:
`655a7c7542b6a671aed5e964c939bc5e4f01c1cd5bbed056b52ac52346fbf1f4`.
Reference gzip SHA-256:
`24edc3f81b8be9b32aae07cdea63af3565f5da78fcb6cdf22d11dbbd5ab23aa2`.
These are local integrity hashes, not upstream-published tensor checksums.
Regeneration was byte-identical. The extended generator also reproduced the
existing layer-0 reference byte-for-byte. Weights and generated references
stay in ignored `models/` directories.

This validates float32 component mathematics with real weights, not quantized
numerical parity or pretrained text generation. Released-weight cross-layer
cache consumers, later KV owners, ratio-1 compression, hierarchical candidate
filtering, mHC/block integration and complete model logits are not covered by
this layer-2 test. Those paths have smaller synthetic tests, but real-weight
integration and efficient quantized execution remain separate work.
