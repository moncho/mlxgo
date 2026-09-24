# Released Layer-2/3 Shared Attention Validation

Validated on 2026-09-24 on Apple M3 Pro with MLX 0.32.0 / mlx-c 0.6.0_3 and
PyTorch 2.14.0 CPU reference arithmetic. This extends the real layer-2 test with
its first real cache consumer, layer 3. No production attention changes were
needed: the existing shared-state path passes these checks.

## Scope

The test builds this attention-only composition from released weights:

```text
deterministic inputs -> norm2 -> attention2 -> norm3 -> attention3
```

Layer 2 owns the compressed KV, index keys, pending compression group and
selected positions. Layer 3 reuses that owner's KV and selections, but has its
own sliding-window KV. It has no compressor or indexer weights.

Both calls use the same per-forward `attentionState` and stay in one lazy MLX
graph until outputs and caches are evaluated together. Each decode step gets
a fresh shared-state object, just as the model path does. Assertions check
that layer 3 does not replace the owner, its cache handles/lengths or the
shared selection, and never allocates private compressed/index/pending state.
Final cache contents are also compared against the reference, not just handles.

The oracle executes the checksum-pinned upstream attention, compressor,
indexer, normalization, rotary and cache methods. All activation/cache
quantizers remain disabled. Quantized linears use decoded float32 weights;
`wo_a` retains BF16 weight rounding. Sparse attention uses the existing CPU
mathematical translation rather than the CUDA kernel. The reference's global
shared state is reset between schedules. There is only one KV owner in this
pair, so no source correction is required.

This is **not two complete transformer blocks**: there is no residual, mHC or
FFN operation between the attention calls. Inputs are deterministic test
activations, not hidden states from a pretrained full-model run.

## Reproduce

Keep the layer-2 sample and reference from `COMPRESSED_ATTENTION_VALIDATION.md`.
Only layer 3 requires another download:

```sh
go run ./cmd/fetch-deepseek-sample -set consumer-attention \
  -out models/deepseek-v41-attention3

/tmp/mlxgo-sample-reference-venv/bin/python deepseek/make_attention_reference.py \
  --layer 2 --samples models/deepseek-v41-attention2 \
  --consumer-samples models/deepseek-v41-attention3

export MLXGO_DEEPSEEK_COMPRESSED_ATTENTION_DIR="$PWD/models/deepseek-v41-attention2"
export MLXGO_DEEPSEEK_SHARED_ATTENTION_DIR="$PWD/models/deepseek-v41-attention3"
go test ./deepseek -run TestReleasedSharedAttentionWeights -v -count=1
go test -tags "mlx mlxruntime" ./deepseek \
  -run TestReleasedSharedAttentionForward -v -count=3
```

Use the separate Python 3.12 / torch==2.14.0 environment documented in
`quant/README.md`. `--model-source /path/to/model.py` works offline and still
checks source integrity. `--out /path/to/new.json.gz` regenerates independently;
existing output files are never overwritten.

The downloader fetches 14 tensors, exactly 126,753,280 payload bytes from shard
6, not the whole 7.4 GB shard. Pinned header size/hash, exact ranges, stable
ETag, file hashes and failure cleanup use the existing bounded downloader.
The new manifest cannot be mistaken for layer 0's, even though those shards
have the same total file size. Tests cover this rejection and exact byte counts.

Combined layer-2/3 decoded weights contain 263,960,832 float32 values, requiring
1,055,843,328 bytes before native copies, activations, fixtures and allocator
overhead. The sample download is not a complete-checkpoint loader or a
process-wide memory budget.

## Results

All 126,622,528 newly decoded layer-3 values match their reference hashes, with
signed zero normalized. Native upload/readback checks cover both layers on
both devices. The consumer reference records the owner's manifest hash and
the exact attention-only composition; tests require the matching owner sample.

Full prefill covers 131 tokens. Five independent schedules prefill 1, 2, 127,
128 or 129 tokens and decode two individual tokens. Another session is
interleaved between calls. Two additional goroutines run the 1- and 129-token
schedules concurrently, each with its own interleaved session.

| Consumer comparison | CPU max absolute error | GPU max absolute error |
| --- | ---: | ---: |
| Full output vs reference | 5.0067902e-6 | 7.8678131e-6 |
| Cached outputs vs reference | 5.6624413e-6 | 8.7022781e-6 |
| Cached vs full output | 1.0013580e-5 | 1.1682510e-5 |
| Chronological window KV | 3.2186508e-6 | 4.7683716e-6 |

The owner is checked separately against the existing layer-2 oracle, including
outputs, window KV, compressed KV, index keys and pending normalized input.
Cache lengths and positions must match exactly. The first zero input produces
exactly zero at both layers, guarding against future-token visibility.
Tolerances remain `2e-4*(1+abs(reference))` for outputs and
`5e-5*(1+abs(reference))` for caches, unchanged from the individual-layer tests.

Three consecutive CPU/GPU runs passed, including concurrent sessions. Full
stub and native runtime suites passed with all five real-weight sets enabled.
The consumer downloader tests also pass under the race detector. Real tests
remain opt-in with no automatic weight downloads. Once sharing validation is
enabled, a missing owner setting or missing/corrupt files fails rather than skips.

## Provenance and Limits

Release revision: `df42c109f1defefcbfcedbe7d905718a12266e40`.
Layer-3 manifest SHA-256:
`9fa15d36cdb3accb732d17ee781b31f7824c130d8a63d51c24f59631db9f5fed`.
Consumer reference gzip SHA-256:
`ba5c1ad1cef797b1c0f72ba7a1943be14755607dcac27fcbcf02a7c03f97007f`.
These are local integrity hashes, not upstream-published tensor checksums.
The reference regenerated byte-for-byte. Existing layer-0 and layer-2
references also reproduce unchanged after extracting the shared weight loader.
Downloaded weights and generated references stay in ignored `models/` folders.

The pair has only one cache owner and at most 65 compressed positions, below
the top-512 cutoff. Layer 2's separate long-indexer test covers pruning, but
this pair does not validate long-context cross-layer pruning, later owner
transitions, ratio-1 attention or hierarchical candidates with real weights.
Quantized activation/cache parity, complete real blocks, tokenizer integration
and full-model pretrained generation remain separate milestones.
