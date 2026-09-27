# Combined Packed Weights And Caches

Packed `wkv`, `wq_b` and grouped `wo_a` now pass the existing real attention
references both with ordinary caches and with `QuantizedCaches: true`.
This covers layer 0, compressed owner layer 2 and the layer-2/3 sharing pair
on CPU and GPU. The reference revisions, payload hashes, BF16-exactness guard
for `wo_a` and numerical budgets are unchanged.

This is component validation with real weights and deterministic synthetic
activations, not pretrained generation or full transformer-block parity.
Only the selected attention projections use weight-only MXFP8. Compressor and
indexer projections remain float32. The shared test composes attention sublayers,
not mHC/residual/FFN blocks.

## Correctness Coverage

- Full 131-token prefill and cached schedules around the 128-token window.
- Partial compression groups, chronological window/compressed/index caches,
  pending inputs and compressed cache ownership.
- Exact sparse selection probes over 640 compressed index keys, including top-512.
- Causality, interleaved sessions, eight concurrent single-layer sessions, and
  concurrent short/rollover schedules for the sharing pair.
- Float32-cache output/cache budgets remain `2e-4`/`5e-5` scaled by
  `1+abs(reference)`. Quantized paths use the existing maximum scaled error
  `0.05`, RMS scaled error `0.002`, and at most `0.1%` differing bins for upstream
  decoded caches. Consumer caches have propagated input differences and use
  the existing output-style budget, not the upstream bin-count restriction.

Every case passes without widening its budget. Quantization is discontinuous
at bin thresholds, so this establishes bounded compatibility, not bitwise
CUDA/BF16 parity. Each prefill/decode schedule is checked against its own
independent quantized reference rather than forcing bitwise cached/full equality.

The download-free mixed-model tests also combine packed FP8 attention, FP4
experts and packed caches, including a synthetic compressed owner/consumer pair.
The CI race job includes these tests; real-weight tests remain opt-in and never
download payloads.

## Measurements

Apple M3 Pro, MLX 0.32.0, mlx-c 0.6.0_3, Go 1.27.0. Medians of three 300ms
runs, with no concurrent inference workload. CPU packed-prefill runs exceed
300ms and therefore contain one timed iteration each. All measurements below
are component timings with 128-token context, not full-model throughput.

Layer 0 separates the contributions of weight and cache packing (milliseconds):

| Backend | Workload | Float32 | Packed weights | Packed caches | Combined |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU | Prefill 128 | 76.289 | 1610.683 | 80.457 | 1620.772 |
| CPU | Decode 1 | 5.470 | 14.621 | 6.713 | 15.988 |
| GPU | Prefill 128 | 12.103 | 12.921 | 12.638 | 13.452 |
| GPU | Decode 1 | 4.302 | 2.640 | 4.955 | 3.344 |

Compressed and shared attention (milliseconds):

| Path | Backend | Workload | Float32 | Combined |
| --- | --- | --- | ---: | ---: |
| Layer 2 | CPU | Prefill 128 | 87.762 | 1654.745 |
| Layer 2 | CPU | Decode 1 | 5.870 | 17.695 |
| Layer 2 | GPU | Prefill 128 | 14.071 | 18.367 |
| Layer 2 | GPU | Decode 1 | 4.532 | 4.302 |
| Shared 2-3 | CPU | Prefill 128 | 173.195 | 3284.778 |
| Shared 2-3 | CPU | Decode 1 | 11.372 | 34.220 |
| Shared 2-3 | GPU | Prefill 128 | 26.503 | 32.361 |
| Shared 2-3 | GPU | Decode 1 | 8.481 | 7.212 |

Weight payload falls from 506.490 to 274.575 MB for layer 0, 549.353 to
317.438 MB for layer 2, and 1055.843 to 592.012 MB for the pair. Cache packing
does not change those weights. Median peak active allocator bytes (decimal MB):

| Path | Backend | Prefill float32 | Prefill combined | Decode float32 | Decode combined |
| --- | --- | ---: | ---: | ---: | ---: |
| Layer 0 | CPU | 700.968 | 456.013 | 511.178 | 281.111 |
| Layer 0 | GPU | 691.599 | 501.366 | 511.655 | 283.004 |
| Layer 2 | CPU | 769.653 | 524.295 | 554.829 | 323.361 |
| Layer 2 | GPU | 777.761 | 621.193 | 555.408 | 327.239 |
| Shared 2-3 | CPU | 1276.888 | 799.094 | 1061.859 | 598.412 |
| Shared 2-3 | GPU | 1487.772 | 1088.921 | 1064.972 | 602.733 |

At this context, **GPU weight-only packing is a better layer-0 latency choice
than combining it with cache packing**. Adding caches raises decode from
2.640 to 3.344 ms and peak active memory from 279.772 to 283.004 MB despite
reducing retained window-cache payload. Temporary work matters. Shared-pair
combined decode is about 15% faster than float32, but prefill about 22% slower.
CPU combined execution is substantially slower throughout. Keep defaults
float32 and benchmark actual context/device needs before selecting cache packing;
no long-context speed or full-model memory claim follows from this experiment.

## Reproduce

Use the existing local samples and float32/quantized references documented in
[quantized cache validation](QUANTIZED_CACHE_VALIDATION.md).

```sh
export MLXGO_DEEPSEEK_QUANTIZED_CACHES=1
export MLXGO_DEEPSEEK_ATTENTION_DIR="$PWD/models/deepseek-v41-attention0"
export MLXGO_DEEPSEEK_COMPRESSED_ATTENTION_DIR="$PWD/models/deepseek-v41-attention2"
export MLXGO_DEEPSEEK_SHARED_ATTENTION_DIR="$PWD/models/deepseek-v41-attention3"
go test -tags 'mlx mlxruntime' ./deepseek -run '^TestReleasedCombined' -v
go test -race -tags 'mlx mlxruntime' ./deepseek -run '^TestFP4Model'
go test -tags 'mlx mlxruntime' ./deepseek -run '^$' \
  -bench '^BenchmarkReleasedCombinedLayers$' -benchtime=300ms -count=3
```

The benchmark uses three warmups and fresh sessions per iteration. Prefill
consumes 128 tokens; decode clones every retained cache handle from an evaluated
128-token template and consumes the next token, including rollover. Template
prefill and weight uploads are excluded; cache cloning, input normalization,
attention, cache updates, evaluation and cleanup are timed inside `mlx.Batch`.
No Go output readback is timed. Shared cases include both attention sublayers.
They are not whole-model tokens/second. MLX peak counters exclude idle allocator
cache, Go buffers, loading peaks and RSS, and subtract pre-existing active bytes.

Defaults remain float32. Cache compression reduces retained cache payload but
adds encoding/decoding work and temporaries; memory savings alone do not justify
enabling it for every context length or device.
