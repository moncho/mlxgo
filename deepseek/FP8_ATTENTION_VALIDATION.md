# Packed WKV In Real Attention

An explicit `ModelOptions.FP8AttentionKV` entry now replaces the corresponding
`attn.wkv.weight` projection during both prefill and incremental decoding.
`NewModel` and common checkpoint loading remain float32 by default. The model
owns the packed weights, does not retain float32 duplicates, and shares them
read-only across sessions. Cache quantization remains a separate session option.

This is **weight-only FP8**, with float32 activations and all other projections
unchanged. It is not DeepSeek's activation-quantized GEMM, BF16/CUDA parity,
full released-model loading, or pretrained generation.

## Real Reference

The test uses the existing [layer-0 attention sample](ATTENTION_VALIDATION.md),
including its pinned release revision, weight hashes and independent PyTorch
2.14.0 oracle. Packed `wkv` bytes and block scales are hash-verified directly;
the other weights retain their decoded-reference checks. The test runs the
actual `Session.attention` implementation, not an alternative attention kernel.

Coverage on both CPU and GPU:

- Full 131-token prefill.
- Prefills of 1, 127, 128 and 129 tokens, followed by two incremental tokens.
- Rollover of the 128-token sliding window, chronological cache values and lengths.
- Causality, cached/full agreement, and interleaved independent sessions.

The original float32 budgets are unchanged: output error must be at most
`2e-4 * (1 + abs(reference))`; cache error at most `5e-5 * (1 + abs(reference))`.
No quantized-cache tolerance is substituted into this test. Worst observed
scaled errors against the independent reference were:

| Backend | Attention output | Chronological KV cache |
| --- | ---: | ---: |
| CPU | 2.690e-6 | 1.822e-6 |
| GPU | 2.718e-6 | 1.839e-6 |

Download-free tests construct a two-layer model with one packed and one float32
`wkv`, compare full logits against the decoded-float32 model, and exercise both
ordinary and quantized caches. They verify copied caller buffers, independent
array ownership, model-close behavior, rejection of malformed/duplicate weights,
rollover and eight concurrent sessions. The concurrent tests run in CI under the
race detector. These synthetic tests check integration, not pretrained quality.

Real validation in this milestone covers layer 0 with float32 caches. Combining
real packed projections with real quantized caches, compressed-attention layers,
and other FP8 projections needs separate validation before parity claims.

## Whole-Attention Benchmark

Apple M3 Pro, MLX 0.32.0, mlx-c 0.6.0_3, Go 1.27.0. Medians of three 300ms
benchmark runs, three warmups per case. No other inference benchmark ran
concurrently. Inputs are deterministic synthetic activations with real weights.

Each timed step includes input RMS normalization, all query/KV/output projections,
RoPE, sparse attention, cache updates, native evaluation and session cleanup in
one `mlx.Batch`. Weights and input arrays are uploaded before timing. No tensor
readback to Go is timed. Prefill starts with an empty session and consumes 128
tokens. Decode clones the same already-evaluated 128-token cache into a fresh
session and consumes token 128, including window rollover. Cloning the cache
handle is timed equally for both paths; the template prefill is excluded.

| Backend | Workload | Float32 | Packed WKV | Float32 / packed time |
| --- | --- | ---: | ---: | ---: |
| CPU | 128-token prefill | 78.701 ms | 125.679 ms | 0.63x |
| CPU | Decode after 128 tokens | 5.441 ms | 5.711 ms | 0.95x |
| GPU | 128-token prefill | 12.077 ms | 12.004 ms | 1.01x |
| GPU | Decode after 128 tokens | 4.358 ms | 4.209 ms | 1.04x |

**No material GPU whole-attention speedup is established.** GPU decode ranged
from 4.238-4.683 ms for float32 and 4.200-4.244 ms for packed WKV, with overlap.
CPU prefill is about 60% slower. This is why the option stays explicit rather
than becoming a default. The earlier [isolated projection gain](quant/FP8_LINEAR_VALIDATION.md)
does not extrapolate to the entire attention block.

### Memory

Retained payload for all weights in this attention sample falls from
506,490,112 to 498,707,712 bytes: **7,782,400 bytes (7.78 MB), or 1.54%**.
Only the `wkv` matrix itself is 3.88x smaller. The other matrices dominate.

The benchmark also samples MLX's allocator, subtracting active allocations
present before loading the sample and resetting the global peak after warmup.
Peak includes weights, inputs, outputs, caches and evaluation temporaries.
It excludes idle allocator cache, Go heap buffers, loading-time peaks and
process RSS. No other model is intentionally kept active during each case.

Median peak active bytes, shown in decimal MB:

| Backend | Workload | Float32 | Packed WKV |
| --- | --- | ---: | ---: |
| CPU | 128-token prefill | 700.968 MB | 693.186 MB |
| CPU | Decode after 128 tokens | 511.153 MB | 503.362 MB |
| GPU | 128-token prefill | 691.599 MB | 683.817 MB |
| GPU | Decode after 128 tokens | 511.652 MB | 503.870 MB |

Steady active bytes after prefill warmup and session cleanup were 509.124 MB
versus 501.341 MB on both devices. Decode retains a template cache as well;
GPU steady readings varied by less than 0.5 MB across runs as temporary resources
were released. These allocator snapshots are not whole-process memory savings.

## Reproduce

Use the existing sample and independent reference described in
[attention validation](ATTENTION_VALIDATION.md). Tests do not download weights.

```sh
export MLXGO_DEEPSEEK_ATTENTION_DIR="$PWD/models/deepseek-v41-attention0"
go test -tags 'mlx mlxruntime' ./deepseek -run '^TestReleasedFP8AttentionForward$' -v
go test -tags 'mlx mlxruntime' ./deepseek -run '^TestFP8Model'
go test -race -tags 'mlx mlxruntime' ./deepseek -run '^TestFP8Model'
go test -tags 'mlx mlxruntime' ./deepseek -run '^$' \
  -bench '^BenchmarkReleasedAttentionFP8$' -benchtime=300ms -count=3
```

`mlx.GetMemoryUsage` exposes active, cache and peak byte counters;
`mlx.ResetPeakMemory` resets only the process-wide peak statistic. Coordinate
profiling with other MLX users and evaluate work before reading counters.

The subsequent [query projection report](FP8_QUERY_VALIDATION.md) validates
packed `wq_b` independently and together with `wkv`. The numbers above remain
the original KV-only measurements. The [grouped output report](FP8_OUTPUT_VALIDATION.md)
separately validates `wo_a` with a BF16-exactness guard. Packed FP4 experts still
need separate validation; do not infer their correctness or performance from
these replaced matrices.
