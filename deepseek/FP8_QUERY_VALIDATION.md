# Packed Query Expansion In Real Attention

`ModelOptions.FP8AttentionQB` optionally replaces `layers.N.attn.wq_b.weight`
with checkpoint E4M3FN bytes and E8M0 block scales. Its logical shape is
`[Heads*HeadDim, QRank]`. It can be selected independently of `FP8AttentionKV`.
The constructor copies the bytes, retains no float32 duplicate, and shares
immutable packed parameters across sessions. Float32 remains the default.
The compressor indexer's query projection and grouped `wo_a` are unchanged.

This is weight-only FP8 with float32 activations, not the released
activation-quantized GEMM, BF16/CUDA parity, full-model loading or pretrained
generation. All real-weight checks here use layer 0 and float32 caches.

## Independent Projection Reference

The existing [attention sample](ATTENTION_VALIDATION.md) supplies the pinned
`[32768,1280]` query expansion matrix: 41,943,040 real weights. The new offline
`quant/make_query_projection_reference.py` validates snapshot/manifest provenance,
source offsets, payload lengths and hashes before decoding with PyTorch 2.14.0's
native FP8 casts. Float64 matrix multiplication supplies the independent oracle.
The decoded float32 hash must match the existing attention oracle.

Three exact binary-fraction input rows exercise dense signed and sparse inputs.
They repeat for batch sizes 1, 3, 32 and 128 to exercise the native batch kernels;
repetition is not additional input diversity. The existing projection budget is
unchanged: `abs(error) <= 1e-5 + 2e-6 * sum(abs(input * weight))`.

| Backend | Maximum absolute error | Maximum error / max(1, L1) |
| --- | ---: | ---: |
| CPU | 2.3842e-7 | 1.1684e-8 |
| GPU | 3.5763e-7 | 1.7667e-8 |

Generated oracle SHA256:
`b78142c7f1058662d91744f900d3a51eec9a6acf8eaa95b2273f7da77b890866`.
Weights and generated real-weight references remain local, ignored files.

## Attention And Model Integration

Query-only and combined query/KV modes run the same real attention checks as
the [KV-only adapter](FP8_ATTENTION_VALIDATION.md): 131-token full prefill,
prefills of 1/127/128/129 followed by two decode steps, window rollover,
chronological cache checks, causality, cached/full agreement and interleaved
sessions. Output/cache scaled-error budgets remain `2e-4`/`5e-5`, respectively.

| Mode | Backend | Worst scaled output error | Worst scaled cache error |
| --- | --- | ---: | ---: |
| Query only | CPU | 2.112e-6 | 1.809e-6 |
| Query only | GPU | 2.700e-6 | 1.925e-6 |
| Query + KV | CPU | 2.795e-6 | 1.822e-6 |
| Query + KV | GPU | 2.718e-6 | 1.839e-6 |

Download-free model tests cover KV-only, query-only and combined selection,
mixed packed/float32 layers, ordinary and quantized caches, caller-buffer copies,
independent ownership, close behavior, malformed/duplicate inputs and eight
concurrent sessions. Synthetic logit differences from decoded float32 models
were below `3.58e-7`. CI's existing `TestFP8Model` race checks cover all modes.
Real combined packed-weight/quantized-cache and compressed-layer validation
are still separate work; synthetic coverage does not establish those claims.

## Whole-Attention Measurements

Apple M3 Pro, MLX 0.32.0, mlx-c 0.6.0_3, Go 1.27.0. Medians of three 300ms
runs, three warmups per case, with no concurrent inference benchmark. Each
step includes input RMSNorm, all projections, RoPE, sparse attention, cache
updates, evaluation and session cleanup inside one `mlx.Batch`. Uploads and
Go readback are excluded. Decode clones the same evaluated 128-token template
cache and consumes one token, including rollover. Prefill uses an empty cache.
The CPU packed prefill cases exceed 300ms and therefore have one timed
iteration per run; these measurements are directional, not a broad benchmark.

| Backend | Workload | Float32 | Packed query | Packed query + KV |
| --- | --- | ---: | ---: | ---: |
| CPU | 128-token prefill | 76.455 ms | 908.259 ms | 962.690 ms |
| CPU | Decode after 128 tokens | 6.207 ms | 10.861 ms | 11.010 ms |
| GPU | 128-token prefill | 12.412 ms | 12.754 ms | 12.774 ms |
| GPU | Decode after 128 tokens | 4.532 ms | 3.616 ms | 3.451 ms |

Combined GPU decode took about **24% less time** (1.31x throughput), while GPU
prefill took about 3% longer. Decode ranges did not overlap: float32
4.481-4.657 ms, query-only 3.558-3.667 ms, combined 3.450-3.624 ms.
CPU prefill was about 12.6x slower with both packed projections; CPU decode
about 1.8x slower. These results support explicit GPU memory/latency tradeoffs,
not an automatic default or an end-to-end model speed claim.

### Memory

All-attention weight payload falls from 506,490,112 bytes to 381,971,712 for
query-only, or **374,189,312 bytes** with query + KV: **132.30 MB (26.12%)** saved.
The query matrix alone falls from 167,772,160 to 43,253,760 bytes, including
block scales expanded per row, saving 124,518,400 bytes. No float32 duplicate
is retained for either replaced matrix.

Median peak active allocator bytes, in decimal MB:

| Backend | Workload | Float32 | Packed query | Packed query + KV |
| --- | --- | ---: | ---: | ---: |
| CPU | 128-token prefill | 700.968 | 574.353 | 566.571 |
| CPU | Decode after 128 tokens | 511.153 | 386.626 | 378.852 |
| GPU | 128-token prefill | 691.599 | 567.081 | 559.298 |
| GPU | Decode after 128 tokens | 511.652 | 387.134 | 379.351 |

Median steady active memory after prefill warmup/cleanup was 509.124 MB,
384.605 MB and 376.823 MB on both devices. One combined GPU steady reading
was 379.707 MB while temporaries were still retained. Peak counters are reset
after warmup and pre-existing active allocations are subtracted. These figures
include weights, inputs, caches and temporaries, but exclude idle allocator
cache, Go buffers, loading peaks and process RSS. Counters are process-wide.

## Reproduce

Use the existing local sample and attention reference; no test downloads weights.
The generator requires the pinned PyTorch 2.14.0 environment.

```sh
python deepseek/quant/make_query_projection_reference.py \
  --samples models/deepseek-v41-attention0
export MLXGO_DEEPSEEK_ATTENTION_DIR="$PWD/models/deepseek-v41-attention0"
export MLXGO_DEEPSEEK_QUERY_REFERENCE="$MLXGO_DEEPSEEK_ATTENTION_DIR/query-projection-reference.json.gz"
go test -tags 'mlx mlxruntime' ./deepseek \
  -run '^TestReleasedFP8Query(Projection|AttentionForward)$' -v
go test -race -tags 'mlx mlxruntime' ./deepseek -run '^TestFP8Model'
go test -tags 'mlx mlxruntime' ./deepseek -run '^$' \
  -bench '^BenchmarkReleasedAttentionFP8$/(cpu|gpu)/(prefill128|decode128)/(float32|qb|kv_qb)$' \
  -benchtime=300ms -count=3
```

The generator refuses to overwrite an existing output. Supply `--out` with a
fresh local path to check deterministic reproduction. The real projection test
skips unless `MLXGO_DEEPSEEK_QUERY_REFERENCE` is set; routine CI runs the
download-free model coverage. The subsequent [grouped output report](FP8_OUTPUT_VALIDATION.md)
validates `wo_a` and its conversion semantics separately. The measurements here
remain the query-only milestone, before that extension.
