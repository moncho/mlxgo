# Released-Weight Validation

Validated on 2026-09-23 using Apple M3 Pro, MLX CPU and Metal GPU, and PyTorch
2.14.0 CPU reference arithmetic. This is a component-level result, not evidence
that the complete released DeepSeek model is supported.

## Inputs and Provenance

- Repository: `deepseek-ai/DeepSeek-V4.1-Flash`.
- Revision: `df42c109f1defefcbfcedbe7d905718a12266e40`.
- Shard: `model-00003-of-00048.safetensors`.
- Downloaded tensor payload: 42,478,080 bytes, plus the 258,144-byte header.
- Local sample manifest SHA-256:
  `d2673e671f9f9a00fccf91ce38d9a3def32b761cd4591d9a00101671ffc5197d`.
- Generated `reference.json` SHA-256:
  `6d6ca231b962eadf1aada68638c09665441d0cdc3ed49ecb2739138d77092341`.

The downloader checks the audit-pinned header hash, exact byte ranges and stable
shard ETag/size. The reference generator validates sample filenames, storage
shapes, offsets, lengths and hashes against that header and local manifest.
Payload hashes are locally measured, not upstream-published tensor checksums.

## Decoding

| Tensor (all layer 0) | Logical shape | Values checked | Result |
| --- | --- | ---: | --- |
| `attn.wkv` | `[512,5120]` | 2,621,440 | FP8 block decoding matches |
| `ffn.experts.0.w1` | `[2304,5120]` | 11,796,480 | FP4 row decoding matches |
| `attn.wo_a` | `[8192,4096]` | 33,554,432 | FP8 decoding and BF16 rounding match |

All 47,972,352 weights were checked, not a random subset. Host decoding used
32-row chunks; MLX-worker decoding used whole matrices. Comparison hashes are
bit-exact after canonicalizing signed zero to match the official FP4 table.
The `wo_a` float32 pre-rounding values were checked separately as well.

Executing the checksum-pinned official FP4-to-FP8 converter on the complete
sampled expert produced **zero numerical mismatches**, with maximum absolute
error zero. This verifies losslessness for this sample only; it does not establish
losslessness for every possible scale distribution or every released expert.
The unmodified official `wo_a` conversion block also matched the independently
expanded 32x32 scaling plus BF16 rounding.

## Native Computation

Three deterministic inputs per matrix/group (two dense, one sparse) exercise
all columns and output rows. `wo_a` runs as eight independent groups, not as an
ordinary flattened projection. The comparison target uses float64 accumulation.

| Tensor | Outputs per device | CPU max absolute error | GPU max absolute error |
| --- | ---: | ---: | ---: |
| `attn.wkv` | 1,536 | 5.2154064e-7 | 4.4703484e-7 |
| `ffn.experts.0.w1` | 6,912 | 0 | 0 |
| `attn.wo_a` | 24,576 | 7.0780516e-7 | 7.0780516e-7 |

All 66,048 CPU/GPU outputs passed the fixed tolerance documented in the README.
Maximum error divided by `max(1, sum(abs(x_i*w_i)))` was 2.1368068e-8. All decoded
weights also passed native upload/readback checks, including the BF16 round trip
for `wo_a` before float32 computation.

## Scope and Reproduction

Follow [the README instructions](README.md#validate-the-downloaded-samples).
The full stub and native runtime suites and the quant package race checks passed.
The released-sample tests are opt-in; ordinary CI still uses synthetic fixtures
and does not download these files. The generated reference was independently
regenerated and compared byte-for-byte.

This does not test complete attention, routing, Engram, activation quantization,
quantized GPU kernels, full-model logits or text generation. It does not change
the model's memory requirements. No decoder correction was necessary for these
three samples, and no model payload or generated projection fixture is checked
into the repository.
