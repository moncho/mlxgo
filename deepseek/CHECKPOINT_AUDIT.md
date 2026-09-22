# Released Checkpoint Audit

This is an observed compatibility report, not a loading or conversion feature.
**The released model is still not runnable in mlxgo.** The audit reads the
index, configuration, conversion source and safetensors headers, never tensor
payloads. It does not execute downloaded code.

Audited on 2026-09-22 against `deepseek-ai/DeepSeek-V4.1-Flash`, immutable revision
`df42c109f1defefcbfcedbe7d905718a12266e40`. The [JSON report](testdata/released-checkpoint.audit.json)
records source checksums, every shard's header checksum and size, and grouped
source-to-target mappings. The compressed [metadata snapshot](testdata/released-checkpoint.metadata.json.gz)
contains all original headers and the exact index/config/conversion source for
offline verification. It contains no weight payloads. Upstream source and
metadata are covered by [DeepSeek's MIT license](THIRD_PARTY_LICENSE).

## Observed Inventory

| Check | Result |
| --- | ---: |
| Shards inspected | 48 |
| Indexed tensors, including scales | 96,085 |
| Safetensors headers read, including length prefixes | 10,685,312 bytes |
| Indexed payload size, not downloaded | 510,286,023,000 bytes |
| Text-backbone weight tensors mapped | 46,966 |
| Associated text scale tensors validated | 46,412 |
| Vision tensors explicitly excluded | 306 |
| DSpark tensors explicitly excluded | 2,401 |
| Missing, misrouted, malformed or unexplained tensors | 0 |

The header inventory exactly matches the index and its declared payload size.
Every backbone weight matches its expected logical shape after accounting for
packed storage. Scale pairing is checked across shards, not just within a file.
The independent shape schema is regression-tested against
`deepseek.Config.ParameterShapes` on both reduced fixtures, including Engram.

Headers imply 748,494,669,424 logical text parameters, including both large
Engram tables. Expanding all of them to float32 would require 2,993,978,677,696
bytes for parameters alone. This is an estimate from shapes, not an allocation
or a proposed loading strategy. Sharding does not remove that memory requirement.

## Validated Mapping Rules

The observed release already uses names such as `layers.0.attn.wq_a.weight`.
The mapper also follows the pinned conversion script's aliases: remove a leading
`model.`, rename `self_attn` to `attn`, `mlp` to `ffn` outside vision,
`weight_scale_inv` to `scale`, and `e_score_correction_bias` to `bias`.
Normalized name collisions fail. Aliases are tested, not inferred from substrings.

| Text weights | Count | Required conversion, not yet implemented |
| --- | ---: | --- |
| F32 | 320 | Preserve values and shape |
| BF16 | 234 | Cast to the experimental float32 contract |
| F8_E4M3 projections | 290 | Decode with F8_E8M0 scales per 32x32 block |
| F8_E4M3 `attn.wo_a` | 40 | Decode blocks, then reproduce the reference converter's BF16 rounding |
| F8_E4M3 Engram tables | 2 | Decode selected rows with one scale per 32 columns |
| I8 routed-expert matrices | 46,080 | Treat bytes as packed E2M1, low nibble then high nibble, with one scale per row per 32 logical columns |

The scale inventory contains 46,412 F8_E8M0 tensors. For a logical `[out,in]`
matrix, normal FP8 scales are `[out/32,in/32]`; Engram scales are
`[rows,head_dim/32]`; packed experts store `[out,in/2]` bytes and use scales
`[out,in/32]`. Both stored weight and scale shapes are checked. `I8` is only
classified as packed FP4 for recognized routed-expert weights, never globally.

Vision encoders, the aligner, image markers and `ffn.gate.bias_vl` are excluded
from text inference. `mtp.*` belongs to the unsupported DSpark path. Their
safetensors structure and routing are checked, but their architecture-specific
shapes and numerical behavior are not validated by this text audit.

Sources: pinned [config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/config.json),
[conversion script](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/inference/convert.py),
and [model implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/inference/model.py).

## Reproduce

Offline, with no MLX runtime, network or weights:

```sh
go run ./cmd/audit-deepseek \
  -snapshot deepseek/testdata/released-checkpoint.metadata.json.gz
go test ./internal/deepseekaudit ./checkpoint ./cmd/audit-deepseek
```

Refresh from the same immutable revision into **new** files:

```sh
go run ./cmd/audit-deepseek -remote \
  -save-snapshot /tmp/deepseek-metadata.json.gz \
  -out /tmp/deepseek-audit.json
```

Existing output paths, including symlinks, are never overwritten. Remote mode
uses exact HTTP byte ranges: eight bytes for the length, then exactly the JSON
header. It rejects a server that ignores Range before reading its body, checks
Content-Range/response length, and requires stable file size and a strong ETag across the
two requests. Headers/indexes are bounded at 16 MiB each and application-read
response bodies at 64 MiB in total. Config/index/conversion checksums pin the
shape and naming contract. An interrupted or failed download produces no new
snapshot. Ordinary CI uses the checked-in snapshot, not the network.

`metadata_valid: true` means the inventory and mapping checks passed.
`ready_to_load` remains false. Header hashes identify metadata, not verified
full-shard contents. Payload values, scale values, rounding and model outputs
have not been compared. Tokenizer/Engram hash preparation, cache quantization,
efficient expert execution and memory constraints remain separate blockers.

The next bounded milestone is numerical decoder validation for E4M3, E8M0 and
packed E2M1 on small reference cases, before touching released weight payloads.
