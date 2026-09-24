# Released Expert Validation

Layer 0, expert 0 from DeepSeek-V4.1-Flash now runs through `deepseek.Expert`
using all three real weight matrices, decoded into float32. This is one
pretrained component, not a complete pretrained model or a quantized-kernel test.

## Reproduce

From the repository root, after the initial projection samples have been fetched:

```sh
go run ./cmd/fetch-deepseek-sample -set expert \
  -reuse models/deepseek-v41-sample -out models/deepseek-v41-expert0

# Use the Python 3.12 / torch==2.14.0 environment from quant/README.md.
/tmp/mlxgo-sample-reference-venv/bin/python deepseek/make_expert_reference.py \
  --samples models/deepseek-v41-expert0

export MLXGO_DEEPSEEK_EXPERT_DIR="$PWD/models/deepseek-v41-expert0"
go test ./deepseek -run TestReleasedExpertDecoding -v -count=1
go test -tags "mlx mlxruntime" ./deepseek -run 'Test(ReleasedExpert|ExpertClipping)' -v -count=1
```

Omit `-reuse` to fetch all 18.8 MB. With reuse, only 12.5 MB of new payload is
downloaded. Both modes validate the pinned shard header and exact HTTP ranges;
the new output directory must not already exist. Existing source samples are
never modified. Corrupt reused files fail instead of being silently trusted.

The generator fetches only the pinned official `model.py` source, not weights.
Pass `--model-source /path/to/model.py` for an offline run. Its SHA-256 is checked
before extracting the original `Expert.forward` method. The FP4 table and model
configuration come from the checksum-verified audit snapshot. Outputs refuse
overwrite; `--out /path/to/new.json` supports separate reference regeneration.

Only the reference's quantized linear operations are replaced with ordinary
PyTorch float32 linear operations. The original clipping, SiLU, multiplication,
routing placement and final projection remain intact. This deliberately does
not reproduce dynamic activation quantization or BF16 activation rounding.

## Library APIs

- `quant.ReadMatrix` streams raw weights/scales with a caller-set numeric-buffer
  budget. It verifies lengths and encodings; callers supply validated shapes and
  integrity hashes. The expert test uses 64 MiB per matrix and checks all hashes.
- `deepseek.Expert(x, weights, routing, limit)` evaluates one float32 expert with
  `x [tokens,dim]`, `routing [tokens,1]` and weights in `[output,input]` order.
  Use routing values of one for an unweighted expert and zero limit to disable
  clipping. It borrows inputs and returns a caller-owned output array.

Each matrix has 11,796,480 decoded values (45 MiB float32), for 135 MiB across
the three retained native matrices. The reader budget is **not** a process-wide
memory cap: Go buffers, MLX copies, readback and temporary arrays require more.
This approach is bounded component validation, not an efficient full-model loader.

## Recorded Results

Validated on 2026-09-23, Apple M3 Pro, MLX CPU/Metal and PyTorch 2.14.0 CPU.
Release revision: `df42c109f1defefcbfcedbe7d905718a12266e40`.

All **35,389,440** decoded weights matched reference hashes after signed-zero
normalization. Upload/readback preserved them on both devices. Six deterministic
inputs include two dense ordinary inputs, two 64x stress inputs, a zero input and
a sparse input. Stress inputs exercise 1,961 positive gate clamps, 2,012 negative
gate values that must remain unclamped, 2,019 positive up clamps and 1,956 negative
up clamps at the released limit of 10.

| Case | Outputs per device | CPU max absolute error | GPU max absolute error |
| --- | ---: | ---: | ---: |
| Clipped, unweighted | 30,720 | 3.0517578125e-5 | 3.0517578125e-5 |
| Clipped, routed | 30,720 | 2.288818359375e-5 | 2.288818359375e-5 |
| Unclipped, routed | 30,720 | 9.765625e-4 | 9.765625e-4 |

All **184,320** output checks passed `abs_error <= 3e-5 + 2e-6 * L1`, where L1 is
the float64 sum of absolute contributions to the final projection. Maximum error
divided by `max(1,L1)` was 8.5319556e-8. The unclipped stress outputs reached
12,522, explaining their larger absolute rounding differences. Zero-input and
zero-routing rows were also required to be exactly zero.

Local manifest SHA-256:
`dc12f3f5de614a16e1fd8f2d2b356eb0c90802c89b0cb080a0076209ceea2940`.
Reference JSON SHA-256:
`18465187cb71982b819a09f515bf0853c2b97e62f4c07dc06f4aa1626c55154f`.
The reference was regenerated and matched byte-for-byte. Hashes describe this
local validation, not upstream-published tensor checksums.

## Regression Coverage and Limits

Ordinary CI tests the bounded reader, reuse corruption handling, and synthetic
expert clipping/shape checks on CPU and GPU. Real-weight tests are opt-in via
`MLXGO_DEEPSEEK_EXPERT_DIR`; missing files are failures once enabled. Model weights
and generated references remain in ignored `models/` directories.

The stub and full native runtime suites passed with both released sample sets
enabled. This validates the complete mathematical expert in float32, but not
MoE expert selection, the full attention path, a transformer layer, dynamic
activation quantization, quantized GPU kernels, logits or text generation.
