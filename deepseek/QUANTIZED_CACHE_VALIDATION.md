# Packed Attention Cache Validation

Validated on 2026-09-25 on Apple M3 Pro, Go 1.27.0, MLX 0.32.0 / mlx-c
0.6.0_3, using a separate Python 3.12 / PyTorch 2.14.0 reference environment.

## Scope

`Model.NewSessionWithOptions(SessionOptions{QuantizedCaches: true})` opts into
device-side cache quantization. Default sessions and `Model.Forward` are unchanged.
Parameters, projections, attention accumulation and pending compression inputs
remain float32. There is no implicit BF16 input/output rounding, quantized GEMM,
released-checkpoint loader, or claim of CUDA/TileLang numerical parity.

| Stored tensor | Values | Scale | Group | Payload vs float32 |
| --- | --- | --- | ---: | ---: |
| Window KV | FP8 E4M3 | E8M0 | 32 | 3.88x smaller |
| Compressed KV | Packed FP4 E2M1 | E4M3 | 16 | 7.11x smaller |
| Index keys | Packed FP4 E2M1 | E8M0 | 32 | 7.53x smaller |

Ratios include scale bytes, but exclude metadata, allocator overhead, pending
inputs, and temporary decoded arrays. They are not peak-memory or speed claims.
Index queries are also FP4/E8M0-quantized before scoring, but are not retained.

Entries are quantized after RoPE at the corresponding upstream call sites. Keys
are derived from the unrotated compressed latent before compressed-value RoPE.
Appends and window trims operate on encoded rows and their matching scales;
existing entries are not requantized. Decoding is lazy and device-local. Shared
consumers use their owner's compressed cache and selection, not private copies.
Sessions evaluate both encoded values and scales and close all retained handles.

Quantizer shape/type errors return immediately. Nonfinite values retain the
graph API's scale-255/NaN policy; lazy evaluation is not a finite-value validator.
This remains an inference-only API, without a straight-through gradient estimator.

## Independent Reference

Source revision: `df42c109f1defefcbfcedbe7d905718a12266e40`.
Both `model.py` and `kernel.py` are checksum-verified. Architectural forward
methods come from the pinned source; its quantizer calls now execute the same
independent PyTorch CPU formulas used by the
[activation reference](quant/ACTIVATION_VALIDATION.md). FP8 uses actual PyTorch
casts; FP4 uses nearest-even level selection because PyTorch lacks the CPU cast.
No Go quantizer is invoked by the oracle.

The synthetic full-model fixture retains the previously documented upstream
index-owner publication correction for incomplete compression groups. Float32
linear, sparse-attention and Sinkhorn reference replacements are unchanged.
The old unquantized fixtures are not overwritten; extracting the reusable CPU
quantizer reproduced the existing 19-case activation fixture byte-for-byte.

### Synthetic Full Model

The new checked-in fixture contains 8 layers, 26,952 parameters, 32-wide
attention/index heads, and 13 input tokens. All 14 schedules (seven prefill
lengths, with/without YaRN) match independent logits within the existing **2e-5
absolute** tolerance on CPU and GPU. The oracle's largest cached/full-prefill
error is 2.69e-7. Tests cover multiple KV/index sources, candidate selection,
ratio-1/ratio-2 compression, partial groups, window rollover, and eight concurrent
sessions across two models. Lifecycle tests cover validation, bounds, closure
and exact preservation of previously packed cache bytes. Default-cache storage
is checked separately so the opt-in path cannot silently change defaults.

Ordinary runtime CI runs the synthetic checks, native session race tests, and
the new `deepseek-smoke -quantized-caches` example without downloads or Python.

### Released Weight Samples

Separate local oracles cover layer 0, layer 2, and the layer-2/3 attention pair.
These use the existing released-weight samples with deterministic inputs, not
a complete pretrained model run. All prior prefill/window-boundary schedules,
interleaved sessions and concurrent sharing checks are retained. Quantized
top-512 selection from 640 real-derived keys matches exactly on both devices.

**Float32 accumulation differences cross quantization thresholds.** Tight
unquantized tolerances are not valid end-to-end parity criteria here. For
example, the PyTorch oracle itself differs by 0.00829 between one layer-2 cached
schedule and full prefill, and by 0.02501 after its layer-3 consumer. Tests compare
each schedule with its own reference, not with a supposedly identical full run.

For these experimental real-weight checks, scaled error is
`abs(got-reference)/(1+abs(reference))`. The explicit acceptance budget is maximum
scaled error <= 0.05 and RMS scaled error <= 0.002. Owner cache comparisons also
limit different decoded entries to 0.1%; pending float32 inputs keep their prior
5e-5 scaled tolerance. The downstream consumer receives already-different owner
outputs, so its window uses the numerical budget, not the sparse-bin limit.
That distinction does not relax dtype, shape, ownership or byte-retention checks.
These are compatibility limits, **not proof of exact quantized inference parity**.

Observed rounded worst errors across full/prefill/decode schedules:

| Component | CPU max scaled | GPU max scaled | GPU RMS scaled |
| --- | ---: | ---: | ---: |
| Layer 0 | 0.00367 | 0.00368 | 0.000108 |
| Layer 2 | 0.000011 | 0.00721 | 0.000287 |
| Layer 3 after layer 2 | 0.000361 | 0.02340 | 0.001110 |

Layer-2 compressed values and index keys match exactly in these schedules.
Some FP8 window bins differ. Propagation changes up to 2,060/65,536 decoded
consumer-window entries (3.15%) on GPU, despite the small RMS output error.
This is a material limitation to revisit before claiming released-model parity
or enabling the option by default. Identical-input quantizer tests remain
byte-exact; their tolerances were not widened.

## Reproduce

From the repository root:

```sh
go run -tags mlx ./cmd/deepseek-smoke -device gpu -quantized-caches
go test -tags "mlx mlxruntime" ./deepseek -run TestQuantized -count=1
go test -race -tags "mlx mlxruntime" ./deepseek -run TestQuantizedSession -count=1
```

To reproduce the synthetic fixture without replacing the checked-in file:

```sh
python deepseek/make_model_fixture.py --source-dir /path/to/pinned/inference \
  --quantized-caches --out /tmp/model-quantized-reproduced.json.gz
```

Use the separate PyTorch environment and pinned sources described above. For
each downloaded attention sample directory, run the existing generator with
`--quantized-caches --kernel-source /path/to/pinned/inference/kernel.py` in
addition to the usual sample/layer arguments. Layer 3 also uses
`--consumer-samples models/deepseek-v41-attention3` with layer 2's samples.
The generator creates `quantized-cache-reference.json.gz` and refuses overwrites.
It leaves the float32 oracle intact.

```sh
export MLXGO_DEEPSEEK_QUANTIZED_CACHES=1
export MLXGO_DEEPSEEK_ATTENTION_DIR="$PWD/models/deepseek-v41-attention0"
export MLXGO_DEEPSEEK_COMPRESSED_ATTENTION_DIR="$PWD/models/deepseek-v41-attention2"
export MLXGO_DEEPSEEK_SHARED_ATTENTION_DIR="$PWD/models/deepseek-v41-attention3"
go test -tags "mlx mlxruntime" ./deepseek -run TestReleasedQuantized -v -count=1
```

Once enabled, missing/corrupt oracles fail rather than skip. References include
the quantization mode, source hashes and weight manifest provenance; a quantized
oracle is rejected by an unquantized test, and vice versa.

SHA-256 values of the generated gzip artifacts:

- Synthetic model: `2caa8ab2920429f2c5c058633a5f1b91d2deea5243870773d9aee5b7c32f0275`
- Layer 0: `ace389a91d265cdb1724b2c1930cd8061746aa39ab8c055d00c25b3f845936d1`
- Layer 2: `5b16accb5b72ad0c5702c530aba124419c34a8fd49590cd5be9b35f76e7d433f`
- Layer 3: `e456a7038a9afe749a5e04aaffacc829b2174f8de8499aa20760845f965ca158`

Only the small synthetic model belongs in source control. Released weights and
their generated oracles remain under ignored `models/`.
