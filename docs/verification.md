# Verification

Run commands from the repository root.

### Reference Verification

CI runs 119 Hugging Face tokenizer cases, small-transformer cache comparisons,
LoRA finite-difference gradients, learning/frozen-weight/checkpoint tests, and
AdamW checks against a scalar reference. It never downloads model weights.

The real-checkpoint tests are opt-in:

```sh
MLXGO_QWEN2_DIR="$PWD/models/Qwen2.5-0.5B-Instruct" \
  go test -tags "mlx mlxruntime" ./qwen2 -run TestQwenGolden -v
MLXGO_QWEN2_DIR="$PWD/models/Qwen2.5-0.5B-Instruct" MLXGO_QWEN2_MEMORY=1 \
  go test -tags "mlx mlxruntime" ./qwen2 -run TestQwenMemory -v
```

Golden verification requires all 32 reference tokens to match and maximum
absolute prefill-logit error of at most 0.25. These limits are fixed in the
test, not chosen from the Go output. The memory test warms up for 200 tokens,
then checks retained RSS after two more 200-token runs against a 64 MiB growth
limit. This detects regression in retained memory, not all possible leaks.

Fixtures were generated with the pinned versions in
`qwen2/requirements-reference.txt`, using the upstream
[mlx-lm Qwen2 model](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen2.py)
and its built-in `ConcatenateKVCache`. The model uses fused bias projections
and compiled SwiGLU to match reference bf16 rounding. Different MLX versions,
kernel choices, or cache layouts can change greedy decisions near tied logits.
The C++ attention shim supports both mlx-c 0.6.0 and the newer `force_fused`
signature without a version-dependent function-pointer cast.

To deliberately regenerate fixtures on a Metal-capable Mac:

```sh
python3.12 -m venv .venv-qwen
.venv-qwen/bin/pip install -r qwen2/requirements-reference.txt
.venv-qwen/bin/python qwen2/make_fixture.py
```

The checked-in tokenizer vocabulary is a reduced fixture for the reference
corpus; applications must load the full tokenizer from the model directory.

