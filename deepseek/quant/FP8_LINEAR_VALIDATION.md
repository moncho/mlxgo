# Packed FP8 Projection Validation

This validates **one real weight-only projection**, not released-model inference
or DeepSeek's activation-quantized GEMM. This report measures the standalone
`FP8Linear` adapter. Its opt-in session integration is covered separately by
[real attention validation](../FP8_ATTENTION_VALIDATION.md). Default sessions,
checkpoint loaders and float32 reference budgets are unchanged.

## Layout And Execution

The target is `layers.0.attn.wkv.weight`, shape `[512,5120]`, from the existing
[pinned sample](REAL_WEIGHT_VALIDATION.md). Its E4M3FN weight bytes and E8M0
32x32 block scales are uploaded without requantization. Each block scale is
repeated for 32 rows, adapting to MLX's row-group-32 layout. Four unchanged
weight bytes are viewed as each UInt32 word, low byte first on Apple Silicon.

`mlx.MXFP8Matmul` calls `mlx_quantized_matmul` with mode `mxfp8`, group size
32, bits 8 and transpose enabled. Inputs remain floating point. The native
[MLX 0.32.0 operation](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/ops.cpp)
and [CPU packed kernel](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/cpu/quantized.cpp)
are used directly. No decoded full-weight array is built by the adapter.
Construction validates finite weights using a 32-float scratch buffer before
uploading. Native underflow, exceptional values in activations and accumulation
follow MLX; this is not a bit-exact replacement for `Decode` for every exponent.

This path does not apply `wo_a`'s BF16 conversion, handle grouped weights, pack
FP4 experts, or automatically replace model projections. Inference is the
tested use case; no training support is claimed for this adapter.

A separate [grouped output adapter](../FP8_OUTPUT_VALIDATION.md) now composes
one `FP8Linear` per `wo_a` group and rejects weights changed by BF16 conversion.
That validation does not change the standalone contract or measurements here.

## Correctness

The real-weight test verifies manifest, weight and scale hashes against the
existing independent PyTorch 2.14.0 reference. Its three canonical input rows
are repeated to test batches of 1, 3, 32 and 128 tokens. Reference outputs use
float64 dot products of the decoded float32 weights, not another MLX kernel.
Repetition exercises kernel batch-size changes, not new activation diversity.

The unchanged projection budget is:

```text
abs(actual - reference) <= 1e-5 + 2e-6 * sum(abs(x_i * w_i))
```

Worst errors over those batches on the tested machine:

| Backend | Maximum absolute error | Maximum L1-normalized error |
| --- | ---: | ---: |
| CPU | 2.683e-7 | 5.477e-9 |
| GPU | 3.428e-7 | 6.992e-9 |

Download-free runtime tests additionally cover both scale layouts, every finite
E4M3 byte code in small matrices, varying scales, batches 1/3/16/33/128, copied
input buffers, packed dtypes/shapes, closing a layer before output evaluation,
shared ownership, compiled graphs, strided inputs and eight concurrent callers.
The low-level binding is tested with Float32/Float16/BFloat16 inputs and invalid
shapes/dtypes/closed handles. Constructor validation covers malformed lengths,
dimensions, formats, NaN codes/scales and float32 overflow. The existing CI
runtime job runs the synthetic tests and its quant-package race job covers
concurrency; real-weight tests skip without the local sample environment.

## Storage And Timing

Apple M3 Pro, MLX 0.32.0, mlx-c 0.6.0_3, Go 1.27.0. Times below are medians
of three 300ms benchmark runs, with five warmups per case. Each iteration
builds, evaluates and closes a fresh output inside one `Batch`; weights and
inputs are reused. Upload, validation and full-matrix float32 decoding are
excluded for both paths. The float32 baseline is ordinary `Matmul(x, w.T)`.
Every iteration synchronizes evaluation; these are not isolated kernel timings
or whole-model tokens/second. Benchmark inputs are deterministic synthetic
activations applied to the real weights.

| Backend | Tokens | Float32 | Packed FP8 | Float32 time / FP8 time |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1 | 59.97 us | 417.08 us | 0.14x |
| CPU | 32 | 276.34 us | 12,996.67 us | 0.021x |
| CPU | 128 | 655.79 us | 52,061.08 us | 0.013x |
| GPU | 1 | 234.90 us | 162.80 us | 1.44x |
| GPU | 32 | 305.91 us | 249.36 us | 1.23x |
| GPU | 128 | 401.39 us | 391.00 us | 1.03x |

The GPU gain is workload-dependent; the 128-token difference is small.
**CPU packed execution is substantially slower.** Keep the choice explicit.

Retained weight payload is 2,621,440 weight bytes + 81,920 repeated scale bytes
= **2,703,360 bytes**, versus **10,485,760 bytes** for float32: **3.88x smaller**.
The original checkpoint has only 2,560 block-scale bytes. The adapter trades
that modest scale expansion for compatibility with MLX's existing kernel.
These are exact array-payload sizes, not measurements of process RSS, allocator
peak usage, temporary kernel memory, or whole-model memory savings.

## Reproduce

Use the existing sample and `reference.json`, generated as described in
[real-weight validation](REAL_WEIGHT_VALIDATION.md). Nothing is downloaded by
these commands:

```sh
export MLXGO_DEEPSEEK_SAMPLE_DIR="$PWD/models/deepseek-v41-sample"
go test -tags 'mlx mlxruntime' ./deepseek/quant -run 'TestReleasedFP8Linear' -v
go test -tags 'mlx mlxruntime' ./ ./deepseek/quant -run 'Test(RuntimeMXFP8|FP8Linear)'
go test -race -tags 'mlx mlxruntime' ./deepseek/quant -run 'TestFP8Linear'
go test -tags 'mlx mlxruntime' ./deepseek/quant -run '^$' \
  -bench '^BenchmarkReleasedFP8Linear$' -benchtime=300ms -count=3
```

The [attention integration report](../FP8_ATTENTION_VALIDATION.md) measures explicit
selection inside a real attention block. Activation-quantized GEMM requires its
own independent reference; these measurements do not establish it.
