package quant

import (
	"fmt"

	mlx "github.com/moncho/mlxgo"
)

// FP8Linear owns one experimental packed FP8 projection. It does not quantize
// activations or implement DeepSeek's FP8 activation GEMM. Copies share native
// array ownership; Close invalidates every copy. Forward may be called from
// multiple goroutines. Calls are serialized by the MLX worker.
type FP8Linear struct {
	weights mlx.Array
	scales  mlx.Array
}

// NewFP8Linear copies a checkpoint matrix into packed MLX storage. It accepts
// FP8Block32 or FP8Row32, with positive dimensions divisible by 32. Block scales
// are repeated per row without changing the weight bytes or requantizing them.
// No full float32 weight matrix is allocated. Validation uses a bounded chunk.
// NaN codes/scales or float32 weight overflow are rejected before native upload.
// This does not apply the optional BF16 rounding used by some checkpoint layers.
func NewFP8Linear(data, scales []byte, rows, cols int, format Format) (*FP8Linear, error) {
	if rows <= 0 || cols <= 0 || rows%32 != 0 || cols%32 != 0 || rows > int(^uint(0)>>1)/cols || rows > 1<<31-1 || cols > 1<<31-1 {
		return nil, fmt.Errorf("quant: FP8 linear dimensions must be positive multiples of 32 without overflow")
	}
	if format != FP8Block32 && format != FP8Row32 {
		return nil, fmt.Errorf("quant: FP8 linear requires FP8Block32 or FP8Row32")
	}
	groups := cols / 32
	scaleRows := rows
	if format == FP8Block32 {
		scaleRows /= 32
	}
	if len(data) != rows*cols || len(scales) != scaleRows*groups {
		return nil, fmt.Errorf("quant: incorrect FP8 linear data or scale length")
	}
	var decoded [32]float32
	for row := 0; row < rows; row++ {
		sr := row
		if format == FP8Block32 {
			sr /= 32
		}
		for g := 0; g < groups; g++ {
			i, j := row*cols+g*32, sr*groups+g
			if err := Decode(decoded[:], data[i:i+32], scales[j:j+1], 1, 32, FP8Row32, Float32); err != nil {
				return nil, fmt.Errorf("quant: FP8 weight row %d group %d: %w", row, g, err)
			}
		}
	}
	rowScales := scales
	if format == FP8Block32 {
		rowScales = make([]byte, rows*groups)
		for row := 0; row < rows; row++ {
			copy(rowScales[row*groups:(row+1)*groups], scales[(row/32)*groups:(row/32+1)*groups])
		}
	}
	var layer *FP8Linear
	err := mlx.Batch(func() error {
		bytes, err := mlx.NewUInt8(data, []int{rows, cols})
		if err != nil {
			return err
		}
		defer bytes.Close()
		weights, err := mlx.View(bytes, mlx.UInt32)
		if err != nil {
			return err
		}
		s, err := mlx.NewUInt8(rowScales, []int{rows, groups})
		if err != nil {
			weights.Close()
			return err
		}
		layer = &FP8Linear{weights: weights, scales: s}
		return nil
	})
	return layer, err
}

// Forward projects Float32 inputs [tokens, input width] to [tokens, output
// width]. The caller owns the lazy output. Weights remain packed throughout.
func (l *FP8Linear) Forward(x mlx.Array) (mlx.Array, error) {
	var out mlx.Array
	err := mlx.Batch(func() error {
		if l == nil {
			return fmt.Errorf("quant: nil FP8 linear")
		}
		dtype, err := x.DType()
		if err != nil {
			return err
		}
		if dtype != mlx.Float32 {
			return fmt.Errorf("quant: FP8 linear input must be Float32")
		}
		out, err = mlx.MXFP8Matmul(x, l.weights, l.scales)
		return err
	})
	return out, err
}

// Close releases the layer's packed weights and scales. It is idempotent.
// Already-built outputs retain their graph dependencies and can still be used.
func (l *FP8Linear) Close() error {
	if l == nil {
		return nil
	}
	return mlx.Batch(func() error {
		err := l.weights.Close()
		other := l.scales.Close()
		if err != nil {
			return err
		}
		return other
	})
}
