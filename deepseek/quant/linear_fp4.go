package quant

import (
	"fmt"
	mlx "github.com/moncho/mlxgo"
)

// FP4Linear owns an experimental weight-only MXFP4 projection. Copies share
// handles; Close invalidates every copy. Concurrent calls are serialized on
// the MLX worker. This is inference-only, not an activation-quantized GEMM.
type FP4Linear struct{ weights, scales mlx.Array }

// NewFP4Linear copies row-scaled checkpoint bytes without requantization or
// allocating a full decoded matrix. rows and cols must be positive multiples
// of 32. data holds rows*cols/2 E2M1 bytes (earlier element in the low nibble);
// scales holds rows*cols/32 E8M0 codes. Nonfinite decoded values are rejected.
// Native underflow and accumulation follow MLX, not bit-exact CPU decoding.
func NewFP4Linear(data, scales []byte, rows, cols int) (*FP4Linear, error) {
	if rows <= 0 || cols <= 0 || rows%32 != 0 || cols%32 != 0 || rows > int(^uint(0)>>1)/cols || rows > 1<<31-1 || cols > 1<<31-1 {
		return nil, fmt.Errorf("quant: invalid FP4 linear dimensions")
	}
	if len(data) != rows*cols/2 || len(scales) != rows*cols/32 {
		return nil, fmt.Errorf("quant: incorrect FP4 linear data or scale length")
	}
	var chunk [32]float32
	for i := 0; i < len(scales); i++ {
		if err := Decode(chunk[:], data[i*16:(i+1)*16], scales[i:i+1], 1, 32, FP4Row32, Float32); err != nil {
			return nil, fmt.Errorf("quant: FP4 group %d: %w", i, err)
		}
	}
	var layer *FP4Linear
	err := mlx.Batch(func() error {
		b, err := mlx.NewUInt8(data, []int{rows, cols / 2})
		if err != nil {
			return err
		}
		defer b.Close()
		w, err := mlx.View(b, mlx.UInt32)
		if err != nil {
			return err
		}
		s, err := mlx.NewUInt8(scales, []int{rows, cols / 32})
		if err != nil {
			w.Close()
			return err
		}
		layer = &FP4Linear{weights: w, scales: s}
		return nil
	})
	return layer, err
}

// Forward returns a caller-owned lazy Float32 [tokens,rows] array from Float32
// inputs [tokens,cols]. Activations are not quantized. Weights remain packed.
func (l *FP4Linear) Forward(x mlx.Array) (mlx.Array, error) {
	var out mlx.Array
	err := mlx.Batch(func() error {
		if l == nil {
			return fmt.Errorf("quant: nil FP4 linear")
		}
		dt, err := x.DType()
		if err != nil {
			return err
		}
		if dt != mlx.Float32 {
			return fmt.Errorf("quant: FP4 linear input must be Float32")
		}
		out, err = mlx.MXFP4Matmul(x, l.weights, l.scales)
		return err
	})
	return out, err
}

// Close releases packed weights and scales. It is idempotent. Previously
// built lazy outputs retain their dependencies and can still be evaluated.
func (l *FP4Linear) Close() error {
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
