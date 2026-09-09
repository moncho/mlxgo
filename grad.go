package mlx

import (
	"fmt"
	"math"
)

// ClipGradNorm returns new float32 gradients scaled by min(1, maxNorm/norm),
// where norm is the global L2 norm across all tensors, before clipping.
// A zero maxNorm disables clipping but still checks the norm. Inputs are borrowed;
// callers must close every returned array, even when no scaling was necessary.
// The norm is accumulated in float32. Nonfinite norms (including reduction
// overflow) return an error without changing inputs. This performs evaluation.
func ClipGradNorm(grads []Array, maxNorm float32) (clipped []Array, norm float32, err error) {
	if len(grads) == 0 {
		return nil, 0, fmt.Errorf("mlxgo: gradients must not be empty")
	}
	if maxNorm < 0 || math.IsNaN(float64(maxNorm)) || math.IsInf(float64(maxNorm), 0) {
		return nil, 0, fmt.Errorf("mlxgo: max gradient norm must be finite and nonnegative")
	}
	err = Batch(func() error {
		total, err := NewScalarFloat32(0)
		if err != nil {
			return err
		}
		defer func() { _ = total.Close() }()
		for i, g := range grads {
			dt, err := g.DType()
			if err != nil {
				return fmt.Errorf("mlxgo: gradient %d: %w", i, err)
			}
			if dt != Float32 {
				return fmt.Errorf("mlxgo: gradient %d must be float32", i)
			}
			squared, err := Square(g)
			if err != nil {
				return err
			}
			sum, err := Sum(squared, false)
			_ = squared.Close()
			if err != nil {
				return err
			}
			next, err := Add(total, sum)
			_ = sum.Close()
			if err != nil {
				return err
			}
			_ = total.Close()
			total = next
		}
		data, err := total.Float32Data()
		if err != nil {
			return err
		}
		norm = float32(math.Sqrt(float64(data[0])))
		if math.IsNaN(float64(norm)) || math.IsInf(float64(norm), 0) {
			return fmt.Errorf("mlxgo: nonfinite gradient norm %g", norm)
		}
		factor := float32(1)
		if maxNorm > 0 && norm > maxNorm {
			factor = maxNorm / norm
		}
		scale, err := NewScalarFloat32(factor)
		if err != nil {
			return err
		}
		defer scale.Close()
		for _, g := range grads {
			out, err := Multiply(g, scale)
			if err != nil {
				return err
			}
			clipped = append(clipped, out)
		}
		return Eval(clipped...)
	})
	if err != nil {
		_ = CloseArrays(clipped)
		clipped = nil
	}
	return
}
