package quant

import (
	"fmt"
	"log"

	mlx "github.com/moncho/mlxgo"
)

// FP4LinearBank owns stacked, weight-only MXFP4 matrices of equal shape.
// It is inference-only. Close invalidates copies; lazy outputs retain their
// dependencies. Construction copies packed arrays, never decoded matrices.
type FP4LinearBank struct {
	weights, scales mlx.Array
	closed          bool
}

// StackFP4Linears copies packed projections into one evaluated bank. Inputs
// remain caller-owned and can be closed after construction. Temporary source
// arrays and the stacked destination coexist during construction.
func StackFP4Linears(layers []*FP4Linear) (*FP4LinearBank, error) {
	if len(layers) == 0 {
		return nil, fmt.Errorf("quant: empty FP4 linear bank")
	}
	b := &FP4LinearBank{}
	err := mlx.Batch(func() error {
		weights, scales := make([]mlx.Array, len(layers)), make([]mlx.Array, len(layers))
		var rows, cols int
		for i, l := range layers {
			if l == nil {
				return fmt.Errorf("nil projection")
			}
			if _, err := l.weights.DType(); err != nil {
				return err
			}
			shape := l.weights.Shape()
			if len(shape) != 2 {
				return fmt.Errorf("invalid projection shape")
			}
			if i == 0 {
				rows, cols = shape[0], shape[1]
			} else if shape[0] != rows || shape[1] != cols {
				return fmt.Errorf("unequal projection shapes")
			}
			weights[i], scales[i] = l.weights, l.scales
		}
		var err error
		b.weights, err = mlx.Stack(weights)
		if err != nil {
			return err
		}
		b.scales, err = mlx.Stack(scales)
		if err != nil {
			return err
		}
		return mlx.Eval(b.weights, b.scales)
	})
	if err != nil {
		b.Close()
		log.Printf("quant: prepare FP4 linear bank: %v", err)
		return nil, fmt.Errorf("quant: could not prepare FP4 linear bank")
	}
	return b, nil
}

// Gather multiplies x by the selected transposed matrices. rhsIndices selects
// bank entries; lhsIndices optionally selects flattened input batches. Indices
// stay on the device. Callers must supply in-range indices. Empty lhsIndices
// uses MLX's implicit batch selection. Output is caller-owned and lazy.
func (b *FP4LinearBank) Gather(x, lhsIndices, rhsIndices mlx.Array) (out mlx.Array, err error) {
	err = mlx.Batch(func() error {
		if b == nil || b.closed {
			return fmt.Errorf("quant: closed FP4 linear bank")
		}
		var e error
		out, e = mlx.GatherMXFP4(x, b.weights, b.scales, lhsIndices, rhsIndices, false)
		return e
	})
	return out, err
}

// Linear returns an independently owned view of one packed bank entry. It
// shares the bank's allocation without copying packed or decoded weights.
// Closing either object leaves the other usable. The caller owns the result.
func (b *FP4LinearBank) Linear(index int) (out *FP4Linear, err error) {
	err = mlx.Batch(func() error {
		if b == nil || b.closed {
			return fmt.Errorf("closed FP4 linear bank")
		}
		shape := b.weights.Shape()
		if len(shape) != 3 {
			return fmt.Errorf("invalid or closed FP4 linear bank")
		}
		if index < 0 || index >= shape[0] {
			return fmt.Errorf("FP4 bank index outside range")
		}
		view := func(a mlx.Array) (mlx.Array, error) {
			dims := a.Shape()
			if len(dims) != 3 {
				return mlx.Array{}, fmt.Errorf("invalid or closed FP4 bank storage")
			}
			part, err := mlx.Slice(a, []int{index, 0, 0}, []int{index + 1, dims[1], dims[2]}, []int{1, 1, 1})
			if err != nil {
				return mlx.Array{}, err
			}
			defer part.Close()
			return mlx.Reshape(part, dims[1:])
		}
		out = &FP4Linear{}
		var e error
		out.weights, e = view(b.weights)
		if e != nil {
			return e
		}
		out.scales, e = view(b.scales)
		return e
	})
	if err != nil {
		out.Close()
		log.Printf("quant: view FP4 bank entry: %v", err)
		return nil, fmt.Errorf("quant: could not view FP4 bank entry")
	}
	return out, nil
}

// Retain returns independent handles sharing this projection's packed storage.
// Closing either projection leaves the other usable. No weight data is copied.
func (l *FP4Linear) Retain() (out *FP4Linear, err error) {
	err = mlx.Batch(func() error {
		if l == nil {
			return fmt.Errorf("nil FP4 linear")
		}
		out = &FP4Linear{}
		for _, pair := range []struct {
			src mlx.Array
			dst *mlx.Array
		}{{l.weights, &out.weights}, {l.scales, &out.scales}} {
			shape := pair.src.Shape()
			if len(shape) != 2 {
				return fmt.Errorf("invalid or closed FP4 linear storage")
			}
			var e error
			*pair.dst, e = mlx.Reshape(pair.src, shape)
			if e != nil {
				return e
			}
		}
		return nil
	})
	if err != nil {
		out.Close()
		log.Printf("quant: retain FP4 linear: %v", err)
		return nil, fmt.Errorf("quant: could not retain FP4 linear")
	}
	return out, nil
}

// Close releases bank handles. It is idempotent and serialized on the worker.
func (b *FP4LinearBank) Close() error {
	if b == nil {
		return nil
	}
	return mlx.Batch(func() error {
		if b.closed {
			return nil
		}
		b.closed = true
		return mlx.CloseArrays([]mlx.Array{b.weights, b.scales})
	})
}
