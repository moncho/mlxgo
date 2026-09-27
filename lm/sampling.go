package lm

import (
	"fmt"
	"math"

	mlx "github.com/moncho/mlxgo"
)

// SamplingOptions controls selection; zero temperature is greedy and zero top-p disables filtering.
type SamplingOptions struct {
	Temperature float32
	TopP        float32
	Seed        uint64
}

// Validate checks sampling options without calling MLX.
func (o SamplingOptions) Validate() error {
	if math.IsNaN(float64(o.Temperature)) || math.IsInf(float64(o.Temperature), 0) || o.Temperature < 0 {
		return fmt.Errorf("lm: temperature must be finite and nonnegative")
	}
	if math.IsNaN(float64(o.TopP)) || math.IsInf(float64(o.TopP), 0) || o.TopP < 0 || o.TopP > 1 {
		return fmt.Errorf("lm: top-p must be in [0,1]")
	}
	return nil
}

// Sample generates with MLX's global RNG; other random operations can alter a seeded sequence.
func Sample(s Session, prompt []int32, count int, o SamplingOptions, eos ...int32) ([]int32, error) {
	return Stream(s, prompt, count, o, nil, eos...)
}

type scope struct {
	arrays []mlx.Array
	err    error
}

func (s *scope) add(a mlx.Array, err error) mlx.Array {
	if err != nil {
		if s.err == nil {
			s.err = err
		}
	} else {
		s.arrays = append(s.arrays, a)
	}
	return a
}

func sampledPick(logits mlx.Array, o SamplingOptions) (token int32, err error) {
	err = mlx.Batch(func() error {
		s := &scope{}
		defer func() { _ = mlx.CloseArrays(s.arrays) }()
		x := s.add(mlx.AsType(logits, mlx.Float32))
		temperature := s.add(mlx.NewScalarFloat32(o.Temperature))
		x = s.add(mlx.Divide(x, temperature))
		var order mlx.Array
		filtered := o.TopP > 0 && o.TopP < 1
		if filtered {
			negative := s.add(mlx.Negative(x))
			order = s.add(mlx.ArgSortAxis(negative, -1))
			sorted := s.add(mlx.SortAxis(negative, -1))
			x = s.add(mlx.Negative(sorted))
			probability := s.add(mlx.SoftmaxAxis(x, -1, true))
			cumulative := s.add(mlx.CumsumAxis(probability, -1))
			previous := s.add(mlx.Subtract(cumulative, probability))
			threshold := s.add(mlx.NewScalarFloat32(o.TopP))
			mask := s.add(mlx.Greater(previous, threshold))
			excluded := s.add(mlx.NewScalarFloat32(float32(math.Inf(-1))))
			x = s.add(mlx.Where(mask, excluded, x))
		}
		if s.err != nil {
			return s.err
		}
		choice := s.add(mlx.RandomCategorical(x, -1))
		if filtered {
			choice = s.add(mlx.ExpandDims(choice, -1))
			choice = s.add(mlx.TakeAlongAxis(order, choice, -1))
		}
		if s.err != nil {
			return s.err
		}
		token, err = tokenID(choice)
		return err
	})
	return token, err
}
