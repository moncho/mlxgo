//go:build mlx && mlxruntime

package lm

import (
	mlx "github.com/moncho/mlxgo"
	"slices"
	"testing"
)

type scriptedSession struct {
	inputs           [][]int32
	arrays           []mlx.Array
	badShape, closed bool
}

func (s *scriptedSession) Position() int { return len(s.inputs) }
func (s *scriptedSession) Close() error  { s.closed = true; return nil }
func (s *scriptedSession) Step(input []int32) (mlx.Array, error) {
	s.inputs = append(s.inputs, slices.Clone(input))
	shape := []int{1, 1, 3}
	if s.badShape {
		shape = []int{3}
	}
	values := []float32{0, 2, 1}
	if len(s.inputs) > 1 {
		values = []float32{3, 1, 0}
	}
	a, err := mlx.NewFloat32(values, shape)
	if err == nil {
		s.arrays = append(s.arrays, a)
	}
	return a, err
}
func TestGreedyEOSAndOwnership(t *testing.T) {
	if err := mlx.SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}
	s := &scriptedSession{}
	got, err := Greedy(s, []int32{7, 8}, 10, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []int32{1, 0}) || len(s.inputs) != 2 || !slices.Equal(s.inputs[1], []int32{1}) {
		t.Fatalf("EOS generation: %v %v", got, s.inputs)
	}
	if s.closed {
		t.Fatal("decoder closed caller-owned session")
	}
	for _, a := range s.arrays {
		if _, err := a.DType(); err == nil {
			t.Fatal("decoder leaked logits")
		}
	}
	bad := &scriptedSession{badShape: true}
	if _, err := Greedy(bad, []int32{1}, 1); err == nil {
		t.Fatal("accepted invalid shape")
	}
	for _, a := range bad.arrays {
		if _, err := a.DType(); err == nil {
			t.Fatal("invalid shape leaked logits")
		}
	}
}
