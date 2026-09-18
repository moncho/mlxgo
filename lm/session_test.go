package lm

import (
	"errors"
	mlx "github.com/moncho/mlxgo"
	"testing"
)

type errorSession struct{ calls int }

func (s *errorSession) Step([]int32) (mlx.Array, error) {
	s.calls++
	return mlx.Array{}, errors.New("test error")
}
func (s *errorSession) Position() int { return 0 }
func (s *errorSession) Close() error  { return nil }

func TestGreedyValidation(t *testing.T) {
	s := &errorSession{}
	if _, err := Greedy(nil, []int32{1}, 2); err == nil {
		t.Fatal("nil session")
	}
	if _, err := Greedy(s, nil, 2); err == nil {
		t.Fatal("empty prompt")
	}
	if _, err := Greedy(s, []int32{1}, -1); err == nil {
		t.Fatal("negative count")
	}
	if got, err := Greedy(s, []int32{1}, 0); err != nil || len(got) != 0 || s.calls != 0 {
		t.Fatal("zero-count generation touched session")
	}
	if _, err := Greedy(s, []int32{1}, 1); err == nil || s.calls != 1 {
		t.Fatal("step error not propagated")
	}
}
