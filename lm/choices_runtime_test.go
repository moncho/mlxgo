//go:build mlx && mlxruntime

package lm

import (
	"errors"
	"math"
	"slices"
	"sync"
	"testing"

	mlx "github.com/moncho/mlxgo"
)

func TestChoicesRuntimeMembershipAndOwnership(t *testing.T) {
	if err := mlx.SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}
	c, err := NewChoices([][]int32{{2, 0}, {2, 1, 0}})
	if err != nil {
		t.Fatal(err)
	}
	s := &scriptedSession{}
	got, err := c.Greedy(s, []int32{9}, 3)
	if err != nil || !slices.Equal(got, []int32{2, 0}) {
		t.Fatalf("got=%v err=%v", got, err)
	}
	if s.closed || len(s.inputs) != 2 || !slices.Equal(s.inputs[1], []int32{2}) {
		t.Fatal("invalid session ownership or token consumption")
	}
	for _, a := range s.arrays {
		if _, err := a.DType(); err == nil {
			t.Fatal("leaked logits")
		}
	}
	bad := &scriptedSession{badShape: true}
	if _, err := c.Greedy(bad, []int32{9}, 3); err == nil {
		t.Fatal("accepted invalid logits shape")
	}
	for _, a := range bad.arrays {
		if _, err := a.DType(); err == nil {
			t.Fatal("leaked failed logits")
		}
	}
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			got, err := c.Greedy(&scriptedSession{}, []int32{9}, 3)
			if err != nil || !slices.Equal(got, []int32{2, 0}) {
				t.Errorf("concurrent trie reuse: %v %v", got, err)
			}
		})
	}
	wg.Wait()
}

func TestConstrainedPickEdges(t *testing.T) {
	for _, test := range []struct {
		values []float32
		want   int32
		bad    bool
	}{
		{[]float32{99, 2, 2}, 1, false},
		{[]float32{99, float32(math.Inf(-1)), 2}, 2, false},
		{[]float32{99, float32(math.Inf(-1)), float32(math.Inf(-1))}, 0, true},
		{[]float32{99, float32(math.NaN()), 2}, 0, true},
		{[]float32{99, float32(math.Inf(1)), 2}, 0, true},
	} {
		a, err := mlx.NewFloat32(test.values, []int{1, 1, 3})
		if err != nil {
			t.Fatal(err)
		}
		got, err := constrainedPick(a, []int32{1, 2}, 2)
		a.Close()
		if (err != nil) != test.bad || !test.bad && got != test.want {
			t.Fatalf("got=%d err=%v want=%d bad=%v", got, err, test.want, test.bad)
		}
	}
	a, err := mlx.NewFloat32([]float32{1, 2, 3}, []int{1, 1, 3})
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	if _, err := constrainedPick(a, []int32{1}, 3); err == nil {
		t.Fatal("accepted out-of-vocabulary choice")
	}
	b, err := mlx.AsType(a, mlx.BFloat16)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	if got, err := constrainedPick(b, []int32{0, 2}, 2); err != nil || got != 2 {
		t.Fatal("BF16 selection failed", got, err)
	}
}

type failingChoiceSession struct{ scriptedSession }

var choiceStepError = errors.New("choice step failed")

func (s *failingChoiceSession) Step(input []int32) (mlx.Array, error) {
	if len(s.inputs) == 1 {
		return mlx.Array{}, choiceStepError
	}
	return s.scriptedSession.Step(input)
}

func TestChoicesLongPathAndPartialError(t *testing.T) {
	c, err := NewChoices([][]int32{{2, 1, 0}})
	if err != nil {
		t.Fatal(err)
	}
	s := &scriptedSession{}
	got, err := c.Greedy(s, []int32{9}, 3)
	if err != nil || !slices.Equal(got, []int32{2, 1, 0}) || len(s.inputs) != 3 {
		t.Fatalf("forced complete path: %v %v", got, err)
	}
	failed := &failingChoiceSession{}
	got, err = c.Greedy(failed, []int32{9}, 3)
	if !errors.Is(err, choiceStepError) || !slices.Equal(got, []int32{2}) || failed.closed {
		t.Fatalf("partial error contract: %v %v", got, err)
	}
	for _, a := range append(s.arrays, failed.arrays...) {
		if _, err := a.DType(); err == nil {
			t.Fatal("partial/complete generation leaked logits")
		}
	}
}
