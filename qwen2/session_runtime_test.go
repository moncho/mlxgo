//go:build mlx && mlxruntime

package qwen2

import (
	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/lm"
	"slices"
	"strings"
	"testing"
)

func TestSharedSessionAdapter(t *testing.T) {
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			c := tinyConfig()
			w := tinyWeights(t, c)
			s, err := NewSession(w, c)
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			prompt := []int32{1, 2, 3}
			got, err := lm.Greedy(s, prompt, 4)
			if err != nil {
				t.Fatal(err)
			}
			sequence := slices.Clone(prompt)
			var want []int32
			for range 4 {
				logits, err := Forward(w, c, sequence, nil)
				if err != nil {
					t.Fatal(err)
				}
				data := floatData(t, logits)
				logits.Close()
				best := 0
				for i := range data {
					if data[i] > data[best] {
						best = i
					}
				}
				want = append(want, int32(best))
				sequence = append(sequence, int32(best))
			}
			if !slices.Equal(got, want) {
				t.Fatalf("adapter tokens %v, want %v", got, want)
			}
			if s.Position() != len(prompt)+len(got)-1 {
				t.Fatal("wrong position")
			}
			s.Close()
			s.Close()
			if a, err := s.Step([]int32{1}); err == nil {
				a.Close()
				t.Fatal("used closed session")
			}
			a, err := Forward(w, c, prompt, nil)
			if err != nil {
				t.Fatal("closing session closed borrowed weights:", err)
			}
			a.Close()
		})
	}
}

func TestSessionWithTrainedAdapters(t *testing.T) {
	if err := mlx.SetDefaultGPU(); err != nil {
		t.Fatal(err)
	}
	c := tinyConfig()
	w := tinyWeights(t, c)
	a, err := NewAdapters(c, 2, 2, 42, strings.Repeat("0", 64))
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	if err = a.Train(w, []Example{{Inputs: []int32{1, 2, 3}, Targets: []int32{7, 7, 7}}}, TrainOptions{Steps: 3, BatchSize: 1, LearningRate: .02, Seed: 42}); err != nil {
		t.Fatal(err)
	}
	s, err := NewSessionWithAdapters(w, c, a)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	var all []int32
	for _, ids := range [][]int32{{1, 2, 3}, {4}, {5}} {
		all = append(all, ids...)
		got, err := s.Step(ids)
		if err != nil {
			t.Fatal(err)
		}
		want, err := a.Forward(w, all, nil)
		if err != nil {
			got.Close()
			t.Fatal(err)
		}
		maxError(t, floatData(t, got), floatData(t, want), 1e-4)
		got.Close()
		want.Close()
	}
	if _, err := NewSessionWithAdapters(w, c, nil); err == nil {
		t.Fatal("accepted nil adapters")
	}
	bad := c
	bad.MaxPositions++
	if _, err := NewSessionWithAdapters(w, bad, a); err == nil {
		t.Fatal("accepted mismatched config")
	}
	s.Close()
	out, err := a.Forward(w, []int32{1}, nil)
	if err != nil {
		t.Fatal("session closed borrowed adapters:", err)
	}
	out.Close()
}
