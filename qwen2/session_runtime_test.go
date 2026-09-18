//go:build mlx && mlxruntime

package qwen2

import (
	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/lm"
	"slices"
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
