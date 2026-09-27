//go:build mlx && mlxruntime

package lm

import (
	"math"
	"slices"
	"testing"

	mlx "github.com/moncho/mlxgo"
)

type fixedSession struct{ values []float32 }

func (s *fixedSession) Step([]int32) (mlx.Array, error) {
	return mlx.NewFloat32(s.values, []int{1, 1, len(s.values)})
}
func (s *fixedSession) Position() int { return 0 }
func (s *fixedSession) Close() error  { return nil }

func TestSampling(t *testing.T) {
	for _, device := range []mlx.DeviceType{mlx.DeviceCPU, mlx.DeviceGPU} {
		if err := mlx.SetDefaultDevice(device, 0); err != nil {
			t.Fatal(err)
		}
		s := &fixedSession{[]float32{1, 4, 2, 3, 0}}
		greedy, err := Greedy(s, []int32{0}, 32)
		if err != nil {
			t.Fatal(err)
		}
		zero, err := Sample(s, []int32{0}, 32, SamplingOptions{})
		if err != nil || !slices.Equal(greedy, zero) {
			t.Fatalf("greedy mismatch %v %v", zero, err)
		}
		o := SamplingOptions{Temperature: 2, TopP: .9, Seed: 23}
		a, err := Sample(s, []int32{0}, 64, o)
		if err != nil {
			t.Fatal(err)
		}
		b, err := Sample(s, []int32{0}, 64, o)
		if err != nil || !slices.Equal(a, b) {
			t.Fatalf("seed mismatch: %v", err)
		}
		o.Seed++
		c, err := Sample(s, []int32{0}, 64, o)
		if err != nil || slices.Equal(a, c) {
			t.Fatalf("seed did not change output: %v", err)
		}
		for _, id := range append(a, c...) {
			if id < 0 || int(id) >= len(s.values) {
				t.Fatalf("invalid ID %d", id)
			}
		}
		for _, p := range []float32{.1, .8} {
			logits, err := s.Step(nil)
			if err != nil {
				t.Fatal(err)
			}
			values, err := logits.Float32Data()
			logits.Close()
			if err != nil {
				t.Fatal(err)
			}
			order := []int{0, 1, 2, 3, 4}
			slices.SortFunc(order, func(i, j int) int {
				if values[i] > values[j] {
					return -1
				}
				return 1
			})
			total := 0.0
			for _, v := range values {
				total += math.Exp(float64(v))
			}
			allowed := map[int32]bool{}
			cumulative := 0.0
			for _, id := range order {
				if cumulative > float64(p) {
					break
				}
				allowed[int32(id)] = true
				cumulative += math.Exp(float64(values[id])) / total
			}
			ids, err := Sample(s, []int32{0}, 128, SamplingOptions{Temperature: 1, TopP: p})
			if err != nil {
				t.Fatal(err)
			}
			for _, id := range ids {
				if !allowed[id] {
					t.Fatalf("p=%v selected %d outside %v", p, id, allowed)
				}
			}
		}
	}
}
