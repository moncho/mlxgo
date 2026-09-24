//go:build mlx && mlxruntime

package deepseek

import (
	"testing"

	mlx "github.com/moncho/mlxgo"
)

// Keep the released dimensions and >128 tokens: small model fixtures did not
// expose the CPU failure in the former rank-5 broadcasted projection.
func TestGroupedAttentionProjectionRealShape(t *testing.T) {
	const tokens, groups, width, rank = 131, 8, 4096, 1024
	input := make([]float32, tokens*groups*width)
	for i := range input {
		input[i] = float32((i*17)%257-128) / 128
	}
	weights := make([]float32, groups*rank*width)
	want := make([]float32, tokens*groups*rank)
	for g := 0; g < groups; g++ {
		for r := 0; r < rank; r++ {
			col := (r*3 + g*7) % (width - 1)
			a := float32(g+1) / 8
			weights[(g*rank+r)*width+col] = a
			weights[(g*rank+r)*width+width-1] = -.5
			for n := 0; n < tokens; n++ {
				want[(n*groups+g)*rank+r] = input[(n*groups+g)*width+col]*a - input[(n*groups+g)*width+width-1]*.5
			}
		}
	}
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			a, err := mlx.NewFloat32(input, []int{1, tokens, 64, 512})
			x := owned(t, a, err)
			a, err = mlx.NewFloat32(weights, []int{groups * rank, width})
			w := owned(t, a, err)
			a, err = one(func(s *scope) mlx.Array {
				return groupedAttentionProjection(s, x, w, Config{Heads: 64, HeadDim: 512, Groups: groups, ORank: rank})
			})
			y := owned(t, a, err)
			got, err := y.Float32Data()
			if err != nil {
				t.Fatal(err)
			}
			compareAttention(t, "real-shape grouped projection", got, want, 1e-6)
		})
	}
}
