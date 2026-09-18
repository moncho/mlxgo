//go:build mlx && mlxruntime

package mlx

import (
	"math"
	"testing"
)

func TestLog1pPrecision(t *testing.T) {
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", SetDefaultCPU}, {"gpu", SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			input := []float32{-.5, -1e-8, 0, 1e-8, 1, 100}
			x, err := NewFloat32(input, []int{len(input)})
			if err != nil {
				t.Fatal(err)
			}
			defer x.Close()
			y, err := Log1p(x)
			if err != nil {
				t.Fatal(err)
			}
			defer y.Close()
			data, err := y.Float32Data()
			if err != nil {
				t.Fatal(err)
			}
			for i, v := range data {
				want := math.Log1p(float64(input[i]))
				if math.IsNaN(float64(v)) || math.Abs(float64(v)-want) > 1e-6*math.Abs(want)+1e-15 {
					t.Fatalf("Log1p(%g)=%g, want %g", input[i], v, want)
				}
			}
		})
	}
}
