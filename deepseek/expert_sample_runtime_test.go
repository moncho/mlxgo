//go:build mlx && mlxruntime

package deepseek

import (
	"math"
	"testing"

	mlx "github.com/moncho/mlxgo"
)

func TestReleasedExpertForward(t *testing.T) {
	dir, ref := readExpertReference(t)
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			arrays := make([]mlx.Array, 3)
			for i, m := range ref.Matrices {
				a, err := mlx.NewFloat32(loadExpertMatrix(t, dir, m), []int{m.Rows, m.Cols})
				arrays[i] = owned(t, a, err)
				read, err := a.Float32Data()
				if err != nil || matrixSHA(read) != m.DecodedSHA {
					t.Fatalf("expert native upload changed: %v", err)
				}
			}
			w := ExpertWeights{Gate: arrays[0], Down: arrays[1], Up: arrays[2]}
			a, err := mlx.NewFloat32(expertInputs(ref.Dim), []int{6, ref.Dim})
			x := owned(t, a, err)
			for _, c := range ref.Cases {
				t.Run(c.Name, func(t *testing.T) {
					a, err := mlx.NewFloat32(c.Routing, []int{6, 1})
					routing := owned(t, a, err)
					a, err = Expert(x, w, routing, c.Limit)
					y := owned(t, a, err)
					got, err := y.Float32Data()
					if err != nil || len(got) != len(c.Output) {
						t.Fatalf("invalid expert output: %v", err)
					}
					var maxAbs, maxNorm float64
					for i, v := range got {
						e := math.Abs(float64(v) - c.Output[i])
						if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || e > 3e-5+2e-6*c.L1[i] {
							t.Fatalf("output %d: got %g want %g abs_error=%g l1=%g", i, v, c.Output[i], e, c.L1[i])
						}
						if (i/ref.Dim == 4 || c.Routing[i/ref.Dim] == 0) && v != 0 {
							t.Fatalf("nonzero output for zero input/routing at %d", i)
						}
						maxAbs = math.Max(maxAbs, e)
						maxNorm = math.Max(maxNorm, e/math.Max(1, c.L1[i]))
					}
					t.Logf("%d expert outputs: max_abs_error=%g max_l1_normalized_error=%g", len(got), maxAbs, maxNorm)
				})
			}
		})
	}
}

func TestExpertClippingAndValidation(t *testing.T) {
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			checkExpertClippingAndValidation(t)
		})
	}
}

func checkExpertClippingAndValidation(t *testing.T) {
	a, err := mlx.NewFloat32([]float32{1, 0, 0, 1}, []int{2, 2})
	x := owned(t, a, err)
	a, err = mlx.NewFloat32([]float32{-20, 0, 20, 0}, []int{2, 2})
	gate := owned(t, a, err)
	a, err = mlx.NewFloat32([]float32{20, 0, -20, 0}, []int{2, 2})
	up := owned(t, a, err)
	a, err = mlx.NewFloat32([]float32{.25, 1}, []int{2, 1})
	routing := owned(t, a, err)
	w := ExpertWeights{Gate: gate, Up: up, Down: x}
	a, err = Expert(x, w, routing, 10)
	y := owned(t, a, err)
	got, err := y.Float32Data()
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{(-20 / (1 + math.Exp(20))) * 10 * .25, (10 / (1 + math.Exp(-10))) * -10 * .25, 0, 0}
	for i := range want {
		if math.Abs(float64(got[i])-want[i]) > 1e-6*math.Max(1, math.Abs(want[i])) {
			t.Fatal(got, want)
		}
	}
	for _, bad := range []float32{-1, float32(math.NaN()), float32(math.Inf(1))} {
		if a, err := Expert(x, w, routing, bad); err == nil {
			a.Close()
			t.Fatal("accepted invalid limit")
		}
	}
	for _, run := range []func() (mlx.Array, error){
		func() (mlx.Array, error) { return Expert(mlx.Array{}, w, routing, 10) },
		func() (mlx.Array, error) { return Expert(x, w, x, 10) },
		func() (mlx.Array, error) {
			return Expert(x, ExpertWeights{Gate: routing, Up: up, Down: x}, routing, 10)
		},
	} {
		if a, err := run(); err == nil {
			a.Close()
			t.Fatal("accepted invalid input")
		}
	}
}
