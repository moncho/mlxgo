//go:build mlx && mlxruntime

package quant

import (
	"math"
	"testing"

	mlx "github.com/moncho/mlxgo"
)

func TestDecodedWeightsInMLX(t *testing.T) {
	f := readFixture(t)
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			for _, c := range f.Cases {
				// Subnormal behavior is tested exhaustively on the host. This
				// smoke test does not assume Metal preserves denormal arithmetic.
				if len(c.Input) == 0 {
					continue
				}
				t.Run(c.Name, func(t *testing.T) {
					data := decodeCase(t, c)
					w, err := mlx.NewFloat32(data, []int{c.Rows, c.Cols})
					if err != nil {
						t.Fatal(err)
					}
					defer w.Close()
					stored := w
					if c.Rounding == "bfloat16" {
						stored, err = mlx.AsType(w, mlx.BFloat16)
						if err != nil {
							t.Fatal(err)
						}
						defer stored.Close()
					}
					wf, err := mlx.AsType(stored, mlx.Float32)
					if err != nil {
						t.Fatal(err)
					}
					defer wf.Close()
					read, err := wf.Float32Data()
					if err != nil {
						t.Fatal(err)
					}
					if len(read) != len(data) {
						t.Fatal("shape changed")
					}
					for i, v := range read {
						if v != data[i] {
							t.Fatalf("native upload changed element %d", i)
						}
					}
					transposed, err := mlx.Transpose(wf)
					if err != nil {
						t.Fatal(err)
					}
					defer transposed.Close()
					input := make([]float32, len(c.Input))
					for i, b := range c.Input {
						input[i] = math.Float32frombits(b)
					}
					x, err := mlx.NewFloat32(input, []int{2, c.Cols})
					if err != nil {
						t.Fatal(err)
					}
					defer x.Close()
					y, err := mlx.Matmul(x, transposed)
					if err != nil {
						t.Fatal(err)
					}
					defer y.Close()
					got, err := y.Float32Data()
					if err != nil {
						t.Fatal(err)
					}
					if len(got) != len(c.Projection) {
						t.Fatal("unexpected projection shape")
					}
					for i, v := range got {
						want := float64(math.Float32frombits(c.Projection[i]))
						if math.IsNaN(float64(v)) || math.Abs(float64(v)-want) > 1e-5*math.Max(1, math.Abs(want)) {
							t.Fatalf("projection %d: got %g want %g", i, v, want)
						}
					}
				})
			}
		})
	}
}

func TestDecodeOnMLXWorker(t *testing.T) {
	for _, c := range readFixture(t).Cases {
		t.Run(c.Name, func(t *testing.T) {
			want := decodeCase(t, c)
			dst := make([]float32, len(want))
			format, rounding := options(t, c)
			if err := mlx.Batch(func() error {
				return Decode(dst, c.Data, c.Scales, c.Rows, c.Cols, format, rounding)
			}); err != nil {
				t.Fatal(err)
			}
			for i, v := range dst {
				if math.Float32bits(v) != math.Float32bits(want[i]) {
					t.Fatalf("MLX worker decoding differs at %d: %08x != %08x", i, math.Float32bits(v), math.Float32bits(want[i]))
				}
			}
		})
	}
}
