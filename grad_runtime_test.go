//go:build mlx && mlxruntime

package mlx

import (
	"math"
	"slices"
	"strings"
	"testing"
)

func TestClipGradNorm(t *testing.T) {
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", SetDefaultCPU}, {"gpu", SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			for _, tc := range []struct {
				name               string
				data               []float32
				limit, norm, scale float32
			}{
				{"clipped", []float32{3, -4, 12}, 6.5, 13, .5},
				{"below", []float32{3, -4, 12}, 20, 13, 1},
				{"boundary", []float32{3, -4, 12}, 13, 13, 1},
				{"disabled", []float32{3, -4, 12}, 0, 13, 1},
				{"zeros", []float32{0, 0, 0}, 1, 0, 1},
			} {
				t.Run(tc.name, func(t *testing.T) {
					// Unequal tensor sizes distinguish global from per-tensor clipping.
					grads := []Array{mustNewFloat32(t, tc.data[:2], []int{1, 2}), mustNewFloat32(t, tc.data[2:], []int{})}
					defer CloseArrays(grads)
					out, norm, err := ClipGradNorm(grads, tc.limit)
					if err != nil {
						t.Fatal(err)
					}
					defer CloseArrays(out)
					if norm != tc.norm {
						t.Fatalf("norm=%g want %g", norm, tc.norm)
					}
					var got []float32
					for i, a := range out {
						if !slices.Equal(a.Shape(), grads[i].Shape()) {
							t.Fatal("shape changed")
						}
						d, err := a.Float32Data()
						if err != nil {
							t.Fatal(err)
						}
						got = append(got, d...)
					}
					for i, v := range got {
						if v != tc.data[i]*tc.scale {
							t.Fatalf("element %d: got %g want %g", i, v, tc.data[i]*tc.scale)
						}
					}
					_ = CloseArrays(out)
					var original []float32
					for _, g := range grads {
						d, err := g.Float32Data()
						if err != nil {
							t.Fatalf("closing output invalidated input: %v", err)
						}
						original = append(original, d...)
					}
					if !slices.Equal(original, tc.data) {
						t.Fatal("inputs changed")
					}
				})
			}
			t.Run("strided", func(t *testing.T) {
				base := mustNewFloat32(t, []float32{3, 4}, []int{2})
				defer base.Close()
				view, err := BroadcastTo(base, []int{4, 2})
				if err != nil {
					t.Fatal(err)
				}
				defer view.Close()
				out, norm, err := ClipGradNorm([]Array{view}, 5)
				if err != nil {
					t.Fatal(err)
				}
				defer CloseArrays(out)
				data, err := out[0].Float32Data()
				if err != nil || norm != 10 || !slices.Equal(data, []float32{1.5, 2, 1.5, 2, 1.5, 2, 1.5, 2}) {
					t.Fatalf("strided: norm=%g data=%v err=%v", norm, data, err)
				}
			})
			for _, value := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)), math.MaxFloat32} {
				for _, limit := range []float32{0, 1} {
					g := mustNewFloat32(t, []float32{value}, []int{1})
					out, _, err := ClipGradNorm([]Array{g}, limit)
					_ = g.Close()
					_ = CloseArrays(out)
					if err == nil || !strings.Contains(err.Error(), "nonfinite gradient norm") || out != nil {
						t.Fatalf("value=%g limit=%g: out=%v err=%v", value, limit, out, err)
					}
				}
			}
		})
	}
}

func TestClipGradNormInvalidArrays(t *testing.T) {
	valid := mustNewFloat32(t, []float32{1}, []int{1})
	defer valid.Close()
	closed := mustNewFloat32(t, []float32{1}, []int{1})
	_ = closed.Close()
	integer, err := NewInt32([]int32{1}, []int{1})
	if err != nil {
		t.Fatal(err)
	}
	defer integer.Close()
	for _, bad := range []Array{{}, closed, integer} {
		out, _, err := ClipGradNorm([]Array{valid, bad}, 1)
		_ = CloseArrays(out)
		if err == nil || out != nil {
			t.Fatalf("invalid array accepted: out=%v err=%v", out, err)
		}
	}
}
