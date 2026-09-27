//go:build mlx && mlxruntime

package mlx

import (
	"fmt"
	"math"
	"sync"
	"testing"
)

func TestRuntimeFullDevices(t *testing.T) {
	defer SetDefaultCPU()
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", SetDefaultCPU}, {"gpu", SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			for _, dtype := range []DType{Bool, UInt8, UInt16, UInt32, UInt64, Int8, Int16, Int32, Int64, Float16, Float32, BFloat16, Complex64} {
				t.Run(dtype.String(), func(t *testing.T) {
					value := .75
					if dtype <= Int64 {
						value = 7.75
					}
					a, err := Full([]int{2, 3}, value, dtype)
					if err != nil {
						t.Fatal(err)
					}
					defer a.Close()
					gotType, err := a.DType()
					if err != nil || gotType != dtype {
						t.Fatal(gotType, err)
					}
					b, err := AsType(a, Float32)
					if err != nil {
						t.Fatal(err)
					}
					defer b.Close()
					got, err := b.Float32Data()
					if err != nil {
						t.Fatal(err)
					}
					want := float32(.75)
					if dtype <= Int64 {
						want = 7
					}
					if dtype == Bool {
						want = 1
					}
					if len(got) != 6 {
						t.Fatal(got)
					}
					for _, v := range got {
						if v != want {
							t.Fatal(got, want)
						}
					}
				})
			}
			// These integers lose bits if scalar construction goes through float32.
			for _, tc := range []struct {
				dtype DType
				value float64
			}{{Int32, 16777217}, {UInt32, 16777217}, {Int64, -1099511627777}, {UInt64, 1099511627777}} {
				a, err := Full([]int{2}, tc.value, tc.dtype)
				if err != nil {
					t.Fatal(err)
				}
				if tc.dtype == UInt64 || tc.dtype == UInt32 {
					b, err := AsType(a, UInt64)
					if err != nil {
						t.Fatal(err)
					}
					got, err := b.UInt64Data()
					b.Close()
					if err != nil || len(got) != 2 || got[0] != uint64(tc.value) || got[1] != uint64(tc.value) {
						t.Fatal(got, err)
					}
				} else {
					b, err := AsType(a, Int64)
					if err != nil {
						t.Fatal(err)
					}
					got, err := b.Int64Data()
					b.Close()
					if err != nil || len(got) != 2 || got[0] != int64(tc.value) || got[1] != int64(tc.value) {
						t.Fatal(got, err)
					}
				}
				a.Close()
			}
			for _, v := range []float64{math.NaN(), math.Inf(1), math.Inf(-1), math.Copysign(0, -1)} {
				a, err := Full(nil, v, Float32)
				if err != nil {
					t.Fatal(err)
				}
				got, err := a.Float32Data()
				a.Close()
				if err != nil || len(got) != 1 {
					t.Fatal(got, err)
				}
				if math.IsNaN(v) {
					if !math.IsNaN(float64(got[0])) {
						t.Fatal(got)
					}
				} else if float64(got[0]) != v || math.Signbit(float64(got[0])) != math.Signbit(v) {
					t.Fatal(got, v)
				}
			}
			a, err := Full([]int{0, 2}, 1, Float32)
			if err != nil {
				t.Fatal(err)
			}
			got, err := a.Float32Data()
			a.Close()
			if err != nil || len(got) != 0 {
				t.Fatal(got, err)
			}
			for _, tc := range []struct {
				shape []int
				dtype DType
			}{{[]int{-1}, Float32}, {[]int{1}, DType(-1)}} {
				a, err := Full(tc.shape, 1, tc.dtype)
				if err == nil {
					a.Close()
					t.Fatal("invalid full accepted")
				}
			}
			var wg sync.WaitGroup
			for range 8 {
				wg.Go(func() {
					for range 50 {
						a, err := Full([]int{2}, .75, Float32)
						if err != nil {
							t.Error(err)
							return
						}
						v, err := a.Float32Data()
						a.Close()
						if err != nil || len(v) != 2 || v[0] != .75 || v[1] != .75 {
							t.Error(fmt.Sprint(v, err))
							return
						}
					}
				})
			}
			wg.Wait()
		})
	}
	if err := SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}
	want := 1 + math.Ldexp(1, -40)
	a, err := Full([]int{2}, want, Float64)
	if err != nil {
		t.Fatal(err)
	}
	got, err := a.Float64Data()
	a.Close()
	if err != nil || len(got) != 2 || got[0] != want {
		t.Fatal(got, err)
	}
	if err := SetDefaultGPU(); err != nil {
		t.Fatal(err)
	}
	a, err = Full([]int{2}, want, Float64)
	if err == nil {
		err = a.Eval()
		a.Close()
	}
	if err == nil {
		t.Fatal("GPU float64 unexpectedly accepted")
	}
}

func TestRuntimeFullTransforms(t *testing.T) {
	defer SetDefaultCPU()
	for _, set := range []func() error{SetDefaultCPU, SetDefaultGPU} {
		if err := set(); err != nil {
			t.Fatal(err)
		}
		fn := func(in []Array) ([]Array, error) {
			fill, err := Full(in[0].Shape(), .75, Float32)
			if err != nil {
				return nil, err
			}
			defer fill.Close()
			y, err := Multiply(in[0], fill)
			if err != nil {
				return nil, err
			}
			defer y.Close()
			out, err := Sum(y, false)
			if err != nil {
				return nil, err
			}
			return []Array{out}, nil
		}
		x, err := NewFloat32([]float32{1, 2}, []int{2})
		if err != nil {
			t.Fatal(err)
		}
		compiled, err := Compile(fn, false)
		if err != nil {
			t.Fatal(err)
		}
		out, err := compiled.Apply(x)
		if err != nil {
			t.Fatal(err)
		}
		assertFloat32Data(t, out[0], []float32{2.25})
		CloseArrays(out)
		compiled.Close()
		vg, err := NewValueAndGrad(fn)
		if err != nil {
			t.Fatal(err)
		}
		values, grads, err := vg.Apply(x)
		if err != nil {
			t.Fatal(err)
		}
		assertFloat32Data(t, values[0], []float32{2.25})
		assertFloat32Data(t, grads[0], []float32{.75, .75})
		CloseArrays(values)
		CloseArrays(grads)
		vg.Close()
		x.Close()
	}
}
