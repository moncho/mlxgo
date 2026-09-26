//go:build mlx && mlxruntime

package quant

import (
	"fmt"
	"math"
	"reflect"
	"sync"
	"testing"

	mlx "github.com/moncho/mlxgo"
)

func linearDevices(t *testing.T, fn func(*testing.T)) {
	t.Helper()
	defer mlx.SetDefaultCPU()
	for _, d := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(d.name, func(t *testing.T) {
			if err := d.set(); err != nil {
				t.Fatal(err)
			}
			fn(t)
		})
	}
}

func TestReleasedFP8Linear(t *testing.T) {
	dir, cases := readReleasedReference(t)
	c := cases[0] // wkv: no grouped projection or BF16 checkpoint conversion.
	data, scales, format := releasedBytes(t, dir, c)
	linearDevices(t, func(t *testing.T) {
		layer, err := NewFP8Linear(data, scales, c.Rows, c.Cols, format)
		if err != nil {
			t.Fatal(err)
		}
		defer layer.Close()
		for _, tokens := range []int{1, 3, 32, 128} {
			t.Run(fmt.Sprintf("tokens%d", tokens), func(t *testing.T) {
				base := releasedInputs(c)
				input := make([]float32, tokens*c.Cols)
				for m := 0; m < tokens; m++ {
					copy(input[m*c.Cols:(m+1)*c.Cols], base[(m%3)*c.Cols:(m%3+1)*c.Cols])
				}
				x, err := mlx.NewFloat32(input, []int{tokens, c.Cols})
				if err != nil {
					t.Fatal(err)
				}
				defer x.Close()
				y, err := layer.Forward(x)
				if err != nil {
					t.Fatal(err)
				}
				defer y.Close()
				got, err := y.Float32Data()
				if err != nil {
					t.Fatal(err)
				}
				if len(got) != tokens*c.Rows {
					t.Fatal("incorrect projection length")
				}
				var maxAbs, maxNormalized float64
				for i, v := range got {
					j := i % len(c.Projection)
					delta := math.Abs(float64(v) - c.Projection[j])
					l1 := c.ProjectionL1[j]
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || delta > 1e-5+2e-6*l1 {
						t.Fatalf("projection %d: got %g want %g error %g l1 %g", i, v, c.Projection[j], delta, l1)
					}
					maxAbs = math.Max(maxAbs, delta)
					maxNormalized = math.Max(maxNormalized, delta/math.Max(1, l1))
				}
				t.Logf("%d real-weight projections: max_abs=%g max_l1_normalized=%g; packed payload=%d bytes, float32=%d bytes", len(got), maxAbs, maxNormalized, c.Rows*c.Cols+c.Rows*c.Cols/32, 4*c.Rows*c.Cols)
			})
		}
	})
}

func TestFP8LinearCompiled(t *testing.T) {
	linearDevices(t, func(t *testing.T) {
		data := make([]byte, 32*32)
		for i := range data {
			data[i] = 0x38
		}
		layer, err := NewFP8Linear(data, []byte{127}, 32, 32, FP8Block32)
		if err != nil {
			t.Fatal(err)
		}
		defer layer.Close()
		compiled, err := mlx.Compile(func(in []mlx.Array) ([]mlx.Array, error) {
			y, err := layer.Forward(in[0])
			if err != nil {
				return nil, err
			}
			return []mlx.Array{y}, nil
		}, false)
		if err != nil {
			t.Fatal(err)
		}
		defer compiled.Close()
		for _, tokens := range []int{1, 3, 33} {
			// Transpose produces a non-row-contiguous input for the kernel.
			values := make([]float32, 32*tokens)
			for i := range values {
				values[i] = float32(i%tokens + 1)
			}
			base, err := mlx.NewFloat32(values, []int{32, tokens})
			if err != nil {
				t.Fatal(err)
			}
			defer base.Close()
			x, err := mlx.Transpose(base)
			if err != nil {
				t.Fatal(err)
			}
			defer x.Close()
			for repeat := 0; repeat < 2; repeat++ {
				out, err := compiled.Apply(x)
				if err != nil {
					t.Fatal(err)
				}
				if len(out) != 1 {
					for _, a := range out {
						a.Close()
					}
					t.Fatal("compiled output count")
				}
				got, err := out[0].Float32Data()
				out[0].Close()
				if err != nil {
					t.Fatal(err)
				}
				for i, v := range got {
					if v != float32(32*(i/32+1)) {
						t.Fatalf("compiled output %d: %g", i, v)
					}
				}
			}
		}
	})
}

func TestFP8Linear(t *testing.T) {
	linearDevices(t, func(t *testing.T) {
		for _, format := range []Format{FP8Block32, FP8Row32} {
			for _, tokens := range []int{1, 3, 16, 33, 128} {
				t.Run(fmt.Sprintf("format%d/tokens%d", format, tokens), func(t *testing.T) {
					const rows, cols = 64, 96
					data := make([]byte, rows*cols)
					for i := range data {
						data[i] = byte(i%127) | byte((i/127)%2)<<7
					}
					ns := rows * cols / 32
					if format == FP8Block32 {
						ns /= 32
					}
					scales := make([]byte, ns)
					for i := range scales {
						scales[i] = byte(120 + i%12)
					}
					decoded := make([]float32, rows*cols)
					if err := Decode(decoded, data, scales, rows, cols, format, Float32); err != nil {
						t.Fatal(err)
					}
					layer, err := NewFP8Linear(data, scales, rows, cols, format)
					if err != nil {
						t.Fatal(err)
					}
					defer layer.Close()
					if dt, _ := layer.weights.DType(); dt != mlx.UInt32 {
						t.Fatal("weights are not packed UInt32")
					}
					if dt, _ := layer.scales.DType(); dt != mlx.UInt8 {
						t.Fatal("scales are not UInt8")
					}
					if !reflect.DeepEqual(layer.weights.Shape(), []int{rows, cols / 4}) || !reflect.DeepEqual(layer.scales.Shape(), []int{rows, cols / 32}) {
						t.Fatal("incorrect packed shape")
					}
					// Upload must copy caller-owned bytes before returning.
					clear(data)
					clear(scales)
					input := make([]float32, tokens*cols)
					for i := range input {
						input[i] = float32(i*37%257-128) / 128
					}
					x, err := mlx.NewFloat32(input, []int{tokens, cols})
					if err != nil {
						t.Fatal(err)
					}
					defer x.Close()
					y, err := layer.Forward(x)
					if err != nil {
						t.Fatal(err)
					}
					defer y.Close()
					// Graph owns its dependencies even if the layer is closed before eval.
					copyOfLayer := *layer
					if err := layer.Close(); err != nil {
						t.Fatal(err)
					}
					if z, err := copyOfLayer.Forward(x); err == nil {
						z.Close()
						t.Fatal("closed copy accepted")
					}
					got, err := y.Float32Data()
					if err != nil {
						t.Fatal(err)
					}
					if len(got) != tokens*rows {
						t.Fatal("output shape")
					}
					for m := 0; m < tokens; m++ {
						for r := 0; r < rows; r++ {
							var want, l1 float64
							for k := 0; k < cols; k++ {
								p := float64(input[m*cols+k]) * float64(decoded[r*cols+k])
								want += p
								l1 += math.Abs(p)
							}
							v := float64(got[m*rows+r])
							if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v-want) > 1e-5+2e-6*l1 {
								t.Fatalf("[%d,%d] got %g want %g l1 %g", m, r, v, want, l1)
							}
						}
					}
				})
			}
		}
	})
}

func TestFP8LinearConcurrent(t *testing.T) {
	linearDevices(t, func(t *testing.T) {
		data := make([]byte, 32*32)
		for i := range data {
			data[i] = 0x38
		} // E4M3 1.0
		layer, err := NewFP8Linear(data, []byte{127}, 32, 32, FP8Block32)
		if err != nil {
			t.Fatal(err)
		}
		defer layer.Close()
		var wg sync.WaitGroup
		errs := make(chan error, 8)
		for g := 0; g < 8; g++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for i := 0; i < 50; i++ {
					// Deliberately separate build/eval/close dispatches across callers.
					x, err := mlx.Ones([]int{1, 32}, mlx.Float32)
					if err != nil {
						errs <- err
						return
					}
					y, err := layer.Forward(x)
					x.Close()
					if err != nil {
						errs <- err
						return
					}
					v, err := y.Float32Data()
					y.Close()
					if err != nil {
						errs <- err
						return
					}
					for _, n := range v {
						if n != 32 {
							errs <- fmt.Errorf("got %g want 32", n)
							return
						}
					}
				}
			}()
		}
		wg.Wait()
		close(errs)
		for err := range errs {
			t.Error(err)
		}
	})
}

func BenchmarkReleasedFP8Linear(b *testing.B) {
	dir, cases := readReleasedReference(b)
	c := cases[0]
	data, scales, format := releasedBytes(b, dir, c)
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		b.Run(device.name, func(b *testing.B) {
			if err := device.set(); err != nil {
				b.Fatal(err)
			}
			for _, tokens := range []int{1, 32, 128} {
				b.Run(fmt.Sprintf("tokens%d", tokens), func(b *testing.B) {
					for _, packed := range []bool{false, true} {
						name := "float32"
						if packed {
							name = "packed"
						}
						b.Run(name, func(b *testing.B) {
							var layer *FP8Linear
							var wt mlx.Array
							if packed {
								var err error
								layer, err = NewFP8Linear(data, scales, c.Rows, c.Cols, format)
								if err != nil {
									b.Fatal(err)
								}
								defer layer.Close()
							} else {
								decoded := make([]float32, c.Rows*c.Cols)
								if err := Decode(decoded, data, scales, c.Rows, c.Cols, format, Float32); err != nil {
									b.Fatal(err)
								}
								w, err := mlx.NewFloat32(decoded, []int{c.Rows, c.Cols})
								if err != nil {
									b.Fatal(err)
								}
								defer w.Close()
								wt, err = mlx.Transpose(w)
								if err != nil {
									b.Fatal(err)
								}
								defer wt.Close()
							}
							input := make([]float32, tokens*c.Cols)
							for i := range input {
								input[i] = float32(i*37%257-128) / 128
							}
							x, err := mlx.NewFloat32(input, []int{tokens, c.Cols})
							if err != nil {
								b.Fatal(err)
							}
							defer x.Close()
							step := func() error {
								var y mlx.Array
								var err error
								if packed {
									y, err = layer.Forward(x)
								} else {
									y, err = mlx.Matmul(x, wt)
								}
								if err != nil {
									return err
								}
								defer y.Close()
								return mlx.Eval(y)
							}
							for i := 0; i < 5; i++ {
								if err := mlx.Batch(step); err != nil {
									b.Fatal(err)
								}
							}
							b.ResetTimer()
							for i := 0; i < b.N; i++ {
								if err := mlx.Batch(step); err != nil {
									b.Fatal(err)
								}
							}
							b.StopTimer()
							payload := 4 * c.Rows * c.Cols
							if packed {
								payload = c.Rows*c.Cols + c.Rows*c.Cols/32
							}
							b.ReportMetric(float64(payload), "weight-bytes")
						})
					}
				})
			}
		})
	}
}
