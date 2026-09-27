//go:build mlx && mlxruntime

package quant

import (
	"fmt"
	"math"
	"slices"
	"sync"
	"testing"

	mlx "github.com/moncho/mlxgo"
)

func TestFP4Linear(t *testing.T) {
	linearDevices(t, func(t *testing.T) {
		const rows, cols = 64, 96
		data, scales := make([]byte, rows*cols/2), make([]byte, rows*cols/32)
		for i := range data {
			data[i] = byte(i)
		}
		for i := range scales {
			scales[i] = byte(123 + i%7)
		}
		decoded := make([]float32, rows*cols)
		if err := Decode(decoded, data, scales, rows, cols, FP4Row32, Float32); err != nil {
			t.Fatal(err)
		}
		p, err := NewFP4Linear(data, scales, rows, cols)
		if err != nil {
			t.Fatal(err)
		}
		defer p.Close()
		if !slices.Equal(p.weights.Shape(), []int{rows, cols / 8}) || !slices.Equal(p.scales.Shape(), []int{rows, cols / 32}) {
			t.Fatal("incorrect packed storage")
		}
		clear(data)
		clear(scales)
		compiled, err := mlx.Compile(func(in []mlx.Array) ([]mlx.Array, error) {
			y, err := p.Forward(in[0])
			if err != nil {
				return nil, err
			}
			return []mlx.Array{y}, nil
		}, false)
		if err != nil {
			t.Fatal(err)
		}
		defer compiled.Close()
		for _, tokens := range []int{1, 3, 32, 128} {
			t.Run(fmt.Sprintf("tokens%d", tokens), func(t *testing.T) {
				// Build a transposed view to exercise noncontiguous kernel input.
				input := make([]float32, cols*tokens)
				for k := 0; k < cols; k++ {
					for n := 0; n < tokens; n++ {
						input[k*tokens+n] = float32((k*37+n*17)%257-128) / 128
					}
				}
				a, err := mlx.NewFloat32(input, []int{cols, tokens})
				if err != nil {
					t.Fatal(err)
				}
				defer a.Close()
				x, err := mlx.Transpose(a)
				if err != nil {
					t.Fatal(err)
				}
				defer x.Close()
				for _, compile := range []bool{false, true} {
					var y mlx.Array
					if compile {
						ys, e := compiled.Apply(x)
						err = e
						if err == nil {
							if len(ys) != 1 {
								mlx.CloseArrays(ys)
								t.Fatal("output count")
							}
							y = ys[0]
						}
					} else {
						y, err = p.Forward(x)
					}
					if err != nil {
						t.Fatal(err)
					}
					got, err := y.Float32Data()
					y.Close()
					if err != nil {
						t.Fatal(err)
					}
					if len(got) != tokens*rows {
						t.Fatal("output length")
					}
					for n := 0; n < tokens; n++ {
						for r := 0; r < rows; r++ {
							var want, l1 float64
							for k := 0; k < cols; k++ {
								v := float64(input[k*tokens+n]) * float64(decoded[r*cols+k])
								want += v
								l1 += math.Abs(v)
							}
							v := float64(got[n*rows+r])
							if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v-want) > 1e-5+2e-6*l1 {
								t.Fatalf("compiled=%t n=%d r=%d got=%g want=%g", compile, n, r, v, want)
							}
						}
					}
				}
			})
		}
		compiled.Close()
		x, err := mlx.Ones([]int{1, cols}, mlx.Float32)
		if err != nil {
			t.Fatal(err)
		}
		defer x.Close()
		var wg sync.WaitGroup
		errs := make(chan error, 8)
		for i := 0; i < 8; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				y, err := p.Forward(x)
				if err == nil {
					err = mlx.Eval(y)
					y.Close()
				}
				errs <- err
			}()
		}
		wg.Wait()
		close(errs)
		for err := range errs {
			if err != nil {
				t.Fatal(err)
			}
		}
		bad, err := mlx.AsType(x, mlx.Int32)
		if err != nil {
			t.Fatal(err)
		}
		defer bad.Close()
		if y, err := p.Forward(bad); err == nil {
			y.Close()
			t.Fatal("accepted nonfloat input")
		}
		y, err := p.Forward(x)
		if err != nil {
			t.Fatal(err)
		}
		defer y.Close()
		copyOf := *p
		if err := p.Close(); err != nil {
			t.Fatal(err)
		}
		if z, err := copyOf.Forward(x); err == nil {
			z.Close()
			t.Fatal("closed copy accepted")
		}
		if err := mlx.Eval(y); err != nil {
			t.Fatal("lazy output lost dependencies", err)
		}
	})
}

func TestReleasedFP4Linear(t *testing.T) {
	dir, cases := readReleasedReference(t)
	c := cases[1]
	d, sc, _ := releasedBytes(t, dir, c)
	linearDevices(t, func(t *testing.T) {
		p, err := NewFP4Linear(d, sc, c.Rows, c.Cols)
		if err != nil {
			t.Fatal(err)
		}
		defer p.Close()
		for _, tokens := range []int{1, 3, 32, 128} {
			t.Run(fmt.Sprintf("tokens%d", tokens), func(t *testing.T) {
				base := releasedInputs(c)
				input := make([]float32, tokens*c.Cols)
				for n := 0; n < tokens; n++ {
					copy(input[n*c.Cols:], base[(n%3)*c.Cols:(n%3+1)*c.Cols])
				}
				x, err := mlx.NewFloat32(input, []int{tokens, c.Cols})
				if err != nil {
					t.Fatal(err)
				}
				defer x.Close()
				y, err := p.Forward(x)
				if err != nil {
					t.Fatal(err)
				}
				defer y.Close()
				got, err := y.Float32Data()
				if err != nil {
					t.Fatal(err)
				}
				if len(got) != tokens*c.Rows {
					t.Fatal("output length")
				}
				var maxAbs, maxNorm float64
				for i, v := range got {
					j := i % len(c.Projection)
					delta := math.Abs(float64(v) - c.Projection[j])
					l1 := c.ProjectionL1[j]
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || delta > 1e-5+2e-6*l1 {
						t.Fatalf("projection %d got=%g want=%g error=%g l1=%g", i, v, c.Projection[j], delta, l1)
					}
					maxAbs = math.Max(maxAbs, delta)
					maxNorm = math.Max(maxNorm, delta/math.Max(1, l1))
				}
				t.Logf("%d real FP4 projections max_abs=%g max_l1_normalized=%g", len(got), maxAbs, maxNorm)
			})
		}
	})
}
