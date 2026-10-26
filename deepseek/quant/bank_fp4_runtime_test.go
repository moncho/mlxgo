//go:build mlx && mlxruntime

package quant

import (
	"math"
	"testing"

	mlx "github.com/moncho/mlxgo"
)

func TestFP4LinearBankViews(t *testing.T) {
	linearDevices(t, func(t *testing.T) {
		const count, rows, cols, tokens = 3, 64, 96, 5
		beforeHandles := mlx.LiveArrays()
		func() {
			layers := make([]*FP4Linear, count)
			decoded := make([][]float32, count)
			for e := range layers {
				data, scales := make([]byte, rows*cols/2), make([]byte, rows*cols/32)
				for i := range data {
					data[i] = byte(i*13 + e*59)
				}
				for i := range scales {
					scales[i] = byte(121 + (i+e*3)%7)
				}
				decoded[e] = make([]float32, rows*cols)
				if err := Decode(decoded[e], data, scales, rows, cols, FP4Row32, Float32); err != nil {
					t.Fatal(err)
				}
				var err error
				layers[e], err = NewFP4Linear(data, scales, rows, cols)
				if err != nil {
					t.Fatal(err)
				}
				defer layers[e].Close()
			}
			bank, err := StackFP4Linears(layers)
			if err != nil {
				t.Fatal(err)
			}
			defer bank.Close()
			for _, l := range layers {
				if err := l.Close(); err != nil {
					t.Fatal(err)
				}
			}
			if err := mlx.Synchronize(); err != nil {
				t.Fatal(err)
			}
			before, err := mlx.GetMemoryUsage()
			if err != nil {
				t.Fatal(err)
			}
			views, retained := make([]*FP4Linear, count), make([]*FP4Linear, count)
			for e := range views {
				views[e], err = bank.Linear(e)
				if err != nil {
					t.Fatal(err)
				}
				defer views[e].Close()
				retained[e], err = views[e].Retain()
				if err != nil {
					t.Fatal(err)
				}
				defer retained[e].Close()
				if err := mlx.Eval(views[e].weights, views[e].scales, retained[e].weights, retained[e].scales); err != nil {
					t.Fatal(err)
				}
			}
			if err := mlx.Synchronize(); err != nil {
				t.Fatal(err)
			}
			after, err := mlx.GetMemoryUsage()
			if err != nil {
				t.Fatal(err)
			}
			if after.ActiveBytes != before.ActiveBytes {
				t.Fatalf("views copied weight storage: active bytes %d -> %d", before.ActiveBytes, after.ActiveBytes)
			}
			copyOfBank := *bank
			if err := bank.Close(); err != nil {
				t.Fatal(err)
			}
			if l, err := copyOfBank.Linear(0); err == nil {
				l.Close()
				t.Fatal("closed bank copy accepted")
			}
			input := make([]float32, tokens*cols)
			for i := range input {
				input[i] = float32((i*17)%67-33) / 64
			}
			x, err := mlx.NewFloat32(input, []int{tokens, cols})
			if err != nil {
				t.Fatal(err)
			}
			defer x.Close()
			for e := range views {
				// Build a lazy output, then close its projection. Both the output
				// and the independently retained projection must stay valid.
				y, err := views[e].Forward(x)
				if err != nil {
					t.Fatal(err)
				}
				if err := views[e].Close(); err != nil {
					y.Close()
					t.Fatal(err)
				}
				z, err := retained[e].Forward(x)
				if err != nil {
					y.Close()
					t.Fatal(err)
				}
				for _, a := range []mlx.Array{y, z} {
					got, err := a.Float32Data()
					a.Close()
					if err != nil {
						t.Fatal(err)
					}
					for n := range tokens {
						for r := range rows {
							var want, l1 float64
							for k := range cols {
								v := float64(input[n*cols+k]) * float64(decoded[e][r*cols+k])
								want += v
								l1 += math.Abs(v)
							}
							v := float64(got[n*rows+r])
							if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v-want) > 1e-5+2e-6*l1 {
								t.Fatalf("entry=%d token=%d row=%d got=%g want=%g", e, n, r, v, want)
							}
						}
					}
				}
			}
		}()
		if after := mlx.LiveArrays(); after != beforeHandles {
			t.Fatalf("leaked handles: %d -> %d", beforeHandles, after)
		}
	})
}

func TestFP4LinearViewValidation(t *testing.T) {
	linearDevices(t, func(t *testing.T) {
		before := mlx.LiveArrays()
		func() {
			var nilBank *FP4LinearBank
			if l, err := nilBank.Linear(0); err == nil {
				l.Close()
				t.Fatal("nil bank accepted")
			}
			var nilLinear *FP4Linear
			if l, err := nilLinear.Retain(); err == nil {
				l.Close()
				t.Fatal("nil linear accepted")
			}
			data, scales := make([]byte, 32*32/2), make([]byte, 32)
			for i := range data {
				data[i] = byte(i)
			}
			for i := range scales {
				scales[i] = 125
			}
			l, err := NewFP4Linear(data, scales, 32, 32)
			if err != nil {
				t.Fatal(err)
			}
			defer l.Close()
			bank, err := StackFP4Linears([]*FP4Linear{l})
			if err != nil {
				t.Fatal(err)
			}
			defer bank.Close()
			for _, i := range []int{-1, 1} {
				if v, err := bank.Linear(i); err == nil {
					v.Close()
					t.Fatalf("index %d accepted", i)
				}
			}
			// Failure after retaining the first array must release its handle.
			l.scales.Close()
			if v, err := l.Retain(); err == nil {
				v.Close()
				t.Fatal("closed scales accepted by Retain")
			}
			bank.scales.Close()
			if v, err := bank.Linear(0); err == nil {
				v.Close()
				t.Fatal("closed scales accepted by Linear")
			}
			l.Close()
			if v, err := l.Retain(); err == nil {
				v.Close()
				t.Fatal("closed linear accepted")
			}
			bank.Close()
			if v, err := bank.Linear(0); err == nil {
				v.Close()
				t.Fatal("closed bank accepted")
			}
		}()
		if after := mlx.LiveArrays(); after != before {
			t.Fatalf("validation leaked handles: %d -> %d", before, after)
		}
	})
}
