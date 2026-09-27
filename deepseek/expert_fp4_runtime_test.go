//go:build mlx && mlxruntime

package deepseek

import (
	"fmt"
	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/deepseek/quant"
	"math"
	"testing"
)

func TestFP4ExpertOwnershipAndValidation(t *testing.T) {
	defer mlx.SetDefaultCPU()
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			parts := make([]FP4Weight, 3)
			arrays := make([]mlx.Array, 3)
			defer func() { mlx.CloseArrays(arrays) }()
			for k := range parts {
				parts[k] = FP4Weight{Data: make([]byte, 512), Scales: make([]byte, 32)}
				for i := range parts[k].Data {
					parts[k].Data[i] = []byte{0x22, 0x66, 0x11}[k]
				}
				for i := range parts[k].Scales {
					parts[k].Scales[i] = 127
				}
				d := make([]float32, 1024)
				if err := quant.Decode(d, parts[k].Data, parts[k].Scales, 32, 32, quant.FP4Row32, quant.Float32); err != nil {
					t.Fatal(err)
				}
				var err error
				arrays[k], err = mlx.NewFloat32(d, []int{32, 32})
				if err != nil {
					t.Fatal(err)
				}
			}
			p, err := NewFP4Expert(FP4ExpertWeights{Gate: parts[0], Up: parts[1], Down: parts[2]}, 32, 32)
			if err != nil {
				t.Fatal(err)
			}
			defer p.Close()
			for _, w := range parts {
				clear(w.Data)
				clear(w.Scales)
			}
			values := make([]float32, 96)
			for i := 0; i < 32; i++ {
				values[i] = 1
				values[32+i] = -1
			}
			x, err := mlx.NewFloat32(values, []int{3, 32})
			if err != nil {
				t.Fatal(err)
			}
			defer x.Close()
			r, err := mlx.NewFloat32([]float32{.25, 1, 0}, []int{3, 1})
			if err != nil {
				t.Fatal(err)
			}
			defer r.Close()
			for _, limit := range []float32{0, 10} {
				y, err := p.Forward(x, r, limit)
				if err != nil {
					t.Fatal(err)
				}
				defer y.Close()
				want, err := Expert(x, ExpertWeights{Gate: arrays[0], Up: arrays[1], Down: arrays[2]}, r, limit)
				if err != nil {
					t.Fatal(err)
				}
				defer want.Close()
				got, e := y.Float32Data()
				if e != nil {
					t.Fatal(e)
				}
				ref, e := want.Float32Data()
				if e != nil {
					t.Fatal(e)
				}
				compareAttention(t, "FP4 clipping", got, ref, 1e-6)
			}
			for _, limit := range []float32{-1, float32(math.NaN()), float32(math.Inf(1))} {
				if y, err := p.Forward(x, r, limit); err == nil {
					y.Close()
					t.Fatal("invalid limit accepted")
				}
			}
			for _, args := range [][2]mlx.Array{{mlx.Array{}, r}, {x, x}, {x, mlx.Array{}}} {
				if y, err := p.Forward(args[0], args[1], 10); err == nil {
					y.Close()
					t.Fatal("invalid input accepted")
				}
			}
			compiled, err := mlx.Compile(func(in []mlx.Array) ([]mlx.Array, error) {
				y, err := p.Forward(in[0], in[1], 10)
				if err != nil {
					return nil, err
				}
				return []mlx.Array{y}, nil
			}, false)
			if err != nil {
				t.Fatal(err)
			}
			defer compiled.Close()
			y, err := compiled.Apply(x, r)
			if err != nil {
				t.Fatal(err)
			}
			if err := mlx.Eval(y...); err != nil {
				t.Fatal(err)
			}
			mlx.CloseArrays(y)
			compiled.Close()
			lazy, err := p.Forward(x, r, 10)
			if err != nil {
				t.Fatal(err)
			}
			defer lazy.Close()
			copyOf := *p
			p.Close()
			if err := mlx.Eval(lazy); err != nil {
				t.Fatal("lazy dependencies lost", err)
			}
			if y, err := copyOf.Forward(x, r, 10); err == nil {
				y.Close()
				t.Fatal("closed copy accepted")
			}
			var nilExpert *FP4Expert
			if y, err := nilExpert.Forward(x, r, 10); err == nil {
				y.Close()
				t.Fatal("nil expert accepted")
			}
		})
	}
}

func BenchmarkReleasedFP4Expert(b *testing.B) {
	dir, ref := readExpertReference(b)
	defer mlx.SetDefaultCPU()
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		b.Run(device.name, func(b *testing.B) {
			if err := device.set(); err != nil {
				b.Fatal(err)
			}
			for _, tokens := range []int{1, 128} {
				b.Run(fmt.Sprintf("tokens%d", tokens), func(b *testing.B) {
					for _, packed := range []bool{false, true} {
						name := "float32"
						if packed {
							name = "fp4"
						}
						b.Run(name, func(b *testing.B) {
							before, err := mlx.GetMemoryUsage()
							if err != nil {
								b.Fatal(err)
							}
							var run func(mlx.Array, mlx.Array, float32) (mlx.Array, error)
							if packed {
								p, err := NewFP4Expert(FP4ExpertWeights{Gate: loadPackedExpertWeight(b, dir, ref.Matrices[0]), Down: loadPackedExpertWeight(b, dir, ref.Matrices[1]), Up: loadPackedExpertWeight(b, dir, ref.Matrices[2])}, ref.Dim, ref.Inter)
								if err != nil {
									b.Fatal(err)
								}
								defer p.Close()
								run = p.Forward
							} else {
								var w [3]mlx.Array
								defer func() { mlx.CloseArrays(w[:]) }()
								for i, m := range ref.Matrices {
									w[i], err = mlx.NewFloat32(loadExpertMatrix(b, dir, m), []int{m.Rows, m.Cols})
									if err != nil {
										b.Fatal(err)
									}
								}
								run = func(x, r mlx.Array, limit float32) (mlx.Array, error) {
									return Expert(x, ExpertWeights{Gate: w[0], Down: w[1], Up: w[2]}, r, limit)
								}
							}
							base := expertInputs(ref.Dim)
							values := make([]float32, tokens*ref.Dim)
							for n := 0; n < tokens; n++ {
								copy(values[n*ref.Dim:], base[(n%6)*ref.Dim:(n%6+1)*ref.Dim])
							}
							x, err := mlx.NewFloat32(values, []int{tokens, ref.Dim})
							if err != nil {
								b.Fatal(err)
							}
							defer x.Close()
							routing := make([]float32, tokens)
							for i := range routing {
								routing[i] = .75
							}
							r, err := mlx.NewFloat32(routing, []int{tokens, 1})
							if err != nil {
								b.Fatal(err)
							}
							defer r.Close()
							step := func() error {
								y, err := run(x, r, 10)
								if err != nil {
									return err
								}
								defer y.Close()
								return mlx.Eval(y)
							}
							for i := 0; i < 3; i++ {
								if err := mlx.Batch(step); err != nil {
									b.Fatal(err)
								}
							}
							steady, err := mlx.GetMemoryUsage()
							if err != nil {
								b.Fatal(err)
							}
							if err := mlx.ResetPeakMemory(); err != nil {
								b.Fatal(err)
							}
							b.ResetTimer()
							for i := 0; i < b.N; i++ {
								if err := mlx.Batch(step); err != nil {
									b.Fatal(err)
								}
							}
							b.StopTimer()
							usage, err := mlx.GetMemoryUsage()
							if err != nil {
								b.Fatal(err)
							}
							if usage.PeakBytes < before.ActiveBytes || steady.ActiveBytes < before.ActiveBytes {
								b.Fatal("allocator counters")
							}
							b.ReportMetric(float64(usage.PeakBytes-before.ActiveBytes), "peak-active-bytes")
							b.ReportMetric(float64(steady.ActiveBytes-before.ActiveBytes), "steady-active-bytes")
							bytes := 3 * ref.Dim * ref.Inter * 4
							if packed {
								bytes = 3 * ref.Dim * ref.Inter * (16 + 1) / 32
							}
							b.ReportMetric(float64(bytes), "weight-bytes")
						})
					}
				})
			}
		})
	}
}
