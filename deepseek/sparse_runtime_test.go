//go:build mlx && mlxruntime

package deepseek

import (
	"fmt"
	"math"
	"testing"

	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/deepseek/quant"
)

func sparseFixture(t testing.TB, dim, inter, count int, storage string) (expertRunner, []ExpertWeights) {
	t.Helper()
	weights := make([]ExpertWeights, count+1)
	packed := make(map[int]*FP4Expert)
	for id := -1; id < count; id++ {
		var parts [3]FP4Weight
		var arrays [3]mlx.Array
		for i, shape := range [][]int{{inter, dim}, {inter, dim}, {dim, inter}} {
			n := shape[0] * shape[1]
			parts[i] = FP4Weight{Data: make([]byte, n/2), Scales: make([]byte, n/32)}
			for j := range parts[i].Data {
				parts[i].Data[j] = byte(j*37 + i*29 + (id+2)*7)
			}
			for j := range parts[i].Scales {
				parts[i].Scales[j] = 120
			}
			decoded := make([]float32, n)
			if err := quant.Decode(decoded, parts[i].Data, parts[i].Scales, shape[0], shape[1], quant.FP4Row32, quant.Float32); err != nil {
				t.Fatal(err)
			}
			a, err := mlx.NewFloat32(decoded, shape)
			if err != nil {
				t.Fatal(err)
			}
			arrays[i] = a
			t.Cleanup(func() { a.Close() })
		}
		weights[id+1] = ExpertWeights{Gate: arrays[0], Up: arrays[1], Down: arrays[2]}
		if storage == "fp4" || (storage == "mixed" && id%2 == 0) {
			p, err := NewFP4Expert(FP4ExpertWeights{Gate: parts[0], Up: parts[1], Down: parts[2]}, dim, inter)
			if err != nil {
				t.Fatal(err)
			}
			packed[id] = p
			t.Cleanup(func() { p.Close() })
		}
	}
	return func(s *scope, id int, x, amplitude mlx.Array) mlx.Array {
		if p := packed[id]; p != nil {
			return s.add(p.Forward(x, amplitude, 1))
		}
		return expert(s, x, weights[id+1], amplitude, 1)
	}, weights
}

func denseAssigned(s *scope, x, w, ids mlx.Array, count int, run expertRunner) mlx.Array {
	out := run(s, -1, x, s.add(mlx.Ones([]int{x.Shape()[0], 1}, mlx.Float32)))
	for id := 0; id < count; id++ {
		mask := s.add(mlx.Equal(ids, s.add(mlx.NewScalarInt(id))))
		amplitude := s.add(mlx.SumAxis(s.add(mlx.Where(mask, w, s.scalar(0))), -1, true))
		out = s.add(mlx.Add(out, run(s, id, x, amplitude)))
	}
	return out
}

func compareSparse(t testing.TB, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatal("output length")
	}
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Abs(float64(v-want[i])) > tolerance*(1+math.Abs(float64(want[i]))) {
			t.Fatalf("output %d: %g != %g", i, v, want[i])
		}
	}
}

func TestSparseExperts(t *testing.T) {
	defer mlx.SetDefaultCPU()
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			for _, storage := range []string{"float32", "mixed", "fp4"} {
				t.Run(storage, func(t *testing.T) {
					run, weights := sparseFixture(t, 32, 64, 5, storage)
					xv := make([]float32, 4*32)
					for i := range xv {
						xv[i] = float32(i%19-9) / 8
					}
					x, err := mlx.NewFloat32(xv, []int{4, 32})
					if err != nil {
						t.Fatal(err)
					}
					defer x.Close()
					ids, err := mlx.NewInt32([]int32{3, 1, 0, 3, 1, 0, 3, 0}, []int{4, 2})
					if err != nil {
						t.Fatal(err)
					}
					defer ids.Close()
					w, err := mlx.NewFloat32([]float32{.7, .3, 0, .8, .4, .6, 0, 0}, []int{4, 2})
					if err != nil {
						t.Fatal(err)
					}
					defer w.Close()
					var calls []int
					out, err := one(func(s *scope) mlx.Array {
						return sparseExperts(s, x, w, ids, 5, func(s *scope, id int, a, r mlx.Array) mlx.Array {
							calls = append(calls, id)
							if id >= 0 && a.Shape()[0] == 4 {
								t.Error("full token batch sent to routed expert")
							}
							return run(s, id, a, r)
						})
					})
					if err != nil {
						t.Fatal(err)
					}
					defer out.Close()
					got, err := out.Float32Data()
					if err != nil {
						t.Fatal(err)
					}
					if fmt.Sprint(calls) != "[0 1 3 -1]" {
						t.Fatal("unused experts executed", calls)
					}
					ref, err := one(func(s *scope) mlx.Array { return denseAssigned(s, x, w, ids, 5, run) })
					if err != nil {
						t.Fatal(err)
					}
					defer ref.Close()
					want, err := ref.Float32Data()
					if err != nil {
						t.Fatal(err)
					}
					compareSparse(t, got, want, 2e-5)
					// Public routing path: ties, every expert selected, all score types.
					gw, err := mlx.Zeros([]int{5, 32}, mlx.Float32)
					if err != nil {
						t.Fatal(err)
					}
					defer gw.Close()
					gb, err := mlx.Zeros([]int{5}, mlx.Float32)
					if err != nil {
						t.Fatal(err)
					}
					defer gb.Close()
					for _, score := range []string{"softmax", "sigmoid", "sqrtsoftplus"} {
						for _, k := range []int{1, 2, 5} {
							c := RouterConfig{TopK: k, Score: score, Temperature: 1, Scale: 1.7, Normalize: true}
							a, err := SparseMoE(x, gw, gb, weights[1:], weights[0], c, 1)
							if err != nil {
								t.Fatal(err)
							}
							b, err := MoE(x, gw, gb, weights[1:], weights[0], c, 1)
							if err != nil {
								a.Close()
								t.Fatal(err)
							}
							av, ae := a.Float32Data()
							bv, be := b.Float32Data()
							a.Close()
							b.Close()
							if ae != nil || be != nil {
								t.Fatal(ae, be)
							}
							compareSparse(t, av, bv, 2e-5)
						}
					}
				})
			}
		})
	}
}

func BenchmarkSparseDispatch(b *testing.B) {
	defer mlx.SetDefaultCPU()
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		b.Run(device.name, func(b *testing.B) {
			if err := device.set(); err != nil {
				b.Fatal(err)
			}
			for _, storage := range []string{"float32", "mixed", "fp4"} {
				b.Run(storage, func(b *testing.B) {
					run, _ := sparseFixture(b, 128, 128, 16, storage)
					for _, tokens := range []int{1, 128} {
						b.Run(fmt.Sprintf("tokens%d", tokens), func(b *testing.B) {
							xv := make([]float32, tokens*128)
							for i := range xv {
								xv[i] = float32(i%19-9) / 8
							}
							x, err := mlx.NewFloat32(xv, []int{tokens, 128})
							if err != nil {
								b.Fatal(err)
							}
							defer x.Close()
							gv := make([]float32, 16*128)
							for e := 0; e < 16; e++ {
								for d := 0; d < 128; d++ {
									gv[e*128+d] = float32(math.Sin(float64((e+1)*(d+1)))) / 16
								}
							}
							gw, err := mlx.NewFloat32(gv, []int{16, 128})
							if err != nil {
								b.Fatal(err)
							}
							defer gw.Close()
							gb, err := mlx.Zeros([]int{16}, mlx.Float32)
							if err != nil {
								b.Fatal(err)
							}
							defer gb.Close()
							config := RouterConfig{TopK: 2, Score: "sigmoid", Temperature: 1, Scale: 1, Normalize: true}
							probeW, probeIDs, err := Route(x, gw, gb, config)
							if err != nil {
								b.Fatal(err)
							}
							cast, err := mlx.AsType(probeIDs, mlx.Int32)
							if err != nil {
								b.Fatal(err)
							}
							values, err := cast.Int32Data()
							cast.Close()
							probeW.Close()
							probeIDs.Close()
							if err != nil {
								b.Fatal(err)
							}
							active := map[int32]bool{}
							for _, id := range values {
								active[id] = true
							}
							for _, sparse := range []bool{false, true} {
								b.Run(fmt.Sprintf("sparse_%t", sparse), func(b *testing.B) {
									step := func() error {
										y, err := one(func(s *scope) mlx.Array {
											w, ids := route(s, x, gw, gb, config)
											if s.err != nil {
												return mlx.Array{}
											}
											if sparse {
												return sparseExperts(s, x, w, ids, 16, run)
											}
											return denseAssigned(s, x, w, ids, 16, run)
										})
										if err != nil {
											return err
										}
										defer y.Close()
										return mlx.Eval(y)
									}
									for range 3 {
										if err := mlx.Batch(step); err != nil {
											b.Fatal(err)
										}
									}
									if err := mlx.ResetPeakMemory(); err != nil {
										b.Fatal(err)
									}
									b.ResetTimer()
									for range b.N {
										if err := mlx.Batch(step); err != nil {
											b.Fatal(err)
										}
									}
									b.StopTimer()
									after, err := mlx.GetMemoryUsage()
									if err != nil {
										b.Fatal(err)
									}
									b.ReportMetric(float64(after.PeakBytes), "peak-active-bytes")
									b.ReportMetric(float64(len(active)), "active-experts")
								})
							}
						})
					}
				})
			}
		})
	}
}
