//go:build mlx && mlxruntime

package deepseek

import (
	"fmt"
	"math"
	"sync"
	"testing"

	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/deepseek/quant"
)

// These weights have different signs, scales, rows and columns for every
// expert, including the shared expert at index zero. Both implementations see
// exactly the same decoded values; no quantization error enters comparisons.
func deviceMoEFixture(dim, inter, experts int) []FP4ExpertWeights {
	all := make([]FP4ExpertWeights, experts+1)
	for e := range all {
		parts := [3]FP4Weight{}
		for p := range parts {
			n := dim * inter
			parts[p] = FP4Weight{Data: make([]byte, n/2), Scales: make([]byte, n/32)}
			state := uint32(197 + e*7919 + p*104729)
			for j := range parts[p].Data {
				state = state*1664525 + 1013904223
				parts[p].Data[j] = byte(state >> 24)
			}
			for j := range parts[p].Scales {
				parts[p].Scales[j] = byte(120 + (j+e+p)%3)
			}
		}
		all[e] = FP4ExpertWeights{Gate: parts[0], Up: parts[1], Down: parts[2]}
	}
	return all
}

func deviceMoEDecoded(t testing.TB, raw []FP4ExpertWeights, dim, inter int) []ExpertWeights {
	t.Helper()
	all := make([]ExpertWeights, len(raw))
	for e, w := range raw {
		var arrays [3]mlx.Array
		for p, part := range []FP4Weight{w.Gate, w.Up, w.Down} {
			rows, cols := inter, dim
			if p == 2 {
				rows, cols = dim, inter
			}
			values := make([]float32, rows*cols)
			if err := quant.Decode(values, part.Data, part.Scales, rows, cols, quant.FP4Row32, quant.Float32); err != nil {
				t.Fatal(err)
			}
			a, err := mlx.NewFloat32(values, []int{rows, cols})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { a.Close() })
			arrays[p] = a
		}
		all[e] = ExpertWeights{Gate: arrays[0], Up: arrays[1], Down: arrays[2]}
	}
	return all
}

func deviceMoECloseWeights(all []ExpertWeights) {
	for _, w := range all {
		mlx.CloseArrays([]mlx.Array{w.Gate, w.Up, w.Down})
	}
}

func deviceMoEArray(t testing.TB, values []float32, shape ...int) mlx.Array {
	t.Helper()
	a, err := mlx.NewFloat32(values, shape)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { a.Close() })
	return a
}

func deviceMoEInputs(t testing.TB, tokens, dim, experts int, ties bool) (mlx.Array, mlx.Array, mlx.Array) {
	t.Helper()
	values := make([]float32, tokens*dim)
	for i := range values {
		values[i] = float32(math.Sin(float64(i+1)*1.7)+math.Cos(float64(i+3)*.37)) * 1.5
	}
	gate := make([]float32, experts*dim)
	bias := make([]float32, experts)
	if !ties {
		for e := range experts {
			gate[e*dim+e%dim] = 1.3
			gate[e*dim+(e*7+3)%dim] -= .7
			bias[e] = float32(e%3-1) * .13
		}
	}
	return deviceMoEArray(t, values, tokens, dim), deviceMoEArray(t, gate, experts, dim), deviceMoEArray(t, bias, experts)
}

func deviceMoEData(t testing.TB, a mlx.Array, err error) []float32 {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	values, err := a.Float32Data()
	if err != nil {
		t.Fatal(err)
	}
	return values
}

func deviceMoEPackedRunner(t testing.TB, raw []FP4ExpertWeights, dim, inter int, limit float32) expertRunner {
	t.Helper()
	all := make([]*FP4Expert, len(raw))
	for i, w := range raw {
		p, err := NewFP4Expert(w, dim, inter)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { p.Close() })
		all[i] = p
	}
	return func(s *scope, id int, x, amplitude mlx.Array) mlx.Array {
		return s.add(all[id+1].Forward(x, amplitude, limit))
	}
}

func deviceMoEPackedReference(x, gate, bias mlx.Array, c RouterConfig, experts int, run expertRunner, sparse bool) (mlx.Array, error) {
	return one(func(s *scope) mlx.Array {
		weights, ids := route(s, x, gate, bias, c)
		if s.err != nil {
			return mlx.Array{}
		}
		if sparse {
			return sparseExperts(s, x, weights, ids, experts, run)
		}
		return denseAssigned(s, x, weights, ids, experts, run)
	})
}

func TestDeviceMoEParity(t *testing.T) {
	defer mlx.SetDefaultCPU()
	type parityCase struct {
		name   string
		tokens int
		ties   bool
		config RouterConfig
		limit  float32
	}
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			shapes := [][3]int{{32, 64, 5}, {64, 32, 5}}
			if device.name == "gpu" {
				shapes = append(shapes, [3]int{128, 128, 16}, [3]int{512, 1024, 16})
			}
			for _, shape := range shapes {
				t.Run(fmt.Sprintf("dim%d_inter%d", shape[0], shape[1]), func(t *testing.T) {
					dim, inter, count := shape[0], shape[1], shape[2]
					raw := deviceMoEFixture(dim, inter, count)
					weights := deviceMoEDecoded(t, raw, dim, inter)
					for _, storage := range []string{"float32", "fp4"} {
						t.Run(storage, func(t *testing.T) {
							var bank *DeviceMoE
							var err error
							if storage == "fp4" {
								bank, err = NewFP4DeviceMoE(raw[1:], raw[0], dim, inter)
							} else {
								bank, err = NewDeviceMoE(weights[1:], weights[0])
							}
							if err != nil {
								t.Fatal(err)
							}
							t.Cleanup(func() { bank.Close() })
							cases := []parityCase{
								{"decode", 1, false, RouterConfig{1, "sigmoid", .7, 1.7, true}, 0},
								{"prefill_clipped", 7, false, RouterConfig{2, "sqrtsoftplus", 1.4, .6, true}, .35},
								{"all_unnormalized", 3, false, RouterConfig{count, "softmax", .8, 1.3, false}, 1},
								{"partial_unnormalized", 5, false, RouterConfig{2, "sigmoid", .6, 1.8, false}, .5},
							}
							if count == 16 {
								// Match the measured shapes, assignments and clipping
								// exactly, without repeating the small exhaustive matrix.
								cases = nil
								for _, tokens := range []int{1, 128} {
									for _, k := range []int{2, 8} {
										cases = append(cases, parityCase{fmt.Sprintf("tokens%d_k%d", tokens, k), tokens, false, RouterConfig{k, "sigmoid", 1, 1, true}, 1})
									}
								}
							} else {
								for _, score := range []string{"softmax", "sigmoid", "sqrtsoftplus"} {
									for _, k := range []int{1, 2, count} {
										cases = append(cases, parityCase{fmt.Sprintf("ties_%s_k%d", score, k), 3, true, RouterConfig{k, score, 1.2, 1.7, true}, .6})
									}
								}
							}
							for _, tc := range cases {
								t.Run(tc.name, func(t *testing.T) {
									x, gate, bias := deviceMoEInputs(t, tc.tokens, dim, count, tc.ties)
									y, err := bank.Forward(x, gate, bias, tc.config, tc.limit)
									got := deviceMoEData(t, y, err)
									y, err = MoE(x, gate, bias, weights[1:], weights[0], tc.config, tc.limit)
									want := deviceMoEData(t, y, err)
									compareSparse(t, got, want, 2e-5)
									if storage == "fp4" {
										run := deviceMoEPackedRunner(t, raw, dim, inter, tc.limit)
										for _, sparse := range []bool{false, true} {
											y, err = deviceMoEPackedReference(x, gate, bias, tc.config, count, run, sparse)
											compareSparse(t, got, deviceMoEData(t, y, err), 2e-5)
										}
									} else {
										y, err = SparseMoE(x, gate, bias, weights[1:], weights[0], tc.config, tc.limit)
										compareSparse(t, got, deviceMoEData(t, y, err), 2e-5)
									}
								})
							}
						})
					}
				})
			}
		})
	}
}

func TestDeviceMoEOwnershipCompileAndConcurrency(t *testing.T) {
	defer mlx.SetDefaultCPU()
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			for _, packed := range []bool{false, true} {
				t.Run(fmt.Sprintf("packed_%t", packed), func(t *testing.T) {
					const dim, inter, count = 32, 64, 5
					raw := deviceMoEFixture(dim, inter, count)
					weights := deviceMoEDecoded(t, raw, dim, inter)
					x, gate, bias := deviceMoEInputs(t, 4, dim, count, false)
					other, err := mlx.Negative(x)
					if err != nil {
						t.Fatal(err)
					}
					defer other.Close()
					c := RouterConfig{TopK: 2, Score: "sigmoid", Temperature: .8, Scale: 1.4, Normalize: true}
					inputs := []mlx.Array{x, other}
					wants := make([][]float32, len(inputs))
					for i, input := range inputs {
						y, err := MoE(input, gate, bias, weights[1:], weights[0], c, .4)
						wants[i] = deviceMoEData(t, y, err)
					}
					var bank *DeviceMoE
					if packed {
						bank, err = NewFP4DeviceMoE(raw[1:], raw[0], dim, inter)
					} else {
						bank, err = NewDeviceMoE(weights[1:], weights[0])
					}
					if err != nil {
						t.Fatal(err)
					}
					defer bank.Close()
					deviceMoECloseWeights(weights)
					for _, w := range raw {
						for _, p := range []FP4Weight{w.Gate, w.Up, w.Down} {
							clear(p.Data)
							clear(p.Scales)
						}
					}
					// Compilation rejects host reads of traced routing indices. A
					// second input also detects accidentally frozen first-call routing.
					compiled, err := mlx.Compile(func(in []mlx.Array) ([]mlx.Array, error) {
						y, err := bank.Forward(in[0], in[1], in[2], c, .4)
						if err != nil {
							return nil, err
						}
						return []mlx.Array{y}, nil
					}, false)
					if err != nil {
						t.Fatal(err)
					}
					defer compiled.Close()
					for i, input := range inputs {
						outputs, err := compiled.Apply(input, gate, bias)
						if err != nil {
							t.Fatal(err)
						}
						if len(outputs) != 1 {
							mlx.CloseArrays(outputs)
							t.Fatal("compiled output count")
						}
						compareSparse(t, deviceMoEData(t, outputs[0], nil), wants[i], 2e-5)
					}
					if err := compiled.Close(); err != nil {
						t.Fatal(err)
					}
					errs := make(chan error, 4)
					var wg sync.WaitGroup
					for worker := range 4 {
						wg.Add(1)
						go func() {
							defer wg.Done()
							for round := range 3 {
								i := (worker + round) % len(inputs)
								y, err := bank.Forward(inputs[i], gate, bias, c, .4)
								if err != nil {
									errs <- err
									return
								}
								got, err := y.Float32Data()
								y.Close()
								if err != nil {
									errs <- err
									return
								}
								for j, v := range got {
									want := wants[i][j]
									if math.IsNaN(float64(v)) || math.Abs(float64(v-want)) > 2e-5*(1+math.Abs(float64(want))) {
										errs <- fmt.Errorf("concurrent output %d: %g != %g", j, v, want)
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
					lazy, err := bank.Forward(x, gate, bias, c, .4)
					if err != nil {
						t.Fatal(err)
					}
					copyOf := *bank
					if err := bank.Close(); err != nil {
						t.Fatal(err)
					}
					compareSparse(t, deviceMoEData(t, lazy, nil), wants[0], 2e-5)
					if y, err := copyOf.Forward(x, gate, bias, c, .4); err == nil {
						y.Close()
						t.Fatal("closed copy accepted")
					}
					if err := copyOf.Close(); err != nil {
						t.Fatal("close must be idempotent", err)
					}
					if err := mlx.Eval(x, gate, bias); err != nil {
						t.Fatal("bank closed borrowed inputs", err)
					}
				})
			}
		})
	}
}

func TestDeviceMoESharedExpert(t *testing.T) {
	defer mlx.SetDefaultCPU()
	if err := mlx.SetDefaultGPU(); err != nil {
		t.Fatal(err)
	}
	const dim, inter, count = 32, 64, 3
	raw := deviceMoEFixture(dim, inter, count)
	for _, w := range raw[1:] {
		for _, p := range []FP4Weight{w.Gate, w.Up, w.Down} {
			clear(p.Data)
		}
	}
	weights := deviceMoEDecoded(t, raw, dim, inter)
	x, gate, bias := deviceMoEInputs(t, 3, dim, count, false)
	amplitudes := deviceMoEArray(t, []float32{1, 1, 1}, 3, 1)
	y, err := Expert(x, weights[0], amplitudes, .4)
	want := deviceMoEData(t, y, err)
	var maxAbs float64
	for _, v := range want {
		maxAbs = math.Max(maxAbs, math.Abs(float64(v)))
	}
	if maxAbs < 1e-3 {
		t.Fatal("shared fixture must have a visible nonzero contribution")
	}
	for _, packed := range []bool{false, true} {
		var bank *DeviceMoE
		if packed {
			bank, err = NewFP4DeviceMoE(raw[1:], raw[0], dim, inter)
		} else {
			bank, err = NewDeviceMoE(weights[1:], weights[0])
		}
		if err != nil {
			t.Fatal(err)
		}
		defer bank.Close()
		for _, k := range []int{1, count} {
			c := RouterConfig{TopK: k, Score: "sigmoid", Temperature: 1, Scale: 3, Normalize: true}
			y, err := bank.Forward(x, gate, bias, c, .4)
			compareSparse(t, deviceMoEData(t, y, err), want, 2e-5)
		}
	}
}

func TestDeviceMoEValidation(t *testing.T) {
	defer mlx.SetDefaultCPU()
	if err := mlx.SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}
	const dim, inter, count = 32, 64, 3
	raw := deviceMoEFixture(dim, inter, count)
	weights := deviceMoEDecoded(t, raw, dim, inter)
	badDType, err := mlx.AsType(weights[1].Gate, mlx.Float16)
	if err != nil {
		t.Fatal(err)
	}
	defer badDType.Close()
	closed, err := mlx.Reshape(weights[1].Gate, weights[1].Gate.Shape())
	if err != nil {
		t.Fatal(err)
	}
	closed.Close()
	for _, tc := range []struct {
		name    string
		experts []ExpertWeights
		shared  ExpertWeights
	}{
		{"empty", nil, weights[0]},
		{"missing_gate", []ExpertWeights{{Up: weights[1].Up, Down: weights[1].Down}}, weights[0]},
		{"closed_gate", []ExpertWeights{{Gate: closed, Up: weights[1].Up, Down: weights[1].Down}}, weights[0]},
		{"wrong_dtype", []ExpertWeights{{Gate: badDType, Up: weights[1].Up, Down: weights[1].Down}}, weights[0]},
		{"wrong_up", []ExpertWeights{{Gate: weights[1].Gate, Up: weights[1].Down, Down: weights[1].Down}}, weights[0]},
		{"wrong_down", []ExpertWeights{{Gate: weights[1].Gate, Up: weights[1].Up, Down: weights[1].Gate}}, weights[0]},
		{"different_expert_shape", []ExpertWeights{weights[1], {Gate: weights[2].Down, Up: weights[2].Down, Down: weights[2].Gate}}, weights[0]},
		{"missing_shared", weights[1:], ExpertWeights{}},
	} {
		t.Run("float32/"+tc.name, func(t *testing.T) {
			before := mlx.LiveArrays()
			if bank, err := NewDeviceMoE(tc.experts, tc.shared); err == nil {
				bank.Close()
				t.Fatal("invalid constructor accepted")
			}
			if after := mlx.LiveArrays(); after != before {
				t.Fatalf("failed constructor leaked owned handles: before=%d after=%d", before, after)
			}
		})
	}
	badGate, badUp, badDown, badScale := raw[1], raw[1], raw[1], raw[1]
	badGate.Gate.Data = badGate.Gate.Data[:len(badGate.Gate.Data)-1]
	badUp.Up.Data = nil
	badDown.Down.Scales = nil
	badScale.Gate.Scales = append([]byte(nil), badScale.Gate.Scales...)
	badScale.Gate.Scales[0] = 255
	for _, tc := range []struct {
		name       string
		experts    []FP4ExpertWeights
		shared     FP4ExpertWeights
		dim, inter int
	}{
		{"empty", nil, raw[0], dim, inter},
		{"zero_dim", raw[1:], raw[0], 0, inter},
		{"unaligned_dim", raw[1:], raw[0], 31, inter},
		{"unaligned_inter", raw[1:], raw[0], dim, 63},
		{"bad_gate", []FP4ExpertWeights{badGate}, raw[0], dim, inter},
		{"bad_up", []FP4ExpertWeights{badUp}, raw[0], dim, inter},
		{"bad_down", []FP4ExpertWeights{badDown}, raw[0], dim, inter},
		{"nonfinite_scale", []FP4ExpertWeights{badScale}, raw[0], dim, inter},
		{"bad_shared", raw[1:], badDown, dim, inter},
	} {
		t.Run("fp4/"+tc.name, func(t *testing.T) {
			before := mlx.LiveArrays()
			if bank, err := NewFP4DeviceMoE(tc.experts, tc.shared, tc.dim, tc.inter); err == nil {
				bank.Close()
				t.Fatal("invalid constructor accepted")
			}
			if after := mlx.LiveArrays(); after != before {
				t.Fatalf("failed constructor leaked owned handles: before=%d after=%d", before, after)
			}
		})
	}
	x, gate, bias := deviceMoEInputs(t, 3, dim, count, false)
	c := RouterConfig{TopK: 2, Score: "sigmoid", Temperature: 1, Scale: 1, Normalize: true}
	wrongInput := deviceMoEArray(t, make([]float32, dim), 1, 1, dim)
	for _, packed := range []bool{false, true} {
		var bank *DeviceMoE
		if packed {
			bank, err = NewFP4DeviceMoE(raw[1:], raw[0], dim, inter)
		} else {
			bank, err = NewDeviceMoE(weights[1:], weights[0])
		}
		if err != nil {
			t.Fatal(err)
		}
		defer bank.Close()
		for _, limit := range []float32{-1, float32(math.NaN()), float32(math.Inf(1))} {
			if y, err := bank.Forward(x, gate, bias, c, limit); err == nil {
				y.Close()
				t.Fatal("invalid limit accepted")
			}
		}
		for _, args := range [][3]mlx.Array{{mlx.Array{}, gate, bias}, {wrongInput, gate, bias}, {x, weights[1].Gate, bias}, {x, gate, x}, {x, gate, mlx.Array{}}} {
			if y, err := bank.Forward(args[0], args[1], args[2], c, 1); err == nil {
				y.Close()
				t.Fatal("invalid input accepted")
			}
		}
		for _, config := range []RouterConfig{{0, "sigmoid", 1, 1, true}, {count + 1, "sigmoid", 1, 1, true}, {2, "unknown", 1, 1, true}, {2, "sigmoid", 0, 1, true}, {2, "sigmoid", 1, -1, true}} {
			if y, err := bank.Forward(x, gate, bias, config, 1); err == nil {
				y.Close()
				t.Fatal("invalid router accepted")
			}
		}
	}
	var nilBank *DeviceMoE
	if y, err := nilBank.Forward(x, gate, bias, c, 1); err == nil {
		y.Close()
		t.Fatal("nil bank accepted")
	}
	if err := nilBank.Close(); err != nil {
		t.Fatal(err)
	}
}

// Each leaf owns only the implementation under measurement. In particular,
// packed leaves never retain decoded reference matrices, and float32 device
// leaves close their source arrays before measuring the stacked bank.
func BenchmarkDeviceMoE(b *testing.B) {
	defer mlx.SetDefaultCPU()
	if err := mlx.SetDefaultGPU(); err != nil {
		b.Fatal(err)
	}
	const count = 16
	for _, shape := range [][2]int{{128, 128}, {512, 1024}} {
		for _, tokens := range []int{1, 128} {
			for _, k := range []int{2, 8} {
				for _, storage := range []string{"float32", "fp4"} {
					for _, dispatch := range []string{"dense", "sparse", "device"} {
						name := fmt.Sprintf("gpu/dim%d_inter%d/tokens%d/k%d/%s/%s", shape[0], shape[1], tokens, k, storage, dispatch)
						b.Run(name, func(b *testing.B) {
							dim, inter := shape[0], shape[1]
							if err := mlx.Synchronize(); err != nil {
								b.Fatal(err)
							}
							before, err := mlx.GetMemoryUsage()
							if err != nil {
								b.Fatal(err)
							}
							raw := deviceMoEFixture(dim, inter, count)
							x, gate, bias := deviceMoEInputs(b, tokens, dim, count, false)
							c := RouterConfig{TopK: k, Score: "sigmoid", Temperature: 1, Scale: 1, Normalize: true}
							var forward func() (mlx.Array, error)
							if dispatch == "device" {
								var bank *DeviceMoE
								if storage == "fp4" {
									bank, err = NewFP4DeviceMoE(raw[1:], raw[0], dim, inter)
								} else {
									weights := deviceMoEDecoded(b, raw, dim, inter)
									bank, err = NewDeviceMoE(weights[1:], weights[0])
									deviceMoECloseWeights(weights)
								}
								if err != nil {
									b.Fatal(err)
								}
								b.Cleanup(func() { bank.Close() })
								forward = func() (mlx.Array, error) { return bank.Forward(x, gate, bias, c, 1) }
							} else if storage == "fp4" {
								run := deviceMoEPackedRunner(b, raw, dim, inter, 1)
								forward = func() (mlx.Array, error) {
									return deviceMoEPackedReference(x, gate, bias, c, count, run, dispatch == "sparse")
								}
							} else {
								weights := deviceMoEDecoded(b, raw, dim, inter)
								forward = func() (mlx.Array, error) {
									if dispatch == "sparse" {
										return SparseMoE(x, gate, bias, weights[1:], weights[0], c, 1)
									}
									return MoE(x, gate, bias, weights[1:], weights[0], c, 1)
								}
							}
							probeWeights, probeIDs, err := Route(x, gate, bias, c)
							if err != nil {
								b.Fatal(err)
							}
							cast, err := mlx.AsType(probeIDs, mlx.Int32)
							if err != nil {
								b.Fatal(err)
							}
							ids, err := cast.Int32Data()
							mlx.CloseArrays([]mlx.Array{probeWeights, probeIDs, cast})
							if err != nil {
								b.Fatal(err)
							}
							active := make(map[int32]bool)
							for _, id := range ids {
								active[id] = true
							}
							step := func() error {
								y, err := forward()
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
							if err := mlx.Synchronize(); err != nil {
								b.Fatal(err)
							}
							steady, err := mlx.GetMemoryUsage()
							if err != nil {
								b.Fatal(err)
							}
							if err := mlx.ResetPeakMemory(); err != nil {
								b.Fatal(err)
							}
							b.ReportAllocs()
							b.ResetTimer()
							for range b.N {
								if err := mlx.Batch(step); err != nil {
									b.Fatal(err)
								}
							}
							b.StopTimer()
							if err := mlx.Synchronize(); err != nil {
								b.Fatal(err)
							}
							usage, err := mlx.GetMemoryUsage()
							if err != nil {
								b.Fatal(err)
							}
							if steady.ActiveBytes < before.ActiveBytes || usage.PeakBytes < steady.ActiveBytes {
								b.Fatal("allocator counters decreased; benchmark must run without concurrent MLX work")
							}
							b.ReportMetric(float64(usage.PeakBytes-before.ActiveBytes), "peak-active-bytes")
							b.ReportMetric(float64(steady.ActiveBytes-before.ActiveBytes), "steady-active-bytes")
							b.ReportMetric(float64(usage.PeakBytes-steady.ActiveBytes), "working-peak-bytes")
							b.ReportMetric(float64(len(active)), "active-experts")
							weightBytes := (count + 1) * 3 * dim * inter * 4
							if storage == "fp4" {
								weightBytes = (count + 1) * 3 * dim * inter * 17 / 32
							}
							b.ReportMetric(float64(weightBytes), "weight-bytes")
						})
					}
				}
			}
		}
	}
}
