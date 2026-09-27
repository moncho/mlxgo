//go:build mlx && mlxruntime

package deepseek

import (
	"fmt"
	"maps"
	"math"
	"strings"
	"sync"
	"testing"

	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/deepseek/quant"
)

func fp4ModelInputs(t *testing.T) (Config, map[string]mlx.Array, map[ExpertID]FP4ExpertWeights, map[string]FP8AttentionWeight) {
	return fp4ModelInputsWithCompression(t, false)
}

func fp4ModelInputsWithCompression(t *testing.T, compressed bool) (Config, map[string]mlx.Array, map[ExpertID]FP4ExpertWeights, map[string]FP8AttentionWeight) {
	t.Helper()
	c, params, attn := fp8ModelInputs(t)
	c.InterDim = 32
	if compressed {
		c.Ratios = []int{2, 2}
		c.KVSources = []int{0}
		c.IndexSources = []int{0}
	}
	shapes, err := c.ParameterShapes()
	if err != nil {
		t.Fatal(err)
	}
	for name, shape := range shapes {
		if _, exists := params[name]; exists {
			continue
		}
		n := 1
		for _, d := range shape {
			n *= d
		}
		values := make([]float32, n)
		for i := range values {
			values[i] = float32((i*37+len(name)*11)%127-63) / 512
			if strings.Contains(name, "norm.weight") {
				values[i] = 1
			}
		}
		params[name], err = mlx.NewFloat32(values, shape)
		if err != nil {
			t.Fatal(err)
		}
	}
	weights := make(map[ExpertID]FP4ExpertWeights)
	for layer := 0; layer < c.Layers; layer++ {
		for index := -1; index < c.Experts; index++ {
			id := ExpertID{Layer: layer, Index: index}
			prefix := expertPrefix(id)
			parts := make([]FP4Weight, 3)
			for k, part := range []string{"w1.weight", "w2.weight", "w3.weight"} {
				name := prefix + part
				shape := shapes[name]
				n := shape[0] * shape[1]
				w := FP4Weight{Data: make([]byte, n/2), Scales: make([]byte, n/32)}
				for i := range w.Data {
					w.Data[i] = byte(i*17 + len(name)*3 + k*37)
				}
				for i := range w.Scales {
					w.Scales[i] = byte(121 + i%3)
				}
				decoded := make([]float32, n)
				if err := quant.Decode(decoded, w.Data, w.Scales, shape[0], shape[1], quant.FP4Row32, quant.Float32); err != nil {
					t.Fatal(err)
				}
				previous := params[name]
				previous.Close()
				params[name], err = mlx.NewFloat32(decoded, shape)
				if err != nil {
					t.Fatal(err)
				}
				parts[k] = w
			}
			weights[id] = FP4ExpertWeights{Gate: parts[0], Down: parts[1], Up: parts[2]}
		}
	}
	return c, params, weights, attn
}

func TestFP4ModelSessions(t *testing.T) {
	defer mlx.SetDefaultCPU()
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			for _, mode := range []string{"routed", "shared", "mixed_with_fp8", "compressed_mixed"} {
				t.Run(mode, func(t *testing.T) {
					c, params, weights, attn := fp4ModelInputsWithCompression(t, mode == "compressed_mixed")
					baseline, err := NewModel(c, params)
					if err != nil {
						t.Fatal(err)
					}
					defer baseline.Close()
					inputs := maps.Clone(params)
					o := ModelOptions{FP4Experts: make(map[ExpertID]FP4ExpertWeights)}
					ids := []ExpertID{{Layer: 0, Index: 0}}
					if mode == "shared" {
						ids = []ExpertID{{Layer: 0, Index: -1}}
					}
					if mode == "mixed_with_fp8" || mode == "compressed_mixed" {
						ids = append(ids, ExpertID{Layer: 0, Index: -1}, ExpertID{Layer: 1, Index: 1})
					}
					for _, id := range ids {
						o.FP4Experts[id] = weights[id]
						for _, part := range []string{"w1.weight", "w2.weight", "w3.weight"} {
							delete(inputs, expertPrefix(id)+part)
						}
					}
					if mode == "mixed_with_fp8" || mode == "compressed_mixed" {
						o.FP8AttentionKV = map[int]FP8AttentionWeight{0: attn["layers.0.attn.wkv.weight"]}
						o.FP8AttentionQB = map[int]FP8AttentionWeight{0: attn["layers.0.attn.wq_b.weight"]}
						o.FP8AttentionOA = map[int]FP8AttentionWeight{0: attn["layers.0.attn.wo_a.weight"]}
						for name := range attn {
							delete(inputs, name)
						}
					}
					m, err := NewModelWithOptions(c, inputs, o)
					if err != nil {
						t.Fatal(err)
					}
					defer m.Close()
					if len(m.fp4Experts) != len(ids) {
						t.Fatal("incorrect expert selection")
					}
					for id := range weights {
						_, selected := o.FP4Experts[id]
						for _, part := range []string{"w1.weight", "w2.weight", "w3.weight"} {
							_, ordinary := m.weights[expertPrefix(id)+part]
							if ordinary == selected {
								t.Fatal("incorrect packed/float storage", id)
							}
						}
					}
					for _, w := range weights {
						for _, v := range []FP4Weight{w.Gate, w.Up, w.Down} {
							clear(v.Data)
							clear(v.Scales)
						}
					}
					clear(o.FP4Experts)
					clear(o.FP8AttentionKV)
					clear(o.FP8AttentionQB)
					clear(o.FP8AttentionOA)
					for _, a := range params {
						a.Close()
					}
					tokens := []int32{1, 5, 2, 9, 4, 3}
					for _, cache := range []bool{false, true} {
						t.Run(fmt.Sprintf("caches_%t", cache), func(t *testing.T) {
							want, err := fp8SessionOutput(baseline, tokens, cache)
							if err != nil {
								t.Fatal(err)
							}
							got, err := fp8SessionOutput(m, tokens, cache)
							if err != nil {
								t.Fatal(err)
							}
							compareLogits(t, got, want)
							var wg sync.WaitGroup
							errs := make(chan error, 8)
							for i := 0; i < 8; i++ {
								wg.Add(1)
								go func() {
									defer wg.Done()
									v, e := fp8SessionOutput(m, tokens, cache)
									if e == nil {
										if len(v) != len(want) {
											e = fmt.Errorf("wrong output count")
										} else {
											for i, x := range v {
												if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) || math.Abs(float64(x-want[i])) > 2e-5 {
													e = fmt.Errorf("concurrent logits differ")
													break
												}
											}
										}
									}
									errs <- e
								}()
							}
							wg.Wait()
							close(errs)
							for e := range errs {
								if e != nil {
									t.Fatal(e)
								}
							}
						})
					}
					owned := maps.Clone(m.fp4Experts)
					if err := m.Close(); err != nil {
						t.Fatal(err)
					}
					if _, err := m.NewSession(); err == nil {
						t.Fatal("closed model accepted")
					}
					x, err := mlx.Ones([]int{1, c.Dim}, mlx.Float32)
					if err != nil {
						t.Fatal(err)
					}
					defer x.Close()
					r, err := mlx.Ones([]int{1, 1}, mlx.Float32)
					if err != nil {
						t.Fatal(err)
					}
					defer r.Close()
					for _, p := range owned {
						if y, err := p.Forward(x, r, c.Limit); err == nil {
							y.Close()
							t.Fatal("model leaked expert")
						}
					}
				})
			}
		})
	}
}

func TestFP4ModelValidation(t *testing.T) {
	if err := mlx.SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}
	c, params, weights, _ := fp4ModelInputs(t)
	id := ExpertID{Layer: 0, Index: 0}
	w := weights[id]
	for _, name := range []string{"duplicate", "layer", "index", "negative index", "missing", "unexpected", "late invalid down", "NaN", "dimension"} {
		t.Run(name, func(t *testing.T) {
			config := c.clone()
			p := maps.Clone(params)
			prefix := expertPrefix(id)
			for _, part := range []string{"w1.weight", "w2.weight", "w3.weight"} {
				delete(p, prefix+part)
			}
			o := ModelOptions{FP4Experts: map[ExpertID]FP4ExpertWeights{id: w}}
			switch name {
			case "duplicate":
				p[prefix+"w1.weight"] = params[prefix+"w1.weight"]
			case "layer":
				delete(o.FP4Experts, id)
				o.FP4Experts[ExpertID{Layer: c.Layers}] = w
			case "index":
				delete(o.FP4Experts, id)
				o.FP4Experts[ExpertID{Index: c.Experts}] = w
			case "negative index":
				delete(o.FP4Experts, id)
				o.FP4Experts[ExpertID{Index: -2}] = w
			case "missing":
				delete(p, "norm.weight")
			case "unexpected":
				p["unexpected"] = p["norm.weight"]
				delete(p, "norm.weight")
			case "late invalid down":
				bad := w
				bad.Down.Scales = nil
				o.FP4Experts[id] = bad
			case "NaN":
				bad := w
				bad.Down.Scales = append([]byte(nil), w.Down.Scales...)
				bad.Down.Scales[0] = 255
				o.FP4Experts[id] = bad
			case "dimension":
				config.InterDim = 31
			}
			m, err := NewModelWithOptions(config, p, o)
			if err == nil || m != nil {
				if m != nil {
					m.Close()
				}
				t.Fatal("invalid expert options accepted")
			}
			if name == "NaN" && !strings.Contains(err.Error(), "nonfinite") {
				t.Fatal(err)
			}
			if _, err := params["norm.weight"].Float32Data(); err != nil {
				t.Fatal("failed constructor closed input", err)
			}
		})
	}
}
