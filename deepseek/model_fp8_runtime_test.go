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

func fp8ModelInputs(t *testing.T) (Config, map[string]mlx.Array, map[string]FP8AttentionWeight) {
	t.Helper()
	c := readQuantizedModelFixture(t).Config.clone()
	c.Dim = 32
	c.QRank = 32
	c.ORank = 32
	c.Layers = 2
	c.Ratios = []int{0, 0}
	c.KVSources = nil
	c.IndexSources = nil
	c.CandidateSource = -1
	c.Window = 3
	c.MaxSeq = 8
	shapes, err := c.ParameterShapes()
	if err != nil {
		t.Fatal(err)
	}
	weights := map[string]FP8AttentionWeight{}
	for _, part := range []string{"wkv", "wq_b", "wo_a"} {
		shape := shapes["layers.0.attn."+part+".weight"]
		weight := FP8AttentionWeight{Data: make([]byte, shape[0]*shape[1]), Scales: make([]byte, shape[0]*shape[1]/1024)}
		for i := range weight.Data {
			weight.Data[i] = byte(32+i*13%48) | byte(i%2)<<7
		}
		for i := range weight.Scales {
			weight.Scales[i] = byte(124 + i%2)
		}
		weights["layers.0.attn."+part+".weight"] = weight
	}
	parameters := map[string]mlx.Array{}
	t.Cleanup(func() {
		for _, a := range parameters {
			a.Close()
		}
	})
	for name, shape := range shapes {
		count := 1
		for _, d := range shape {
			count *= d
		}
		data := make([]float32, count)
		for i := range data {
			data[i] = float32((i*37+len(name)*11)%127-63) / 512
		}
		if strings.Contains(name, "norm.weight") {
			for i := range data {
				data[i] = 1
			}
		}
		if weight, ok := weights[name]; ok {
			if err := quant.Decode(data, weight.Data, weight.Scales, shape[0], shape[1], quant.FP8Block32, quant.Float32); err != nil {
				t.Fatal(err)
			}
		}
		a, err := mlx.NewFloat32(data, shape)
		if err != nil {
			t.Fatal(err)
		}
		parameters[name] = a
	}
	return c, parameters, weights
}

func TestFP8ModelSessions(t *testing.T) {
	for _, mode := range []struct {
		name      string
		selection fp8AttentionSelection
	}{
		{"kv", fp8AttentionSelection{kv: true}}, {"qb", fp8AttentionSelection{qb: true}}, {"kv_qb", fp8AttentionSelection{kv: true, qb: true}},
		{"oa", fp8AttentionSelection{oa: true}}, {"kv_oa", fp8AttentionSelection{kv: true, oa: true}},
		{"qb_oa", fp8AttentionSelection{qb: true, oa: true}}, {"kv_qb_oa", fp8AttentionSelection{kv: true, qb: true, oa: true}},
	} {
		t.Run(mode.name, func(t *testing.T) { testFP8ModelSessions(t, mode.selection) })
	}
}

func testFP8ModelSessions(t *testing.T, selection fp8AttentionSelection) {
	defer mlx.SetDefaultCPU()
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			c, parameters, weights := fp8ModelInputs(t)
			baseline, err := NewModel(c, parameters)
			if err != nil {
				t.Fatal(err)
			}
			defer baseline.Close()
			inputs := maps.Clone(parameters)
			options := ModelOptions{}
			if selection.kv {
				options.FP8AttentionKV = map[int]FP8AttentionWeight{0: weights["layers.0.attn.wkv.weight"]}
			}
			if selection.qb {
				options.FP8AttentionQB = map[int]FP8AttentionWeight{0: weights["layers.0.attn.wq_b.weight"]}
			}
			if selection.oa {
				options.FP8AttentionOA = map[int]FP8AttentionWeight{0: weights["layers.0.attn.wo_a.weight"]}
			}
			for name := range weights {
				if selection.selects(name) {
					delete(inputs, name)
				}
			}
			packed, err := NewModelWithOptions(c, inputs, options)
			if err != nil {
				t.Fatal(err)
			}
			defer packed.Close()
			if len(baseline.fp8Attention) != 0 || len(baseline.fp8AttentionOutput) != 0 || len(packed.fp8Attention) != len(options.FP8AttentionKV)+len(options.FP8AttentionQB) || len(packed.fp8AttentionOutput) != len(options.FP8AttentionOA) {
				t.Fatal("incorrect projection selection")
			}
			for name := range weights {
				_, floatExists := packed.weights[name]
				if floatExists == selection.selects(name) || (packed.fp8Attention[name] != nil || len(packed.fp8AttentionOutput[name]) != 0) != selection.selects(name) {
					t.Fatal("incorrect projection storage", name)
				}
			}
			for _, part := range []string{"wkv", "wq_b", "wo_a"} {
				if _, ok := packed.weights["layers.1.attn."+part+".weight"]; !ok {
					t.Fatal("unselected layer changed")
				}
			}
			for _, weight := range weights {
				clear(weight.Data)
				clear(weight.Scales)
			}
			clear(options.FP8AttentionKV)
			clear(options.FP8AttentionQB)
			clear(options.FP8AttentionOA)
			for _, a := range parameters {
				a.Close()
			}
			tokens := []int32{1, 5, 2, 9, 4, 3}
			for _, quantizedCaches := range []bool{false, true} {
				t.Run(fmt.Sprintf("quantized_caches_%v", quantizedCaches), func(t *testing.T) {
					want, err := fp8SessionOutput(baseline, tokens, quantizedCaches)
					if err != nil {
						t.Fatal(err)
					}
					got, err := fp8SessionOutput(packed, tokens, quantizedCaches)
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
							got, err := fp8SessionOutput(packed, tokens, quantizedCaches)
							if err == nil {
								if len(got) != len(want) {
									err = fmt.Errorf("wrong logit count")
								} else {
									for j, v := range got {
										if math.IsNaN(float64(v)) || math.Abs(float64(v-want[j])) > 2e-5 {
											err = fmt.Errorf("concurrent logit %d differs", j)
											break
										}
									}
								}
							}
							errs <- err
						}()
					}
					wg.Wait()
					close(errs)
					for err := range errs {
						if err != nil {
							t.Error(err)
						}
					}
				})
			}
			s, err := packed.NewSession()
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			owned := maps.Clone(packed.fp8Attention)
			ownedOutput := maps.Clone(packed.fp8AttentionOutput)
			if err := packed.Close(); err != nil {
				t.Fatal(err)
			}
			if _, err := packed.NewSession(); err == nil {
				t.Fatal("closed model created session")
			}
			if y, err := s.ForwardAll(tokens[:1]); err == nil {
				y.Close()
				t.Fatal("session used closed model")
			}
			x, err := mlx.Ones([]int{1, c.Dim}, mlx.Float32) // Dim and QRank both 32 in this fixture.
			if err != nil {
				t.Fatal(err)
			}
			defer x.Close()
			for _, projection := range owned {
				if y, err := projection.Forward(x); err == nil {
					y.Close()
					t.Fatal("model did not close packed weight")
				}
			}
			groupInput, err := mlx.Ones([]int{1, c.Heads * c.HeadDim / c.Groups}, mlx.Float32)
			if err != nil {
				t.Fatal(err)
			}
			defer groupInput.Close()
			for _, groups := range ownedOutput {
				for _, projection := range groups {
					if y, err := projection.Forward(groupInput); err == nil {
						y.Close()
						t.Fatal("model did not close packed output group")
					}
				}
			}
		})
	}
}

func fp8SessionOutput(m *Model, tokens []int32, quantizedCaches bool) ([]float32, error) {
	s, err := m.NewSessionWithOptions(SessionOptions{QuantizedCaches: quantizedCaches})
	if err != nil {
		return nil, err
	}
	defer s.Close()
	out, err := closeLogits(s.ForwardAll(tokens[:2]))
	if err != nil {
		return nil, err
	}
	for i := 2; i < len(tokens); i++ {
		v, err := closeLogits(s.ForwardAll(tokens[i : i+1]))
		if err != nil {
			return nil, err
		}
		out = append(out, v...)
	}
	if s.Position() != len(tokens) || s.layers[0].windowLen != m.config.Window {
		return nil, fmt.Errorf("cache did not roll over")
	}
	return out, nil
}

func TestFP8ModelValidation(t *testing.T) {
	if err := mlx.SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}
	c, parameters, weights := fp8ModelInputs(t)
	for _, part := range []string{"wkv", "wq_b", "wo_a"} {
		t.Run(part, func(t *testing.T) {
			nameOfWeight := "layers.0.attn." + part + ".weight"
			weight := weights[nameOfWeight]
			cases := []string{"duplicate", "layer range", "missing", "unexpected", "truncated", "nonfinite", "unaligned"}
			if part == "wo_a" {
				cases = append(cases, "BF16 conversion")
			}
			for _, name := range cases {
				t.Run(name, func(t *testing.T) {
					config := c.clone()
					p := maps.Clone(parameters)
					delete(p, nameOfWeight)
					selected := map[int]FP8AttentionWeight{0: weight}
					o := ModelOptions{}
					if part == "wkv" {
						o.FP8AttentionKV = selected
					} else if part == "wq_b" {
						o.FP8AttentionQB = selected
					} else {
						o.FP8AttentionOA = selected
					}
					switch name {
					case "duplicate":
						p[nameOfWeight] = parameters[nameOfWeight]
					case "layer range":
						delete(selected, 0)
						selected[c.Layers] = weight
					case "missing":
						delete(p, "norm.weight")
					case "unexpected":
						p["invalid.weight"] = p["norm.weight"]
						delete(p, "norm.weight")
					case "truncated":
						selected[0] = FP8AttentionWeight{weight.Data[:1], weight.Scales}
					case "BF16 conversion":
						data := append([]byte(nil), weight.Data...)
						scales := append([]byte(nil), weight.Scales...)
						data[0], scales[0] = 1, 0
						selected[0] = FP8AttentionWeight{data, scales}
					case "nonfinite":
						scales := append([]byte(nil), weight.Scales...)
						scales[0] = 255
						selected[0] = FP8AttentionWeight{weight.Data, scales}
					case "unaligned":
						if part == "wkv" {
							config.Dim = 31
						} else if part == "wq_b" {
							config.QRank = 31
						} else {
							config.ORank = 31
						}
					}
					m, err := NewModelWithOptions(config, p, o)
					if err == nil || m != nil {
						if m != nil {
							m.Close()
						}
						t.Fatal("invalid model accepted")
					}
					if name == "BF16 conversion" && !strings.Contains(err.Error(), name) {
						t.Fatal("wrong failure", err)
					}
					if _, err := parameters["norm.weight"].Float32Data(); err != nil {
						t.Fatal("failed construction closed caller input", err)
					}
				})
			}
		})
	}
}
