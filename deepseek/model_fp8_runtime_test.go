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

func fp8ModelInputs(t *testing.T) (Config, map[string]mlx.Array, FP8AttentionWeight) {
	t.Helper()
	c := readQuantizedModelFixture(t).Config.clone()
	c.Dim = 32
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
	weight := FP8AttentionWeight{Data: make([]byte, c.HeadDim*c.Dim), Scales: []byte{124}}
	for i := range weight.Data {
		weight.Data[i] = byte(32+i*13%48) | byte(i%2)<<7
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
		if name == "layers.0.attn.wkv.weight" {
			if err := quant.Decode(data, weight.Data, weight.Scales, c.HeadDim, c.Dim, quant.FP8Block32, quant.Float32); err != nil {
				t.Fatal(err)
			}
		}
		a, err := mlx.NewFloat32(data, shape)
		if err != nil {
			t.Fatal(err)
		}
		parameters[name] = a
	}
	return c, parameters, weight
}

func TestFP8ModelSessions(t *testing.T) {
	defer mlx.SetDefaultCPU()
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			c, parameters, weight := fp8ModelInputs(t)
			baseline, err := NewModel(c, parameters)
			if err != nil {
				t.Fatal(err)
			}
			defer baseline.Close()
			inputs := maps.Clone(parameters)
			delete(inputs, "layers.0.attn.wkv.weight")
			options := ModelOptions{FP8AttentionKV: map[int]FP8AttentionWeight{0: weight}}
			packed, err := NewModelWithOptions(c, inputs, options)
			if err != nil {
				t.Fatal(err)
			}
			defer packed.Close()
			if len(baseline.fp8AttentionKV) != 0 || len(packed.fp8AttentionKV) != 1 {
				t.Fatal("incorrect projection selection")
			}
			if _, ok := packed.weights["layers.0.attn.wkv.weight"]; ok {
				t.Fatal("duplicate float32 weight retained")
			}
			if _, ok := packed.weights["layers.1.attn.wkv.weight"]; !ok {
				t.Fatal("unselected layer changed")
			}
			clear(weight.Data)
			clear(weight.Scales)
			clear(options.FP8AttentionKV)
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
			owned := packed.fp8AttentionKV[0]
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
			x, err := mlx.Ones([]int{1, c.Dim}, mlx.Float32)
			if err != nil {
				t.Fatal(err)
			}
			defer x.Close()
			if y, err := owned.Forward(x); err == nil {
				y.Close()
				t.Fatal("model did not close packed weight")
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
	c, parameters, weight := fp8ModelInputs(t)
	for _, name := range []string{"duplicate", "layer range", "missing", "unexpected", "truncated", "nonfinite", "unaligned"} {
		t.Run(name, func(t *testing.T) {
			config := c.clone()
			p := maps.Clone(parameters)
			delete(p, "layers.0.attn.wkv.weight")
			o := ModelOptions{FP8AttentionKV: map[int]FP8AttentionWeight{0: weight}}
			switch name {
			case "duplicate":
				p["layers.0.attn.wkv.weight"] = parameters["layers.0.attn.wkv.weight"]
			case "layer range":
				o.FP8AttentionKV = map[int]FP8AttentionWeight{c.Layers: weight}
			case "missing":
				delete(p, "norm.weight")
			case "unexpected":
				p["invalid.weight"] = p["norm.weight"]
				delete(p, "norm.weight")
			case "truncated":
				o.FP8AttentionKV[0] = FP8AttentionWeight{weight.Data[:1], weight.Scales}
			case "nonfinite":
				o.FP8AttentionKV[0] = FP8AttentionWeight{weight.Data, []byte{255}}
			case "unaligned":
				config.HeadDim = 16
			}
			m, err := NewModelWithOptions(config, p, o)
			if err == nil || m != nil {
				if m != nil {
					m.Close()
				}
				t.Fatal("invalid model accepted")
			}
			if _, err := parameters["norm.weight"].Float32Data(); err != nil {
				t.Fatal("failed construction closed caller input", err)
			}
		})
	}
}
