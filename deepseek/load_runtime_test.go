//go:build mlx && mlxruntime

package deepseek

import (
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	mlx "github.com/moncho/mlxgo"
)

func packedBundleFixture(t *testing.T) (Config, map[string]mlx.Array, ModelOptions, LoadOptions) {
	t.Helper()
	c, params, experts, attn := fp4ModelInputsWithCompression(t, true)
	p := maps.Clone(params)
	o := ModelOptions{FP4Experts: make(map[ExpertID]FP4ExpertWeights), FP8AttentionKV: map[int]FP8AttentionWeight{0: attn["layers.0.attn.wkv.weight"]}, FP8AttentionQB: map[int]FP8AttentionWeight{0: attn["layers.0.attn.wq_b.weight"]}, FP8AttentionOA: map[int]FP8AttentionWeight{0: attn["layers.0.attn.wo_a.weight"]}}
	l := LoadOptions{FP4Experts: []ExpertID{{Layer: 0, Index: -1}, {Layer: 0, Index: 0}, {Layer: 1, Index: 1}}, FP8AttentionKV: []int{0}, FP8AttentionQB: []int{0}, FP8AttentionOA: []int{0}}
	for _, id := range l.FP4Experts {
		o.FP4Experts[id] = experts[id]
	}
	layout, _, err := loadLayout(c, l)
	if err != nil {
		t.Fatal(err)
	}
	put := func(name string, data []byte) {
		a, err := mlx.NewUInt8(data, layout[name].shape)
		if err != nil {
			t.Fatal(err)
		}
		p[name] = a
		t.Cleanup(func() { a.Close() })
	}
	weight := func(name string, w FP4Weight) {
		put(name, w.Data)
		put(strings.TrimSuffix(name, ".weight")+".scale", w.Scales)
	}
	for id, w := range o.FP4Experts {
		prefix := expertPrefix(id)
		weight(prefix+"w1.weight", w.Gate)
		weight(prefix+"w2.weight", w.Down)
		weight(prefix+"w3.weight", w.Up)
	}
	for name, w := range attn {
		weight(name, FP4Weight{Data: w.Data, Scales: w.Scales})
	}
	return c, p, o, l
}

func savePackedBundle(t *testing.T, c Config, p map[string]mlx.Array, shards int) string {
	t.Helper()
	dir := t.TempDir()
	b, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(dir, "config.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
	names := slices.Sorted(maps.Keys(p))
	index := map[string]string{}
	for i := 0; i < shards; i++ {
		file := "model.safetensors"
		if shards > 1 {
			file = fmt.Sprintf("model-%05d-of-%05d.safetensors", i+1, shards)
		}
		part := map[string]mlx.Array{}
		for j := i; j < len(names); j += shards {
			part[names[j]] = p[names[j]]
			index[names[j]] = file
		}
		if err := mlx.SaveSafetensors(filepath.Join(dir, file), part, nil); err != nil {
			t.Fatal(err)
		}
	}
	if shards > 1 {
		b, err := json.Marshal(map[string]any{"weight_map": index})
		if err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), b, 0600); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func TestPackedLoadAndSparseSessions(t *testing.T) {
	defer mlx.SetDefaultCPU()
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			c, p, o, l := packedBundleFixture(t)
			layout, _, _ := loadLayout(c, l)
			ordinary := map[string]mlx.Array{}
			for name, a := range p {
				if !layout[name].packed {
					ordinary[name] = a
				}
			}
			manual, err := NewModelWithOptions(c, ordinary, o)
			if err != nil {
				t.Fatal(err)
			}
			defer manual.Close()
			tokens := []int32{1, 5, 2, 9, 4, 3}
			for _, shards := range []int{1, 3} {
				t.Run(fmt.Sprintf("shards%d", shards), func(t *testing.T) {
					m, err := Load(savePackedBundle(t, c, p, shards), c, l)
					if err != nil {
						t.Fatal(err)
					}
					defer m.Close()
					if len(m.fp4Experts) != len(l.FP4Experts) || len(m.fp8Attention) != 2 || len(m.fp8AttentionOutput) != 1 {
						t.Fatal("packed storage missing")
					}
					for _, cache := range []bool{false, true} {
						want, err := fp8SessionOutput(manual, tokens, cache)
						if err != nil {
							t.Fatal(err)
						}
						for _, sparse := range []bool{false, true} {
							options := SessionOptions{SparseExperts: sparse, QuantizedCaches: cache}
							got, err := sessionOutputOptions(m, tokens, options)
							if err != nil {
								t.Fatal(err)
							}
							compareSparse(t, got, want, 2e-5)
							s, err := m.NewSessionWithOptions(options)
							if err != nil {
								t.Fatal(err)
							}
							full, err := closeLogits(s.ForwardAll(tokens))
							s.Close()
							if err != nil {
								t.Fatal(err)
							}
							compareSparse(t, full, want, 2e-5)
							var wg sync.WaitGroup
							for range 8 {
								wg.Go(func() {
									v, err := sessionOutputOptions(m, tokens, options)
									if err != nil {
										t.Error(err)
										return
									}
									for i, x := range v {
										if d := x - want[i]; d > 2e-5 || d < -2e-5 || x != x {
											t.Error("concurrent logits differ")
											return
										}
									}
								})
							}
							wg.Wait()
						}
					}
				})
			}
		})
	}
}

func TestPackedLoadRejects(t *testing.T) {
	if err := mlx.SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}
	c, params, _, l := packedBundleFixture(t)
	for _, kind := range []string{"missing", "extra", "shape", "dtype", "nonfinite", "budget", "unselected", "duplicate"} {
		t.Run(kind, func(t *testing.T) {
			p := maps.Clone(params)
			o := l
			name := "layers.0.ffn.experts.0.w1.scale"
			switch kind {
			case "missing":
				delete(p, name)
			case "extra":
				p["unexpected"] = p[name]
			case "shape":
				a, err := mlx.Reshape(p[name], []int{p[name].Size()})
				if err != nil {
					t.Fatal(err)
				}
				defer a.Close()
				p[name] = a
			case "dtype":
				a, err := mlx.AsType(p[name], mlx.Float32)
				if err != nil {
					t.Fatal(err)
				}
				defer a.Close()
				p[name] = a
			case "nonfinite":
				v, err := p[name].UInt8Data()
				if err != nil {
					t.Fatal(err)
				}
				v[0] = 255
				a, err := mlx.NewUInt8(v, p[name].Shape())
				if err != nil {
					t.Fatal(err)
				}
				defer a.Close()
				p[name] = a
			case "budget":
				o.MaxBytes = 1
			case "unselected":
				o = LoadOptions{}
			case "duplicate":
				o.FP4Experts = append(slices.Clone(o.FP4Experts), o.FP4Experts[0])
			}
			m, err := Load(savePackedBundle(t, c, p, 1), c, o)
			if err == nil {
				m.Close()
				t.Fatal("accepted", kind)
			}
		})
	}
}
