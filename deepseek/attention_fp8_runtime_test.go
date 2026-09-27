//go:build mlx && mlxruntime

package deepseek

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	mlx "github.com/moncho/mlxgo"
)

type fp8AttentionSelection struct{ kv, qb, oa bool }

func (s fp8AttentionSelection) selects(name string) bool {
	return (s.kv && strings.HasSuffix(name, ".attn.wkv.weight")) || (s.qb && strings.HasSuffix(name, ".attn.wq_b.weight")) || (s.oa && strings.HasSuffix(name, ".attn.wo_a.weight"))
}

func readPackedAttentionWeight(t testing.TB, dir string, m attentionTensorReference) FP8AttentionWeight {
	t.Helper()
	read := func(name string, size int, hash string) []byte {
		f, err := os.Open(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		b, err := io.ReadAll(io.LimitReader(f, int64(size)+1))
		h := sha256.Sum256(b)
		if err != nil || len(b) != size || hex.EncodeToString(h[:]) != hash {
			t.Fatalf("packed attention weight mismatch: %s: %v", name, err)
		}
		return b
	}
	count := m.Shape[0] * m.Shape[1]
	return FP8AttentionWeight{Data: read(m.Name+".bin", count, m.DataSHA), Scales: read(strings.TrimSuffix(m.Name, ".weight")+".scale.bin", count/1024, m.ScalesSHA)}
}

func loadSampleAttentionModel(t testing.TB, dir string, ref attentionReference, fp8 fp8AttentionSelection) *Model {
	t.Helper()
	model := &Model{config: ref.Config.Config, weights: map[string]mlx.Array{}}
	t.Cleanup(func() { model.Close() })
	loadSampleAttentionWeights(t, model, dir, ref, fp8)
	return model
}

func loadSampleAttentionWeights(t testing.TB, model *Model, dir string, ref attentionReference, fp8 fp8AttentionSelection) {
	t.Helper()
	for _, m := range ref.Tensors {
		if fp8.selects(m.Name) {
			weight := readPackedAttentionWeight(t, dir, m)
			if err := mlx.Batch(func() error { return model.initFP8Attention(m.Name, m.Shape, weight) }); err != nil {
				t.Fatal(err)
			}
			continue
		}
		a, err := mlx.NewFloat32(loadAttentionTensor(t, dir, m), m.Shape)
		if err != nil {
			t.Fatal(err)
		}
		model.weights[m.Name] = a
		data, err := a.Float32Data()
		if err != nil || matrixSHA(data) != m.DecodedSHA {
			t.Fatalf("native upload mismatch: %s: %v", m.Name, err)
		}
	}
	for _, m := range ref.Tensors {
		if !fp8.selects(m.Name) {
			continue
		}
		if _, exists := model.weights[m.Name]; exists || (model.fp8Attention[m.Name] == nil && len(model.fp8AttentionOutput[m.Name]) == 0) {
			t.Fatal("packed projection not substituted")
		}
	}
}

// Same normalization, attention implementation and cache evaluation as the
// reference test, without copying the output to Go during timing.
func evalSampleAttention(session *Session, x mlx.Array, layers []int) error {
	y, err := one(func(s *scope) mlx.Array {
		shared := &attentionState{}
		current := x
		for _, layer := range layers {
			normal := s.add(mlx.RMSNorm(current, session.model.weights[fmt.Sprintf("layers.%d.attn_norm.weight", layer)], session.model.config.Hyper.NormEpsilon))
			current = session.attention(s, normal, layer, shared)
		}
		return current
	})
	if err != nil {
		return err
	}
	defer y.Close()
	arrays := []mlx.Array{y}
	for _, layer := range layers {
		arrays = append(arrays, session.layers[layer].arrays()...)
	}
	if err := mlx.Eval(arrays...); err != nil {
		return err
	}
	session.offset += x.Shape()[1]
	return nil
}

func BenchmarkReleasedAttentionFP8(b *testing.B) {
	benchmarkAttentionModes(b, 0, false, false)
}

func BenchmarkReleasedCombinedLayers(b *testing.B) {
	for _, scenario := range []struct {
		name   string
		layer  int
		shared bool
	}{{"layer0", 0, false}, {"layer2", 2, false}, {"shared2_3", 2, true}} {
		b.Run(scenario.name, func(b *testing.B) { benchmarkAttentionModes(b, scenario.layer, scenario.shared, true) })
	}
}

func cloneBenchmarkCache(dst *layerCache, src layerCache) error {
	dst.windowLen, dst.pendingLen, dst.compressedLen = src.windowLen, src.pendingLen, src.compressedLen
	for _, p := range []struct {
		from mlx.Array
		to   *mlx.Array
	}{{src.window, &dst.window}, {src.pending, &dst.pending}, {src.compressed, &dst.compressed}, {src.keys, &dst.keys}, {src.windowScale, &dst.windowScale}, {src.compressedScale, &dst.compressedScale}, {src.keyScale, &dst.keyScale}} {
		if p.from == (mlx.Array{}) {
			continue
		}
		a, err := mlx.Reshape(p.from, p.from.Shape())
		if err != nil {
			return err
		}
		*p.to = a
	}
	return nil
}

func benchmarkAttentionModes(b *testing.B, layer int, shared, limited bool) {
	defer mlx.SetDefaultCPU()
	dir, ref := readAttentionLayerReference(b, layer)
	layers := []int{layer}
	var extraDir string
	var extra attentionReference
	if shared {
		dir, ref, extraDir, extra = readSharedAttentionReferencesMode(b, false)
		layers = append(layers, 3)
	}
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		b.Run(device.name, func(b *testing.B) {
			if err := device.set(); err != nil {
				b.Fatal(err)
			}
			for _, decode := range []bool{false, true} {
				name := "prefill128"
				if decode {
					name = "decode128"
				}
				b.Run(name, func(b *testing.B) {
					for _, mode := range []struct {
						name      string
						selection fp8AttentionSelection
						caches    bool
					}{
						{"float32", fp8AttentionSelection{}, false}, {"kv", fp8AttentionSelection{kv: true}, false},
						{"qb", fp8AttentionSelection{qb: true}, false}, {"kv_qb", fp8AttentionSelection{kv: true, qb: true}, false},
						{"oa", fp8AttentionSelection{oa: true}, false}, {"kv_qb_oa", fp8AttentionSelection{kv: true, qb: true, oa: true}, false},
						{"caches", fp8AttentionSelection{}, true}, {"combined", fp8AttentionSelection{kv: true, qb: true, oa: true}, true},
					} {
						if limited && mode.name != "float32" && mode.name != "combined" && (layer != 0 || (mode.name != "caches" && mode.name != "kv_qb_oa")) {
							continue
						}
						b.Run(mode.name, func(b *testing.B) {
							baseline, err := mlx.GetMemoryUsage()
							if err != nil {
								b.Fatal(err)
							}
							m := loadSampleAttentionModel(b, dir, ref, mode.selection)
							defer m.Close()
							if shared {
								m.config = extra.Config.Config
								loadSampleAttentionWeights(b, m, extraDir, extra, mode.selection)
							}
							prefill, err := mlx.NewFloat32(attentionInputs(0, 128, m.config.Dim), []int{1, 128, m.config.Dim})
							if err != nil {
								b.Fatal(err)
							}
							defer prefill.Close()
							x := prefill
							var template *Session
							if decode {
								template, err = m.NewSessionWithOptions(SessionOptions{QuantizedCaches: mode.caches})
								if err != nil {
									b.Fatal(err)
								}
								defer template.Close()
								if err := mlx.Batch(func() error { return evalSampleAttention(template, prefill, layers) }); err != nil {
									b.Fatal(err)
								}
								x, err = mlx.NewFloat32(attentionInputs(128, 1, m.config.Dim), []int{1, 1, m.config.Dim})
								if err != nil {
									b.Fatal(err)
								}
								defer x.Close()
							}
							step := func() error {
								s, err := m.NewSessionWithOptions(SessionOptions{QuantizedCaches: mode.caches})
								if err != nil {
									return err
								}
								defer s.Close()
								if decode {
									for _, layer := range layers {
										if err := cloneBenchmarkCache(&s.layers[layer], template.layers[layer]); err != nil {
											return err
										}
									}
									s.offset = 128
								}
								return evalSampleAttention(s, x, layers)
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
							if usage.PeakBytes < baseline.ActiveBytes || steady.ActiveBytes < baseline.ActiveBytes {
								b.Fatal("unexpected allocator counters")
							}
							b.ReportMetric(float64(usage.PeakBytes-baseline.ActiveBytes), "peak-active-bytes")
							b.ReportMetric(float64(steady.ActiveBytes-baseline.ActiveBytes), "steady-active-bytes")
							payload := 0
							for _, weight := range append(append([]attentionTensorReference{}, ref.Tensors...), extra.Tensors...) {
								count := 1
								for _, dim := range weight.Shape {
									count *= dim
								}
								if mode.selection.selects(weight.Name) {
									payload += count + count/32
								} else {
									payload += 4 * count
								}
							}
							b.ReportMetric(float64(payload), "weight-bytes")
						})
					}
				})
			}
		})
	}
}
