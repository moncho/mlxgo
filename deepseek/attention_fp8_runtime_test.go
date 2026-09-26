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

func loadSampleAttentionModel(t testing.TB, dir string, ref attentionReference, fp8 bool) *Model {
	t.Helper()
	model := &Model{config: ref.Config.Config, weights: map[string]mlx.Array{}}
	t.Cleanup(func() { model.Close() })
	for _, m := range ref.Tensors {
		if fp8 && m.Name == fmt.Sprintf("layers.%d.attn.wkv.weight", ref.Layer) {
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
			weight := FP8AttentionWeight{
				Data:   read(m.Name+".bin", count, m.DataSHA),
				Scales: read(strings.TrimSuffix(m.Name, ".weight")+".scale.bin", count/1024, m.ScalesSHA),
			}
			if err := mlx.Batch(func() error { return model.initFP8AttentionKV(ref.Layer, weight) }); err != nil {
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
	if fp8 {
		if _, exists := model.weights[fmt.Sprintf("layers.%d.attn.wkv.weight", ref.Layer)]; exists || model.fp8AttentionKV[ref.Layer] == nil {
			t.Fatal("packed projection not substituted")
		}
	}
	return model
}

// Same normalization, attention implementation and cache evaluation as the
// reference test, without copying the output to Go during timing.
func evalSampleAttention(session *Session, x mlx.Array) error {
	y, err := one(func(s *scope) mlx.Array {
		normal := s.add(mlx.RMSNorm(x, session.model.weights["layers.0.attn_norm.weight"], session.model.config.Hyper.NormEpsilon))
		return session.attention(s, normal, 0, &attentionState{})
	})
	if err != nil {
		return err
	}
	defer y.Close()
	if err := mlx.Eval(append([]mlx.Array{y}, session.layers[0].arrays()...)...); err != nil {
		return err
	}
	session.offset += x.Shape()[1]
	return nil
}

func BenchmarkReleasedAttentionFP8(b *testing.B) {
	dir, ref := readAttentionReference(b)
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
					for _, packed := range []bool{false, true} {
						name := "float32"
						if packed {
							name = "fp8"
						}
						b.Run(name, func(b *testing.B) {
							baseline, err := mlx.GetMemoryUsage()
							if err != nil {
								b.Fatal(err)
							}
							m := loadSampleAttentionModel(b, dir, ref, packed)
							defer m.Close()
							prefill, err := mlx.NewFloat32(attentionInputs(0, 128, m.config.Dim), []int{1, 128, m.config.Dim})
							if err != nil {
								b.Fatal(err)
							}
							defer prefill.Close()
							x := prefill
							var template *Session
							if decode {
								template, err = m.NewSession()
								if err != nil {
									b.Fatal(err)
								}
								defer template.Close()
								if err := mlx.Batch(func() error { return evalSampleAttention(template, prefill) }); err != nil {
									b.Fatal(err)
								}
								x, err = mlx.NewFloat32(attentionInputs(128, 1, m.config.Dim), []int{1, 1, m.config.Dim})
								if err != nil {
									b.Fatal(err)
								}
								defer x.Close()
							}
							step := func() error {
								s, err := m.NewSession()
								if err != nil {
									return err
								}
								defer s.Close()
								if decode {
									window, err := mlx.Reshape(template.layers[0].window, []int{1, 128, m.config.HeadDim})
									if err != nil {
										return err
									}
									s.layers[0].window = window
									s.layers[0].windowLen = 128
									s.offset = 128
								}
								return evalSampleAttention(s, x)
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
							for _, weight := range ref.Tensors {
								count := 1
								for _, dim := range weight.Shape {
									count *= dim
								}
								if packed && weight.Name == "layers.0.attn.wkv.weight" {
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
