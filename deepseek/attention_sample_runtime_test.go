//go:build mlx && mlxruntime

package deepseek

import (
	"fmt"
	"math"
	"slices"
	"testing"

	mlx "github.com/moncho/mlxgo"
)

// Exercise the existing Session.attention implementation, including its real
// cache management. No alternative Go attention implementation is used here.
func runSampleAttentionLayer(t *testing.T, session *Session, input []float32, layer int) []float32 {
	t.Helper()
	var result []float32
	err := mlx.Batch(func() error {
		n := len(input) / session.model.config.Dim
		x, err := mlx.NewFloat32(input, []int{1, n, session.model.config.Dim})
		if err != nil {
			return err
		}
		defer x.Close()
		y, err := one(func(s *scope) mlx.Array {
			normal := s.add(mlx.RMSNorm(x, session.model.weights[fmt.Sprintf("layers.%d.attn_norm.weight", layer)], session.model.config.Hyper.NormEpsilon))
			return session.attention(s, normal, layer, &attentionState{})
		})
		if err != nil {
			return err
		}
		defer y.Close()
		if err = mlx.Eval(append([]mlx.Array{y}, session.layers[layer].arrays()...)...); err != nil {
			return err
		}
		result, err = y.Float32Data()
		if err == nil {
			session.offset += n
		}
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func compareAttention(t *testing.T, name string, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length %d != %d", name, len(got), len(want))
	}
	var maxAbs, maxScaled float64
	for i, v := range got {
		e := math.Abs(float64(v) - float64(want[i]))
		scale := 1 + math.Abs(float64(want[i]))
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || e > tolerance*scale {
			t.Fatalf("%s[%d] got %g want %g abs_error=%g scaled=%g", name, i, v, want[i], e, e/scale)
		}
		maxAbs = math.Max(maxAbs, e)
		maxScaled = math.Max(maxScaled, e/scale)
	}
	t.Logf("%s: %d values max_abs_error=%g max_scaled_error=%g", name, len(got), maxAbs, maxScaled)
}

func TestReleasedAttentionForward(t *testing.T) {
	testReleasedAttentionLayer(t, 0)
}

func TestReleasedCompressedAttentionForward(t *testing.T) {
	testReleasedAttentionLayer(t, 2)
}

func testReleasedAttentionLayer(t *testing.T, layer int) {
	dir, ref := readAttentionLayerReference(t, layer)
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			model := &Model{config: ref.Config.Config, weights: map[string]mlx.Array{}}
			t.Cleanup(func() { model.Close() })
			for _, m := range ref.Tensors {
				t.Log("loading", m.Name)
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
			fresh := func(t *testing.T) *Session {
				s, err := model.NewSession()
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { s.Close() })
				return s
			}
			t.Log("running full prefill")
			run := func(t *testing.T, session *Session, input []float32) []float32 {
				return runSampleAttentionLayer(t, session, input, layer)
			}
			all := run(t, fresh(t), attentionInputs(0, ref.Tokens, model.config.Dim))
			compareAttention(t, "full prefill", all, ref.Full, 2e-4)
			for _, c := range ref.Cases {
				t.Run(fmt.Sprintf("prefill_%d", c.Prefill), func(t *testing.T) {
					s, other := fresh(t), fresh(t)
					got := run(t, s, attentionInputs(0, c.Prefill, model.config.Dim))
					// Interleaved work on a distinct session must not change this cache.
					run(t, other, attentionInputs(17, 2, model.config.Dim))
					for pos := c.Prefill; pos < c.Prefill+2; pos++ {
						got = append(got, run(t, s, attentionInputs(pos, 1, model.config.Dim))...)
						run(t, other, attentionInputs(19+pos-c.Prefill, 1, model.config.Dim))
					}
					compareAttention(t, "reference cached", got, c.Output, 2e-4)
					compareAttention(t, "Go cached/full", got, all[:len(got)], 2e-4)
					for _, v := range got[:model.config.Dim] {
						if v != 0 {
							t.Fatal("first zero input attended to a future token")
						}
					}
					cacheState := &s.layers[layer]
					if s.offset != c.Prefill+2 || cacheState.windowLen != min(c.Prefill+2, 128) {
						t.Fatal("incorrect cache position/length")
					}
					cache, err := cacheState.window.Float32Data()
					if err != nil {
						t.Fatal(err)
					}
					compareAttention(t, "chronological KV cache", cache, c.Cache, 5e-5)
					if layer == 2 {
						if cacheState.compressedLen != s.offset/2 || cacheState.pendingLen != s.offset%2 {
							t.Fatal("incorrect compressed cache length")
						}
						for _, part := range []struct {
							name  string
							array mlx.Array
							want  []float32
						}{
							{"compressed KV", cacheState.compressed, c.Compressed}, {"index keys", cacheState.keys, c.Keys}, {"pending input", cacheState.pending, c.Pending},
						} {
							if len(part.want) == 0 {
								continue
							}
							data, err := part.array.Float32Data()
							if err != nil {
								t.Fatal(err)
							}
							compareAttention(t, part.name, data, part.want, 5e-5)
						}
					}
				})
			}
			if layer == 2 {
				testReleasedIndexSelection(t, model, ref)
			}
		})
	}
}

func testReleasedIndexSelection(t *testing.T, model *Model, ref attentionReference) {
	session, err := model.NewSession()
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	x, err := mlx.NewFloat32(attentionInputs(0, 1280, 5120), []int{1, 1280, 5120})
	if err != nil {
		t.Fatal(err)
	}
	defer x.Close()
	allKeys, err := one(func(s *scope) mlx.Array {
		w := model.weights
		normal := s.add(mlx.RMSNorm(x, w["layers.2.attn_norm.weight"], model.config.Hyper.NormEpsilon))
		latent, complete := session.compress(s, normal, 2)
		if complete != 640 {
			s.err = fmt.Errorf("expected 640 compressed groups, got %d", complete)
			return mlx.Array{}
		}
		key := s.add(mlx.RMSNorm(s.linear(latent, w["layers.2.attn.indexer.wk.weight"]), w["layers.2.attn.indexer.k_norm.weight"], model.config.Hyper.NormEpsilon))
		return rotary(s, key, model.config, 2, 0, 2, false)
	})
	if err != nil {
		t.Fatal(err)
	}
	defer allKeys.Close()
	keyData, err := allKeys.Float32Data()
	if err != nil {
		t.Fatal(err)
	}
	compareAttention(t, "long compressor/index keys", keyData, ref.IndexKeys, 5e-5)
	for _, probe := range ref.IndexCases {
		t.Run(fmt.Sprintf("index_at_%d", probe.Position), func(t *testing.T) {
			count := probe.Position / 2
			keys, err := one(func(s *scope) mlx.Array { return s.span(allKeys, 1, 0, count) })
			if err != nil {
				t.Fatal(err)
			}
			defer keys.Close()
			session := &Session{model: model, offset: probe.Position}
			x, err := mlx.NewFloat32(attentionInputs(probe.Position, 1, 5120), []int{1, 1, 5120})
			if err != nil {
				t.Fatal(err)
			}
			defer x.Close()
			ids, err := one(func(s *scope) mlx.Array {
				w := model.weights
				x = s.add(mlx.RMSNorm(x, w["layers.2.attn_norm.weight"], model.config.Hyper.NormEpsilon))
				qr := s.add(mlx.RMSNorm(s.linear(x, w["layers.2.attn.wq_a.weight"]), w["layers.2.attn.q_norm.weight"], model.config.Hyper.NormEpsilon))
				return session.index(s, x, qr, 2, &attentionState{owner: &layerCache{keys: keys, compressedLen: count}})
			})
			if err != nil {
				t.Fatal(err)
			}
			defer ids.Close()
			got, err := ids.Int32Data()
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got, probe.Indices) {
				t.Fatalf("top-512 selection mismatch: got %v want %v", got, probe.Indices)
			}
			t.Logf("%d selected indices match exactly from %d real-weight-derived keys", len(got), count)
		})
	}
}
