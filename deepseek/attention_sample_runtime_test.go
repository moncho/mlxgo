//go:build mlx && mlxruntime

package deepseek

import (
	"fmt"
	"math"
	"os"
	"slices"
	"testing"

	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/deepseek/quant"
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

// Quantization makes the float32 input tolerance discontinuous at rounding
// thresholds. This is a bounded numerical compatibility test, NOT bitwise
// upstream parity. Exact identical-input encoding tests live in quant/.
func compareCacheAttention(t *testing.T, name string, got, want []float32, packed, cache bool) {
	t.Helper()
	if !packed {
		tol := 2e-4
		if cache {
			tol = 5e-5
		}
		compareAttention(t, name, got, want, tol)
		return
	}
	if len(got) != len(want) || len(got) == 0 {
		t.Fatal(name, "invalid comparison lengths")
	}
	var worst, squared float64
	different := 0
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("%s nonfinite value %d", name, i)
		}
		delta := math.Abs(float64(v)-float64(want[i])) / (1 + math.Abs(float64(want[i])))
		worst = max(worst, delta)
		squared += delta * delta
		if v != want[i] {
			different++
		}
	}
	rms := math.Sqrt(squared / float64(len(got)))
	t.Logf("%s: %d values max_scaled=%g rms_scaled=%g differing=%d", name, len(got), worst, rms, different)
	// Fixed experimental acceptance limits: no individual output differs by
	// more than 5% of (1+|reference|), aggregate RMS below 0.2%. Cache decoding
	// additionally permits different bins in at most 0.1% of entries.
	if worst > .05 || rms > .002 || (cache && float64(different)/float64(len(got)) > .001) {
		t.Fatalf("%s exceeds experimental quantized compatibility budget", name)
	}
}

func TestReleasedAttentionForward(t *testing.T) {
	testReleasedAttentionLayer(t, 0)
}

func TestReleasedCompressedAttentionForward(t *testing.T) {
	testReleasedAttentionLayer(t, 2)
}

func testReleasedAttentionLayer(t *testing.T, layer int) {
	testReleasedAttentionLayerMode(t, layer, false)
}

func TestReleasedQuantizedAttentionForward(t *testing.T) {
	if os.Getenv("MLXGO_DEEPSEEK_QUANTIZED_CACHES") != "1" {
		t.Skip("set MLXGO_DEEPSEEK_QUANTIZED_CACHES=1 with quantized-cache references")
	}
	for _, layer := range []int{0, 2} {
		t.Run(fmt.Sprintf("layer_%d", layer), func(t *testing.T) { testReleasedAttentionLayerMode(t, layer, true) })
	}
}

func testReleasedAttentionLayerMode(t *testing.T, layer int, packed bool) {
	testReleasedAttentionLayerOptions(t, layer, packed, fp8AttentionSelection{})
}

func TestReleasedFP8AttentionForward(t *testing.T) {
	testReleasedAttentionLayerOptions(t, 0, false, fp8AttentionSelection{kv: true})
}

func TestReleasedFP8QueryAttentionForward(t *testing.T) {
	for _, mode := range []struct {
		name      string
		selection fp8AttentionSelection
	}{
		{"qb", fp8AttentionSelection{qb: true}}, {"kv_qb", fp8AttentionSelection{kv: true, qb: true}},
	} {
		t.Run(mode.name, func(t *testing.T) { testReleasedAttentionLayerOptions(t, 0, false, mode.selection) })
	}
}

func TestReleasedFP8OutputAttentionForward(t *testing.T) {
	for _, mode := range []struct {
		name      string
		selection fp8AttentionSelection
	}{
		{"oa", fp8AttentionSelection{oa: true}}, {"kv_qb_oa", fp8AttentionSelection{kv: true, qb: true, oa: true}},
	} {
		t.Run(mode.name, func(t *testing.T) { testReleasedAttentionLayerOptions(t, 0, false, mode.selection) })
	}
}

func testReleasedAttentionLayerOptions(t *testing.T, layer int, packed bool, fp8 fp8AttentionSelection) {
	dir, ref := readAttentionLayerReferenceMode(t, layer, packed)
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			model := loadSampleAttentionModel(t, dir, ref, fp8)
			fresh := func(t *testing.T) *Session {
				s, err := model.NewSessionWithOptions(SessionOptions{QuantizedCaches: packed})
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
			compareCacheAttention(t, "full prefill", all, ref.Full, packed, false)
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
					compareCacheAttention(t, "reference cached", got, c.Output, packed, false)
					// Quantization can amplify prefill/decode accumulation differences
					// in the independent oracle too. Compare each schedule separately.
					if !packed {
						compareAttention(t, "Go cached/full", got, all[:len(got)], 2e-4)
					}
					for _, v := range got[:model.config.Dim] {
						if v != 0 {
							t.Fatal("first zero input attended to a future token")
						}
					}
					cacheState := &s.layers[layer]
					if s.offset != c.Prefill+2 || cacheState.windowLen != min(c.Prefill+2, 128) {
						t.Fatal("incorrect cache position/length")
					}
					checkCacheStorage(t, model.config, layer, *cacheState, packed)
					window, err := one(func(sc *scope) mlx.Array {
						return s.readAttentionCache(sc, cacheState.window, cacheState.windowScale, quant.FP8Activation32)
					})
					if err != nil {
						t.Fatal(err)
					}
					defer window.Close()
					cache, err := window.Float32Data()
					if err != nil {
						t.Fatal(err)
					}
					compareCacheAttention(t, "chronological KV cache", cache, c.Cache, packed, true)
					if layer == 2 {
						if cacheState.compressedLen != s.offset/2 || cacheState.pendingLen != s.offset%2 {
							t.Fatal("incorrect compressed cache length")
						}
						for _, part := range []struct {
							name   string
							array  mlx.Array
							scales mlx.Array
							format quant.ActivationFormat
							want   []float32
						}{
							{"compressed KV", cacheState.compressed, cacheState.compressedScale, quant.FP4Cache16, c.Compressed}, {"index keys", cacheState.keys, cacheState.keyScale, quant.FP4Index32, c.Keys}, {"pending input", cacheState.pending, mlx.Array{}, quant.FP8Activation32, c.Pending},
						} {
							if len(part.want) == 0 {
								continue
							}
							value, err := one(func(sc *scope) mlx.Array {
								if part.name == "pending input" {
									return part.array
								}
								return s.readAttentionCache(sc, part.array, part.scales, part.format)
							})
							if err != nil {
								t.Fatal(err)
							}
							data, err := value.Float32Data()
							value.Close()
							if err != nil {
								t.Fatal(err)
							}
							compareCacheAttention(t, part.name, data, part.want, packed && part.name != "pending input", true)
						}
					}
				})
			}
			if layer == 2 {
				testReleasedIndexSelection(t, model, ref, packed)
			}
		})
	}
}

func testReleasedIndexSelection(t *testing.T, model *Model, ref attentionReference, packed bool) {
	session, err := model.NewSessionWithOptions(SessionOptions{QuantizedCaches: packed})
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
		key = rotary(s, key, model.config, 2, 0, 2, false)
		if packed {
			d, sc := packCache(s, key, quant.FP4Index32)
			key = unpackCache(s, d, sc, quant.FP4Index32)
		}
		return key
	})
	if err != nil {
		t.Fatal(err)
	}
	defer allKeys.Close()
	keyData, err := allKeys.Float32Data()
	if err != nil {
		t.Fatal(err)
	}
	compareCacheAttention(t, "long compressor/index keys", keyData, ref.IndexKeys, packed, true)
	for _, probe := range ref.IndexCases {
		t.Run(fmt.Sprintf("index_at_%d", probe.Position), func(t *testing.T) {
			count := probe.Position / 2
			keys, err := one(func(s *scope) mlx.Array { return s.span(allKeys, 1, 0, count) })
			if err != nil {
				t.Fatal(err)
			}
			defer keys.Close()
			session := &Session{model: model, offset: probe.Position, options: SessionOptions{QuantizedCaches: packed}}
			x, err := mlx.NewFloat32(attentionInputs(probe.Position, 1, 5120), []int{1, 1, 5120})
			if err != nil {
				t.Fatal(err)
			}
			defer x.Close()
			ids, err := one(func(s *scope) mlx.Array {
				w := model.weights
				x = s.add(mlx.RMSNorm(x, w["layers.2.attn_norm.weight"], model.config.Hyper.NormEpsilon))
				qr := s.add(mlx.RMSNorm(s.linear(x, w["layers.2.attn.wq_a.weight"]), w["layers.2.attn.q_norm.weight"], model.config.Hyper.NormEpsilon))
				owner := &layerCache{keys: keys, compressedLen: count}
				if packed {
					owner.keys, owner.keyScale = packCache(s, keys, quant.FP4Index32)
				}
				return session.index(s, x, qr, 2, &attentionState{owner: owner})
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
