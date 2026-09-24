//go:build mlx && mlxruntime

package deepseek

import (
	"fmt"
	"math"
	"testing"

	mlx "github.com/moncho/mlxgo"
)

// Exercise the existing Session.attention implementation, including its real
// cache management. No alternative Go attention implementation is used here.
func runSampleAttention(t *testing.T, session *Session, input []float32) []float32 {
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
			normal := s.add(mlx.RMSNorm(x, session.model.weights["layers.0.attn_norm.weight"], session.model.config.Hyper.NormEpsilon))
			return session.attention(s, normal, 0, &attentionState{})
		})
		if err != nil {
			return err
		}
		defer y.Close()
		if err = mlx.Eval(append([]mlx.Array{y}, session.layers[0].arrays()...)...); err != nil {
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
	dir, ref := readAttentionReference(t)
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
			all := runSampleAttention(t, fresh(t), attentionInputs(0, ref.Tokens, model.config.Dim))
			compareAttention(t, "full prefill", all, ref.Full, 2e-4)
			for _, c := range ref.Cases {
				t.Run(fmt.Sprintf("prefill_%d", c.Prefill), func(t *testing.T) {
					s, other := fresh(t), fresh(t)
					got := runSampleAttention(t, s, attentionInputs(0, c.Prefill, model.config.Dim))
					// Interleaved work on a distinct session must not change this cache.
					runSampleAttention(t, other, attentionInputs(17, 2, model.config.Dim))
					for pos := c.Prefill; pos < c.Prefill+2; pos++ {
						got = append(got, runSampleAttention(t, s, attentionInputs(pos, 1, model.config.Dim))...)
						runSampleAttention(t, other, attentionInputs(19+pos-c.Prefill, 1, model.config.Dim))
					}
					compareAttention(t, "reference cached", got, c.Output, 2e-4)
					compareAttention(t, "Go cached/full", got, all[:len(got)], 2e-4)
					for _, v := range got[:model.config.Dim] {
						if v != 0 {
							t.Fatal("first zero input attended to a future token")
						}
					}
					if s.offset != c.Prefill+2 || s.layers[0].windowLen != min(c.Prefill+2, 128) {
						t.Fatal("incorrect cache position/length")
					}
					cache, err := s.layers[0].window.Float32Data()
					if err != nil {
						t.Fatal(err)
					}
					compareAttention(t, "chronological KV cache", cache, c.Cache, 5e-5)
				})
			}
		})
	}
}
