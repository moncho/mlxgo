//go:build mlx && mlxruntime

package deepseek

import (
	"fmt"
	"testing"

	mlx "github.com/moncho/mlxgo"
)

// This composes only attention sublayers. It deliberately does not simulate
// the missing mHC, residual and FFN operations of a full transformer block.
func runSharedAttention(t *testing.T, session *Session, input []float32) (owner, consumer []float32) {
	t.Helper()
	err := mlx.Batch(func() error {
		c, w := session.model.config, session.model.weights
		n := len(input) / c.Dim
		x, err := mlx.NewFloat32(input, []int{1, n, c.Dim})
		if err != nil {
			return err
		}
		defer x.Close()
		y, err := compute(func(s *scope) []mlx.Array {
			shared := &attentionState{}
			normal := s.add(mlx.RMSNorm(x, w["layers.2.attn_norm.weight"], c.Hyper.NormEpsilon))
			first := session.attention(s, normal, 2, shared)
			if s.err != nil {
				return nil
			}
			before, indices := session.layers[2], shared.topk
			normal = s.add(mlx.RMSNorm(first, w["layers.3.attn_norm.weight"], c.Hyper.NormEpsilon))
			second := session.attention(s, normal, 3, shared)
			if shared.owner != &session.layers[2] || shared.topk != indices || session.layers[2] != before {
				s.err = fmt.Errorf("consumer replaced owner cache or shared selection")
			}
			consumerCache := session.layers[3]
			if consumerCache.compressedLen != 0 || consumerCache.pendingLen != 0 || consumerCache.compressed != (mlx.Array{}) || consumerCache.keys != (mlx.Array{}) || consumerCache.pending != (mlx.Array{}) {
				s.err = fmt.Errorf("consumer allocated its own compressed state")
			}
			return []mlx.Array{first, second}
		})
		if err != nil {
			return err
		}
		defer mlx.CloseArrays(y)
		arrays := append(append([]mlx.Array{}, y...), session.layers[2].arrays()...)
		arrays = append(arrays, session.layers[3].arrays()...)
		if err := mlx.Eval(arrays...); err != nil {
			return err
		}
		owner, err = y[0].Float32Data()
		if err != nil {
			return err
		}
		consumer, err = y[1].Float32Data()
		if err == nil {
			session.offset += n
		}
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	return owner, consumer
}

func TestReleasedSharedAttentionForward(t *testing.T) {
	ownerDir, ownerRef, consumerDir, consumerRef := readSharedAttentionReferences(t)
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			model := &Model{config: consumerRef.Config.Config, weights: map[string]mlx.Array{}}
			t.Cleanup(func() { model.Close() })
			for _, part := range []struct {
				dir string
				ref attentionReference
			}{{ownerDir, ownerRef}, {consumerDir, consumerRef}} {
				for _, m := range part.ref.Tensors {
					a, err := mlx.NewFloat32(loadAttentionTensor(t, part.dir, m), m.Shape)
					if err != nil {
						t.Fatal(err)
					}
					model.weights[m.Name] = a
					data, err := a.Float32Data()
					if err != nil || matrixSHA(data) != m.DecodedSHA {
						t.Fatalf("native upload mismatch: %s: %v", m.Name, err)
					}
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
			fullOwner, fullConsumer := runSharedAttention(t, fresh(t), attentionInputs(0, 131, model.config.Dim))
			compareAttention(t, "owner full", fullOwner, ownerRef.Full, 2e-4)
			compareAttention(t, "consumer full", fullConsumer, consumerRef.Full, 2e-4)
			checkSchedule := func(t *testing.T, i int) {
				c, oc := consumerRef.Cases[i], ownerRef.Cases[i]
				s, other := fresh(t), fresh(t)
				owner, consumer := runSharedAttention(t, s, attentionInputs(0, c.Prefill, model.config.Dim))
				runSharedAttention(t, other, attentionInputs(17, 2, model.config.Dim))
				for pos := c.Prefill; pos < c.Prefill+2; pos++ {
					a, b := runSharedAttention(t, s, attentionInputs(pos, 1, model.config.Dim))
					owner, consumer = append(owner, a...), append(consumer, b...)
					runSharedAttention(t, other, attentionInputs(19+pos-c.Prefill, 1, model.config.Dim))
				}
				compareAttention(t, "owner cached", owner, oc.Output, 2e-4)
				compareAttention(t, "consumer cached", consumer, c.Output, 2e-4)
				compareAttention(t, "owner cached/full", owner, fullOwner[:len(owner)], 2e-4)
				compareAttention(t, "consumer cached/full", consumer, fullConsumer[:len(consumer)], 2e-4)
				for _, values := range [][]float32{owner, consumer} {
					for _, v := range values[:model.config.Dim] {
						if v != 0 {
							t.Fatal("zero first input saw future tokens")
						}
					}
				}
				end := c.Prefill + 2
				if s.offset != end || s.layers[2].windowLen != min(end, 128) || s.layers[3].windowLen != min(end, 128) || s.layers[2].compressedLen != end/2 || s.layers[2].pendingLen != end%2 {
					t.Fatal("incorrect shared cache lengths/position")
				}
				for _, part := range []struct {
					name  string
					array mlx.Array
					want  []float32
				}{
					{"owner window", s.layers[2].window, oc.Cache}, {"consumer window", s.layers[3].window, c.Cache},
					{"owner compressed", s.layers[2].compressed, oc.Compressed}, {"owner keys", s.layers[2].keys, oc.Keys}, {"owner pending", s.layers[2].pending, oc.Pending},
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
			for i, c := range consumerRef.Cases {
				t.Run(fmt.Sprintf("prefill_%d", c.Prefill), func(t *testing.T) { checkSchedule(t, i) })
			}
			t.Run("concurrent_sessions", func(t *testing.T) {
				for _, i := range []int{0, 4} {
					t.Run(fmt.Sprintf("prefill_%d", consumerRef.Cases[i].Prefill), func(t *testing.T) { t.Parallel(); checkSchedule(t, i) })
				}
			})
		})
	}
}
