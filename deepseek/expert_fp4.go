package deepseek

import (
	"fmt"
	"math"

	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/deepseek/quant"
)

// FP4Weight contains row-major E2M1 nibbles (earlier element in the low nibble)
// and one E8M0 scale per row per 32 columns. Constructors copy both slices.
type FP4Weight struct{ Data, Scales []byte }

// FP4ExpertWeights holds checkpoint-oriented projections: Gate/Up [inter,dim]
// and Down [dim,inter]. This is row-scaled FP4, not FP4 cache encoding.
type FP4ExpertWeights struct{ Gate, Up, Down FP4Weight }

// ExpertID selects a layer and routed expert. Index -1 selects the shared expert.
type ExpertID struct{ Layer, Index int }

// FP4Expert owns a weight-only packed SwiGLU expert. It keeps activations float32
// and is inference-only. Copies share handles; Close invalidates every copy.
// Concurrent calls are serialized on the MLX worker. No full-checkpoint loading
// or DeepSeek activation-quantized GEMM is implied.
type FP4Expert struct {
	gate, up, down *quant.FP4Linear
	dim, inter     int
}

// NewFP4Expert copies three packed matrices. dim and inter must be positive
// multiples of 32. Malformed or nonfinite decoded weights are rejected, and
// partial native allocations are released on failure. No float32 copy is kept.
func NewFP4Expert(w FP4ExpertWeights, dim, inter int) (*FP4Expert, error) {
	p := &FP4Expert{dim: dim, inter: inter}
	err := mlx.Batch(func() error {
		var err error
		p.gate, err = quant.NewFP4Linear(w.Gate.Data, w.Gate.Scales, inter, dim)
		if err != nil {
			return err
		}
		p.up, err = quant.NewFP4Linear(w.Up.Data, w.Up.Scales, inter, dim)
		if err != nil {
			return err
		}
		p.down, err = quant.NewFP4Linear(w.Down.Data, w.Down.Scales, dim, inter)
		return err
	})
	if err != nil {
		p.Close()
		return nil, fmt.Errorf("deepseek: FP4 expert: %w", err)
	}
	return p, nil
}

// Forward takes Float32 x [tokens,dim] and routing [tokens,1], returning a
// caller-owned lazy [tokens,dim] output. Positive limit clips gate from above
// and up on both sides; zero disables clipping. Routing is applied before down.
func (p *FP4Expert) Forward(x, routing mlx.Array, limit float32) (mlx.Array, error) {
	return one(func(s *scope) mlx.Array {
		if p == nil || limit < 0 || math.IsNaN(float64(limit)) || math.IsInf(float64(limit), 0) {
			s.err = fmt.Errorf("deepseek: nil FP4 expert or invalid SwiGLU limit")
			return mlx.Array{}
		}
		s.check(x, "FP4 expert input", -1, p.dim)
		if s.err != nil {
			return mlx.Array{}
		}
		s.check(routing, "FP4 expert routing", x.Shape()[0], 1)
		if s.err != nil {
			return mlx.Array{}
		}
		gate, up := s.add(p.gate.Forward(x)), s.add(p.up.Forward(x))
		return s.add(p.down.Forward(expertActivation(s, gate, up, routing, limit)))
	})
}

// Close is idempotent. Lazy outputs retain their dependencies after close.
func (p *FP4Expert) Close() error {
	if p == nil {
		return nil
	}
	return mlx.Batch(func() error {
		var err error
		for _, l := range []*quant.FP4Linear{p.gate, p.up, p.down} {
			if e := l.Close(); err == nil {
				err = e
			}
		}
		return err
	})
}

func expertPrefix(id ExpertID) string {
	if id.Index == -1 {
		return fmt.Sprintf("layers.%d.ffn.shared_experts.", id.Layer)
	}
	return fmt.Sprintf("layers.%d.ffn.experts.%d.", id.Layer, id.Index)
}

func (m *Model) packedMoE(s *scope, x mlx.Array, layer int) mlx.Array {
	c, w := m.config, m.weights
	p := fmt.Sprintf("layers.%d.ffn.", layer)
	weights, indices := route(s, x, w[p+"gate.weight"], w[p+"gate.bias"], c.Router)
	if s.err != nil {
		return mlx.Array{}
	}
	run := func(index int, amplitude mlx.Array) mlx.Array {
		name := expertPrefix(ExpertID{Layer: layer, Index: index})
		if e := m.fp4Experts[name]; e != nil {
			if index == -1 {
				amplitude = s.add(mlx.Ones([]int{x.Shape()[0], 1}, mlx.Float32))
			}
			return s.add(e.Forward(x, amplitude, c.Limit))
		}
		return expert(s, x, ExpertWeights{Gate: w[name+"w1.weight"], Up: w[name+"w3.weight"], Down: w[name+"w2.weight"]}, amplitude, c.Limit)
	}
	out := run(-1, s.scalar(1))
	for i := 0; i < c.Experts; i++ {
		if s.err != nil {
			return mlx.Array{}
		}
		mask := s.add(mlx.Equal(indices, s.add(mlx.NewScalarInt(i))))
		selected := s.add(mlx.Where(mask, weights, s.scalar(0)))
		amplitude := s.add(mlx.SumAxis(selected, -1, true))
		out = s.add(mlx.Add(out, run(i, amplitude)))
	}
	return out
}
