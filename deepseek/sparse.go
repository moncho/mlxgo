package deepseek

import (
	"fmt"
	"math"

	mlx "github.com/moncho/mlxgo"
)

type expertBatch struct{ rows, slots []int32 }

// Slots index flattened [token,topK] assignments. Inverse restores that order
// after concatenating expert-major outputs, without a dense tokens*experts mask.
func planExperts(ids []int32, tokens, k, experts int) ([]expertBatch, []int32, error) {
	if tokens <= 0 || k <= 0 || experts < k || tokens > math.MaxInt32/k || len(ids) != tokens*k {
		return nil, nil, fmt.Errorf("deepseek: invalid sparse routing dimensions")
	}
	groups := make([]expertBatch, experts)
	for slot, id := range ids {
		if id < 0 || int(id) >= experts {
			return nil, nil, fmt.Errorf("deepseek: sparse expert index out of range")
		}
		for j := slot - slot%k; j < slot; j++ {
			if ids[j] == id {
				return nil, nil, fmt.Errorf("deepseek: duplicate expert for token")
			}
		}
		g := &groups[id]
		g.rows = append(g.rows, int32(slot/k))
		g.slots = append(g.slots, int32(slot))
	}
	inverse := make([]int32, len(ids))
	var offset int32
	for _, g := range groups {
		for _, slot := range g.slots {
			inverse[slot] = offset
			offset++
		}
	}
	return groups, inverse, nil
}

type expertRunner func(*scope, int, mlx.Array, mlx.Array) mlx.Array

// SparseMoE computes only selected token/expert pairs, plus the shared expert.
// It reads routing indices to Go once per call; activations and routing weights
// stay on the device. This eager scheduling boundary is inference-only and must
// not be used inside Compile or autograd callbacks. MoE remains the lazy,
// differentiable reference. Small/dense batches can be slower on this path.
func SparseMoE(x, gateWeight, gateBias mlx.Array, experts []ExpertWeights, shared ExpertWeights, c RouterConfig, limit float32) (mlx.Array, error) {
	return one(func(s *scope) mlx.Array {
		if limit < 0 || math.IsNaN(float64(limit)) || math.IsInf(float64(limit), 0) {
			s.err = fmt.Errorf("deepseek: invalid SwiGLU limit")
			return mlx.Array{}
		}
		weights, indices := route(s, x, gateWeight, gateBias, c)
		if s.err != nil {
			return mlx.Array{}
		}
		if len(experts) != gateWeight.Shape()[0] {
			s.err = fmt.Errorf("deepseek: expert count differs from router")
			return mlx.Array{}
		}
		return sparseExperts(s, x, weights, indices, len(experts), func(s *scope, id int, input, amplitude mlx.Array) mlx.Array {
			w := shared
			if id >= 0 {
				w = experts[id]
			}
			return expert(s, input, w, amplitude, limit)
		})
	})
}

func sparseExperts(s *scope, x, weights, indices mlx.Array, experts int, run expertRunner) mlx.Array {
	if s.err != nil {
		return mlx.Array{}
	}
	shape := x.Shape()
	k := indices.Shape()[1]
	ids, err := s.add(mlx.AsType(indices, mlx.Int32)).Int32Data()
	if err != nil {
		s.err = err
		return mlx.Array{}
	}
	groups, inverse, err := planExperts(ids, shape[0], k, experts)
	if err != nil {
		s.err = err
		return mlx.Array{}
	}
	flatWeights := s.add(mlx.Reshape(weights, []int{len(ids), 1}))
	parts := make([]mlx.Array, 0, min(experts, len(ids)))
	for id, g := range groups {
		if len(g.rows) == 0 {
			continue
		}
		rows := s.add(mlx.NewInt32(g.rows, []int{len(g.rows)}))
		slots := s.add(mlx.NewInt32(g.slots, []int{len(g.slots)}))
		input := s.add(mlx.TakeAxis(x, rows, 0))
		amplitude := s.add(mlx.TakeAxis(flatWeights, slots, 0))
		parts = append(parts, run(s, id, input, amplitude))
		if s.err != nil {
			return mlx.Array{}
		}
	}
	joined := s.add(mlx.ConcatenateAxis(parts, 0))
	order := s.add(mlx.NewInt32(inverse, []int{len(inverse)}))
	ordered := s.add(mlx.TakeAxis(joined, order, 0))
	ordered = s.add(mlx.Reshape(ordered, []int{shape[0], k, shape[1]}))
	routed := s.add(mlx.SumAxis(ordered, 1, false))
	shared := run(s, -1, x, s.add(mlx.Ones([]int{shape[0], 1}, mlx.Float32)))
	return s.add(mlx.Add(shared, routed))
}

func (m *Model) sparseMoE(s *scope, x mlx.Array, layer int) mlx.Array {
	p := fmt.Sprintf("layers.%d.ffn.", layer)
	weights, indices := route(s, x, m.weights[p+"gate.weight"], m.weights[p+"gate.bias"], m.config.Router)
	return sparseExperts(s, x, weights, indices, m.config.Experts, func(s *scope, id int, input, amplitude mlx.Array) mlx.Array {
		name := expertPrefix(ExpertID{Layer: layer, Index: id})
		if e := m.fp4Experts[name]; e != nil {
			return s.add(e.Forward(input, amplitude, m.config.Limit))
		}
		return expert(s, input, ExpertWeights{Gate: m.weights[name+"w1.weight"], Up: m.weights[name+"w3.weight"], Down: m.weights[name+"w2.weight"]}, amplitude, m.config.Limit)
	})
}
