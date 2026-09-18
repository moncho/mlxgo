package deepseek

import (
	"fmt"
	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/lm"
	"slices"
	"sort"
)

// Model owns immutable float32 parameters. Separate sessions may run from
// arbitrary goroutines. Close and native work are serialized on the MLX worker.
type Model struct {
	config  Config
	weights map[string]mlx.Array
	closed  bool
}

// NewModel validates and retains independent handles for all parameters. Inputs
// remain caller-owned. Extra/missing names, shapes and dtypes are rejected.
func NewModel(c Config, parameters map[string]mlx.Array) (m *Model, err error) {
	c = c.clone()
	shapes, err := c.ParameterShapes()
	if err != nil {
		return nil, err
	}
	if len(parameters) != len(shapes) {
		return nil, fmt.Errorf("deepseek: expected %d parameters, got %d", len(shapes), len(parameters))
	}
	m = &Model{config: c, weights: make(map[string]mlx.Array, len(shapes))}
	err = mlx.Batch(func() error {
		s := &scope{}
		keys := make([]string, 0, len(shapes))
		for k := range shapes {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, name := range keys {
			a, ok := parameters[name]
			if !ok {
				return fmt.Errorf("deepseek: missing parameter %s", name)
			}
			s.check(a, name, shapes[name]...)
			if s.err != nil {
				return s.err
			}
			owned, e := mlx.Reshape(a, shapes[name])
			if e != nil {
				return e
			}
			m.weights[name] = owned
		}
		return nil
	})
	if err != nil {
		m.Close()
		return nil, err
	}
	return m, nil
}

func (m *Model) Close() error {
	if m == nil {
		return nil
	}
	return mlx.Batch(func() error {
		if m.closed {
			return nil
		}
		m.closed = true
		var arrays []mlx.Array
		for _, a := range m.weights {
			arrays = append(arrays, a)
		}
		m.weights = nil
		return mlx.CloseArrays(arrays)
	})
}

type layerCache struct {
	window, pending, compressed, keys    mlx.Array
	windowLen, pendingLen, compressedLen int
}

func (c *layerCache) arrays() []mlx.Array {
	var out []mlx.Array
	for _, a := range []mlx.Array{c.window, c.pending, c.compressed, c.keys} {
		if len(a.Shape()) > 0 {
			out = append(out, a)
		}
	}
	return out
}

// Session owns one sequence's caches, including shared compressed KV/index
// sources and partial compression groups. The first call accepts any nonempty
// prompt; subsequent calls accept exactly one token. A failed native evaluation
// invalidates the session. Validation errors do not advance or invalidate it.
type Session struct {
	model           *Model
	layers          []layerCache
	offset          int
	invalid, closed bool
}

var _ lm.Session = (*Session)(nil)

func (m *Model) NewSession() (session *Session, err error) {
	err = mlx.Batch(func() error {
		if m == nil || m.closed {
			return fmt.Errorf("deepseek: closed model")
		}
		session = &Session{model: m, layers: make([]layerCache, m.config.Layers)}
		return nil
	})
	return session, err
}
func (s *Session) Position() int {
	if s == nil {
		return 0
	}
	return s.offset
}
func (s *Session) Close() error {
	if s == nil {
		return nil
	}
	return mlx.Batch(func() error {
		if s.closed {
			return nil
		}
		s.closed = true
		var arrays []mlx.Array
		for i := range s.layers {
			arrays = append(arrays, s.layers[i].arrays()...)
		}
		return mlx.CloseArrays(arrays)
	})
}

// Forward computes all logits [1,len(tokens),vocab] in a fresh temporary session.
func (m *Model) Forward(tokens []int32) (mlx.Array, error) {
	s, err := m.NewSession()
	if err != nil {
		return mlx.Array{}, err
	}
	defer s.Close()
	return s.ForwardAll(tokens)
}

// Step implements lm.Session and returns only the final token's logits.
func (s *Session) Step(tokens []int32) (mlx.Array, error) {
	all, err := s.ForwardAll(tokens)
	if err != nil {
		return mlx.Array{}, err
	}
	defer all.Close()
	return one(func(scope *scope) mlx.Array { return scope.span(all, 1, len(tokens)-1, len(tokens)) })
}

// ForwardAll consumes tokens and returns logits for every consumed position.
// This float32 inference path evaluates results and caches before advancing.
func (session *Session) ForwardAll(tokens []int32) (out mlx.Array, err error) {
	err = mlx.Batch(func() error {
		if session == nil || session.closed || session.invalid || session.model.closed {
			return fmt.Errorf("deepseek: closed or invalid session/model")
		}
		c := session.model.config
		if len(tokens) == 0 || (session.offset > 0 && len(tokens) != 1) || len(tokens) > c.MaxSeq-session.offset {
			return fmt.Errorf("deepseek: expected nonempty prefill or one decode token within %d positions", c.MaxSeq)
		}
		for _, id := range tokens {
			if id < 0 || int(id) >= c.VocabSize {
				return fmt.Errorf("deepseek: token %d out of vocabulary", id)
			}
		}
		var e error
		out, e = one(func(s *scope) mlx.Array {
			w := session.model.weights
			n := len(tokens)
			hc := c.Hyper.Streams
			ids := s.add(mlx.NewInt32(tokens, []int{1, n}))
			x := s.add(mlx.TakeAxis(w["embed.weight"], ids, 0))
			x = s.add(mlx.ExpandDims(x, 2))
			x = s.add(mlx.BroadcastTo(x, []int{1, n, hc, c.Dim}))
			mix := make([]float32, n*hc)
			for i := 0; i < n; i++ {
				mix[i*hc] = 1
			}
			pre := s.add(mlx.NewFloat32(mix, []int{1, n, hc}))
			shared := &attentionState{}
			for i := 0; i < c.Layers; i++ {
				x, pre = session.block(s, x, pre, i, shared)
				if s.err != nil {
					return mlx.Array{}
				}
			}
			x = s.add(HyperPre(x, pre))
			x = s.add(mlx.RMSNorm(x, w["norm.weight"], c.Hyper.NormEpsilon))
			return s.linear(x, w["head.weight"])
		})
		if e != nil {
			session.invalid = true
			return e
		}
		arrays := []mlx.Array{out}
		for i := range session.layers {
			arrays = append(arrays, session.layers[i].arrays()...)
		}
		if e = mlx.Eval(arrays...); e != nil {
			out.Close()
			out = mlx.Array{}
			session.invalid = true
			return e
		}
		session.offset += len(tokens)
		return nil
	})
	return out, err
}

func (session *Session) block(s *scope, x, previous mlx.Array, layer int, shared *attentionState) (mlx.Array, mlx.Array) {
	c, w := session.model.config, session.model.weights
	p := fmt.Sprintf("layers.%d.", layer)
	mixes := func(x mlx.Array, part string) (mlx.Array, mlx.Array, mlx.Array) {
		pre, post, comb, err := HyperMix(x, w[p+"hc_"+part+"_fn"], w[p+"hc_"+part+"_scale"], w[p+"hc_"+part+"_base"], c.Hyper)
		return s.add(pre, err), s.add(post, err), s.add(comb, err)
	}
	attnPre, post, comb := mixes(x, "attn")
	h := s.add(HyperPre(x, previous))
	h = s.add(mlx.RMSNorm(h, w[p+"attn_norm.weight"], c.Hyper.NormEpsilon))
	h = session.attention(s, h, layer, shared)
	x = s.add(HyperPost(h, x, post, comb))
	if s.err != nil {
		return mlx.Array{}, mlx.Array{}
	}
	ffnPre, post, comb := mixes(x, "ffn")
	h = s.add(HyperPre(x, attnPre))
	h = s.add(mlx.RMSNorm(h, w[p+"ffn_norm.weight"], c.Hyper.NormEpsilon))
	if s.err != nil {
		return mlx.Array{}, mlx.Array{}
	}
	n := h.Shape()[1]
	h = s.add(mlx.Reshape(h, []int{n, c.Dim}))
	expert := func(name string) ExpertWeights {
		return ExpertWeights{Gate: w[p+name+"w1.weight"], Up: w[p+name+"w3.weight"], Down: w[p+name+"w2.weight"]}
	}
	experts := make([]ExpertWeights, c.Experts)
	for i := range experts {
		experts[i] = expert(fmt.Sprintf("ffn.experts.%d.", i))
	}
	h = s.add(MoE(h, w[p+"ffn.gate.weight"], w[p+"ffn.gate.bias"], experts, expert("ffn.shared_experts."), c.Router, c.Limit))
	h = s.add(mlx.Reshape(h, []int{1, n, c.Dim}))
	return s.add(HyperPost(h, x, post, comb)), ffnPre
}

// Replacing cache handles retains the graph but never takes ownership of a
// layer scope's handle. On error the whole session is marked unusable.
func retain(s *scope, dst *mlx.Array, value mlx.Array) {
	if s.err != nil {
		return
	}
	owned, err := mlx.Reshape(value, value.Shape())
	if err != nil {
		s.err = err
		return
	}
	_ = dst.Close()
	*dst = owned
}
func appendCache(s *scope, dst *mlx.Array, value mlx.Array, count int) {
	if count > 0 {
		value = s.add(mlx.ConcatenateAxis([]mlx.Array{*dst, value}, 1))
	}
	retain(s, dst, value)
}

func source(c Config, layer int) bool { return slices.Contains(c.KVSources, layer) }
