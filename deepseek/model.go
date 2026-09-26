package deepseek

import (
	"fmt"
	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/deepseek/quant"
	"github.com/moncho/mlxgo/lm"
	"slices"
	"sort"
)

// Model owns immutable parameters, with optional packed attention KV weights.
// Separate sessions may run from arbitrary goroutines. Close and native work
// are serialized on the MLX worker.
type Model struct {
	config         Config
	weights        map[string]mlx.Array
	fp8AttentionKV map[int]*quant.FP8Linear
	closed         bool
}

// NewModel validates and retains independent handles for all parameters. Inputs
// remain caller-owned. Extra/missing names, shapes and dtypes are rejected.
func NewModel(c Config, parameters map[string]mlx.Array) (m *Model, err error) {
	return NewModelWithOptions(c, parameters, ModelOptions{})
}

// FP8AttentionWeight contains E4M3FN row-major weights and E8M0 scales for
// 32x32 blocks. The logical shape is [HeadDim, Dim] from the model config.
// The constructor copies these bytes; callers must not mutate them during construction.
type FP8AttentionWeight struct {
	Data, Scales []byte
}

// ModelOptions selects experimental weight storage, independently of cache options.
type ModelOptions struct {
	// FP8AttentionKV replaces layers.<index>.attn.wkv.weight for selected layers.
	// Omit those names from parameters; duplicate float32 weights are rejected.
	// Only these projections use packed weight-only MXFP8; activations remain
	// float32. HeadDim and Dim must be divisible by 32. CPU execution can be
	// substantially slower. No automatic quantization or checkpoint loading occurs.
	FP8AttentionKV map[int]FP8AttentionWeight
}

// NewModelWithOptions retains the float32 parameters and copies explicitly
// supplied packed weights. Options are immutable after construction. An empty
// options value is equivalent to NewModel; no float32 duplicates are retained.
func NewModelWithOptions(c Config, parameters map[string]mlx.Array, options ModelOptions) (m *Model, err error) {
	c = c.clone()
	shapes, err := c.ParameterShapes()
	if err != nil {
		return nil, err
	}
	replacements := make(map[string]int, len(options.FP8AttentionKV))
	for layer := range options.FP8AttentionKV {
		if layer < 0 || layer >= c.Layers {
			return nil, fmt.Errorf("deepseek: FP8 attention layer %d outside model", layer)
		}
		name := fmt.Sprintf("layers.%d.attn.wkv.weight", layer)
		if _, exists := parameters[name]; exists {
			return nil, fmt.Errorf("deepseek: duplicate float32 and FP8 parameter %s", name)
		}
		replacements[name] = layer
	}
	if len(parameters)+len(replacements) != len(shapes) {
		return nil, fmt.Errorf("deepseek: expected %d parameters, got %d float32 and %d FP8", len(shapes), len(parameters), len(replacements))
	}
	m = &Model{config: c, weights: make(map[string]mlx.Array, len(shapes)), fp8AttentionKV: make(map[int]*quant.FP8Linear, len(replacements))}
	err = mlx.Batch(func() error {
		s := &scope{}
		keys := make([]string, 0, len(shapes))
		for k := range shapes {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, name := range keys {
			if layer, ok := replacements[name]; ok {
				if err := m.initFP8AttentionKV(layer, options.FP8AttentionKV[layer]); err != nil {
					return err
				}
				continue
			}
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

// Called only while constructing an unpublished model on the MLX worker.
func (m *Model) initFP8AttentionKV(layer int, weight FP8AttentionWeight) error {
	p, err := quant.NewFP8Linear(weight.Data, weight.Scales, m.config.HeadDim, m.config.Dim, quant.FP8Block32)
	if err != nil {
		return fmt.Errorf("deepseek: FP8 attention layer %d: %w", layer, err)
	}
	if m.fp8AttentionKV == nil {
		m.fp8AttentionKV = make(map[int]*quant.FP8Linear)
	}
	m.fp8AttentionKV[layer] = p
	return nil
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
		err := mlx.CloseArrays(arrays)
		for _, p := range m.fp8AttentionKV {
			if e := p.Close(); err == nil {
				err = e
			}
		}
		m.fp8AttentionKV = nil
		return err
	})
}

type layerCache struct {
	window, pending, compressed, keys      mlx.Array
	windowScale, compressedScale, keyScale mlx.Array
	windowLen, pendingLen, compressedLen   int
}

func (c *layerCache) arrays() []mlx.Array {
	var out []mlx.Array
	for _, a := range []mlx.Array{c.window, c.pending, c.compressed, c.keys, c.windowScale, c.compressedScale, c.keyScale} {
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
	engram          *EngramHasher
	options         SessionOptions
}

// SessionOptions controls per-sequence inference behavior, not checkpoint format.
type SessionOptions struct {
	// QuantizedCaches stores window KV as FP8 and compressed KV/index keys as
	// packed FP4, and quantizes index queries before scoring. Activations and
	// reconstructed values remain float32 and are not BF16-rounded. Projection
	// weight storage is controlled separately by ModelOptions.
	// This is an experimental reference path, not quantized GEMM or BF16/CUDA
	// numerical parity. It requires HeadDim and (when used) IndexDim divisible
	// by 32. No tensor values round-trip through Go.
	QuantizedCaches bool
}

var _ lm.Session = (*Session)(nil)

func (m *Model) NewSession() (session *Session, err error) {
	return m.NewSessionWithOptions(SessionOptions{})
}

// NewSessionWithOptions creates a sequence with immutable cache options.
// NewSession retains the existing float32-cache behavior.
func (m *Model) NewSessionWithOptions(options SessionOptions) (session *Session, err error) {
	err = mlx.Batch(func() error {
		if m == nil || m.closed {
			return fmt.Errorf("deepseek: closed model")
		}
		if options.QuantizedCaches {
			if m.config.HeadDim <= 0 || m.config.HeadDim%32 != 0 {
				return fmt.Errorf("deepseek: quantized caches require head_dim divisible by 32")
			}
			if len(m.config.KVSources) > 0 && (m.config.IndexDim <= 0 || m.config.IndexDim%32 != 0) {
				return fmt.Errorf("deepseek: quantized caches require index_head_dim divisible by 32")
			}
		}
		session = &Session{model: m, layers: make([]layerCache, m.config.Layers), options: options}
		if m.config.Engram != nil {
			// Model config is already validated and immutable; share metadata, not history.
			session.engram = &EngramHasher{config: *m.config.Engram}
		}
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
		if s.engram != nil {
			s.engram.Reset()
		}
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
		var hashes []int32
		if session.engram != nil {
			hashes, e = session.engram.Hash(tokens, nil)
			if e != nil {
				return e
			}
		}
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
				if c.Engram != nil {
					if index := slices.Index(c.Engram.Layers, i); index >= 0 {
						cols, layers := c.Engram.columns(), len(c.Engram.Layers)
						ids := make([]int32, n*cols)
						for token := 0; token < n; token++ {
							copy(ids[token*cols:], hashes[(token*layers+index)*cols:(token*layers+index+1)*cols])
						}
						p := fmt.Sprintf("layers.%d.engram.", i)
						x = s.add(Engram(x, ids, nil, EngramWeights{w[p+"embed.weight"], w[p+"wkv.weight"], w[p+"q_weight"], w[p+"k_weight"]}, c.Hyper.NormEpsilon))
						if s.err != nil {
							return mlx.Array{}
						}
					}
				}
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
