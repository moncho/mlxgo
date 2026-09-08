package qwen2

import (
	"fmt"
	"math"

	"github.com/moncho/mlxgo"
)

// scope records the first error and owns every intermediate handle in one layer.
type scope struct {
	arrays []mlx.Array
	err    error
}

func (s *scope) add(a mlx.Array, err error) mlx.Array {
	if err != nil && s.err == nil {
		s.err = err
	}
	if err == nil {
		s.arrays = append(s.arrays, a)
	}
	return a
}
func (s *scope) close() { _ = mlx.CloseArrays(s.arrays) }
func (s *scope) take(a mlx.Array) (mlx.Array, error) {
	if s.err != nil {
		return mlx.Array{}, s.err
	}
	// The result is always the most recently allocated array.
	s.arrays = s.arrays[:len(s.arrays)-1]
	return a, nil
}

// Forward returns [1,1,vocab] logits for the last token. It advances cache on
// success; a nil cache computes an uncached full causal pass. Inputs are borrowed.
func Forward(w *Weights, c Config, tokens []int32, cache *KVCache) (mlx.Array, error) {
	return forward(w, c, tokens, cache, nil, false)
}

// ForwardAll returns [1,length,vocab] logits, suitable for next-token training.
func ForwardAll(w *Weights, c Config, tokens []int32) (mlx.Array, error) {
	return forward(w, c, tokens, nil, nil, true)
}

// projectionHook supplies optional trainable additions to q/v projections.
type projectionHook func(layer int, kind string, x, base mlx.Array) (mlx.Array, error)

func forward(w *Weights, c Config, tokens []int32, cache *KVCache, hook projectionHook, all bool) (out mlx.Array, err error) {
	if err := c.Validate(); err != nil {
		return out, err
	}
	if w == nil || w.closed || len(w.Layers) != c.NumLayers {
		return out, fmt.Errorf("qwen2: closed or incompatible weights")
	}
	if len(tokens) == 0 {
		return out, fmt.Errorf("qwen2: tokens must not be empty")
	}
	offset := 0
	if cache != nil {
		if cache.invalid || len(cache.keys) != c.NumLayers {
			return out, fmt.Errorf("qwen2: invalid cache")
		}
		offset = cache.Offset
	}
	if len(tokens) > c.MaxPositions-offset {
		return out, fmt.Errorf("qwen2: context exceeds %d positions", c.MaxPositions)
	}
	for _, id := range tokens {
		if id < 0 || int(id) >= c.VocabSize {
			return out, fmt.Errorf("qwen2: token %d outside vocabulary", id)
		}
	}
	err = mlx.Batch(func() error {
		if w.swiglu == nil {
			var e error
			w.swiglu, e = mlx.Compile(func(in []mlx.Array) ([]mlx.Array, error) {
				if len(in) != 2 {
					return nil, fmt.Errorf("qwen2: SwiGLU expects two inputs")
				}
				gate, e := mlx.SiLU(in[0])
				if e != nil {
					return nil, e
				}
				defer gate.Close()
				out, e := mlx.Multiply(gate, in[1])
				if e != nil {
					return nil, e
				}
				return []mlx.Array{out}, nil
			}, true)
			if e != nil {
				return e
			}
		}
		s := &scope{}
		defer s.close()
		ids := s.add(mlx.NewInt32(tokens, []int{1, len(tokens)}))
		x := s.add(mlx.TakeAxis(w.Embed, ids, 0))
		if s.err != nil {
			return s.err
		}
		for i, l := range w.Layers {
			next, e := layerForward(x, l, c, i, len(tokens), offset, cache, hook, w.swiglu)
			if e != nil {
				return fmt.Errorf("qwen2: layer %d: %w", i, e)
			}
			_ = x.Close()
			x = next
			// Keep only the current residual handle; layer scopes own intermediates.
			s.arrays[len(s.arrays)-1] = x
		}
		h := s.add(mlx.RMSNorm(x, w.Norm, c.RMSNormEps))
		if !all {
			last := s.add(mlx.NewInt32([]int32{int32(len(tokens) - 1)}, []int{1}))
			h = s.add(mlx.TakeAxis(h, last, 1))
		}
		head := s.add(mlx.Transpose(w.Embed))
		out = s.add(mlx.Matmul(h, head))
		var e error
		out, e = s.take(out)
		return e
	})
	if cache != nil {
		if err != nil {
			cache.invalid = true
		} else {
			cache.Offset += len(tokens)
		}
	}
	return out, err
}

func layerForward(x mlx.Array, l Layer, c Config, index, length, offset int, cache *KVCache, hook projectionHook, swiglu *mlx.Closure) (mlx.Array, error) {
	s := &scope{}
	defer s.close()
	h := s.add(mlx.RMSNorm(x, l.InputNorm, c.RMSNormEps))
	project := func(w, b mlx.Array, heads int, kind string) mlx.Array {
		// Fuse the bias before bf16 rounding, as mlx.nn.Linear does.
		a := s.add(mlx.AddMM(b, h, w, 1, 1))
		if hook != nil && kind != "k" && s.err == nil {
			a = s.add(hook(index, kind, h, a))
		}
		a = s.add(mlx.Reshape(a, []int{1, length, heads, c.HeadDim()}))
		return s.add(mlx.TransposeAxes(a, []int{0, 2, 1, 3}))
	}
	q, k, v := project(l.Wq, l.Bq, c.NumHeads, "q"), project(l.Wk, l.Bk, c.NumKVHeads, "k"), project(l.Wv, l.Bv, c.NumKVHeads, "v")
	q = s.add(mlx.RoPE(q, c.HeadDim(), false, c.RopeTheta, 1, offset))
	k = s.add(mlx.RoPE(k, c.HeadDim(), false, c.RopeTheta, 1, offset))
	if s.err != nil {
		return mlx.Array{}, s.err
	}
	if cache != nil {
		// Cache consumes separate handles, while the scope retains its originals.
		kc, e := mlx.Reshape(k, k.Shape())
		if e != nil {
			return mlx.Array{}, e
		}
		vc, e := mlx.Reshape(v, v.Shape())
		if e != nil {
			_ = kc.Close()
			return mlx.Array{}, e
		}
		k, v, e = cache.update(index, kc, vc)
		if e != nil {
			return mlx.Array{}, e
		}
	}
	mask := ""
	if length > 1 {
		mask = "causal"
	}
	a := s.add(mlx.ScaledDotProductAttention(q, k, v, float32(1/math.Sqrt(float64(c.HeadDim()))), mask))
	a = s.add(mlx.TransposeAxes(a, []int{0, 2, 1, 3}))
	a = s.add(mlx.Reshape(a, []int{1, length, c.HiddenSize}))
	a = s.add(mlx.LinearNoBias(a, l.Wo))
	r := s.add(mlx.Add(x, a))
	h = s.add(mlx.RMSNorm(r, l.PostAttnNorm, c.RMSNormEps))
	gate := s.add(mlx.LinearNoBias(h, l.Wgate))
	up := s.add(mlx.LinearNoBias(h, l.Wup))
	if s.err != nil {
		return mlx.Array{}, s.err
	}
	// Match mlx-lm's fused graph: rare sigmoid rounding differences accumulate
	// across layers when these bf16 operations are evaluated separately.
	fused, err := swiglu.Apply(gate, up)
	if err != nil {
		return mlx.Array{}, err
	}
	if len(fused) != 1 {
		_ = mlx.CloseArrays(fused)
		return mlx.Array{}, fmt.Errorf("qwen2: invalid SwiGLU output")
	}
	m := s.add(fused[0], nil)
	m = s.add(mlx.LinearNoBias(m, l.Wdown))
	out := s.add(mlx.Add(r, m))
	return s.take(out)
}
