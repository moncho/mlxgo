// Package deepseek provides an experimental DeepSeek-V4.1 text backbone with
// float32 activations, optional packed attention weights, and differentiable
// building blocks. It is not a pretrained model loader.
// Inputs are borrowed float32 arrays; returned arrays are owned by the caller.
package deepseek

import (
	"fmt"
	"math"

	mlx "github.com/moncho/mlxgo"
)

// ReferenceRevision pins the official implementation used for numerical tests.
const ReferenceRevision = "df42c109f1defefcbfcedbe7d905718a12266e40"

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
func (s *scope) scalar(x float32) mlx.Array { return s.add(mlx.NewScalarFloat32(x)) }
func (s *scope) linear(x, w mlx.Array) mlx.Array {
	return s.add(mlx.Matmul(x, s.add(mlx.Transpose(w))))
}
func (s *scope) span(x mlx.Array, axis, start, end int) mlx.Array {
	ids := make([]int32, end-start)
	for i := range ids {
		ids[i] = int32(start + i)
	}
	return s.add(mlx.TakeAxis(x, s.add(mlx.NewInt32(ids, []int{len(ids)})), axis))
}
func (s *scope) check(a mlx.Array, name string, shape ...int) {
	if s.err != nil {
		return
	}
	dtype, err := a.DType()
	if err != nil {
		s.err = fmt.Errorf("deepseek: %s: %w", name, err)
		return
	}
	if dtype != mlx.Float32 {
		s.err = fmt.Errorf("deepseek: %s must be float32", name)
		return
	}
	got := a.Shape()
	if len(got) != len(shape) {
		s.err = fmt.Errorf("deepseek: %s: shape %v, want %v", name, got, shape)
		return
	}
	for i, dim := range got {
		if dim <= 0 || (shape[i] != -1 && dim != shape[i]) {
			s.err = fmt.Errorf("deepseek: %s: shape %v, want %v", name, got, shape)
			return
		}
	}
}
func compute(fn func(*scope) []mlx.Array) (out []mlx.Array, err error) {
	err = mlx.Batch(func() error {
		s := &scope{}
		defer func() { _ = mlx.CloseArrays(s.arrays) }()
		results := fn(s)
		if s.err != nil {
			return s.err
		}
		for _, a := range results {
			// Distinct handles transfer ownership without detaching the lazy graph.
			owned, e := mlx.Reshape(a, a.Shape())
			if e != nil {
				_ = mlx.CloseArrays(out)
				out = nil
				return e
			}
			out = append(out, owned)
		}
		return nil
	})
	return out, err
}
func one(fn func(*scope) mlx.Array) (mlx.Array, error) {
	out, err := compute(func(s *scope) []mlx.Array { return []mlx.Array{fn(s)} })
	if err != nil {
		return mlx.Array{}, err
	}
	return out[0], nil
}
func positive(x float32) bool { return x > 0 && !math.IsInf(float64(x), 0) && !math.IsNaN(float64(x)) }

// RouterConfig controls text-token routing. Supported scores match the reference.
type RouterConfig struct {
	TopK               int
	Score              string // "sqrtsoftplus", "sigmoid", or "softmax"
	Temperature, Scale float32
	Normalize          bool
}

func (c RouterConfig) validate(experts int) error {
	if c.TopK < 1 || c.TopK > experts || !positive(c.Temperature) || !positive(c.Scale) {
		return fmt.Errorf("deepseek: invalid router configuration")
	}
	if c.Score != "sqrtsoftplus" && c.Score != "sigmoid" && c.Score != "softmax" {
		return fmt.Errorf("deepseek: unsupported router score %q", c.Score)
	}
	return nil
}

// Route returns [tokens,topK] weights and expert indices. Correction bias affects
// selection only. Equal scores select the lowest expert index first; PyTorch does
// not guarantee the same tie ordering. Selection stays on device, with no host
// data reads, so gradients flow through the selected unbiased scores.
func Route(x, weight, bias mlx.Array, c RouterConfig) (weights, indices mlx.Array, err error) {
	out, err := compute(func(s *scope) []mlx.Array {
		w, i := route(s, x, weight, bias, c)
		return []mlx.Array{w, i}
	})
	if err != nil {
		return weights, indices, err
	}
	return out[0], out[1], nil
}
func route(s *scope, x, weight, bias mlx.Array, c RouterConfig) (mlx.Array, mlx.Array) {
	s.check(x, "router input", -1, -1)
	s.check(weight, "router weight", -1, -1)
	if s.err != nil {
		return mlx.Array{}, mlx.Array{}
	}
	dim, experts := x.Shape()[1], weight.Shape()[0]
	s.check(weight, "router weight", experts, dim)
	s.check(bias, "router bias", experts)
	if s.err == nil {
		s.err = c.validate(experts)
	}
	if s.err != nil {
		return mlx.Array{}, mlx.Array{}
	}
	scores := s.add(mlx.Divide(s.linear(x, weight), s.scalar(c.Temperature)))
	switch c.Score {
	case "softmax":
		scores = s.add(mlx.SoftmaxAxis(scores, -1, true))
	case "sigmoid":
		scores = s.add(mlx.Sigmoid(scores))
	default:
		// Log1p preserves tiny positive scores for strongly negative logits.
		positivePart := s.add(mlx.Maximum(scores, s.scalar(0)))
		tail := s.add(mlx.Exp(s.add(mlx.Negative(s.add(mlx.Abs(scores))))))
		scores = s.add(mlx.Sqrt(s.add(mlx.Add(positivePart, s.add(mlx.Log1p(tail))))))
	}
	selection := s.add(mlx.StopGradient(s.add(mlx.Add(scores, bias))))
	axis := s.add(mlx.ArangeDType(0, float64(experts), 1, mlx.Int32))
	chosen := make([]mlx.Array, c.TopK)
	for k := range chosen {
		chosen[k] = s.add(mlx.ArgmaxAxis(selection, -1, true))
		mask := s.add(mlx.Equal(axis, chosen[k]))
		selection = s.add(mlx.Where(mask, s.scalar(float32(math.Inf(-1))), selection))
	}
	indices := s.add(mlx.ConcatenateAxis(chosen, -1))
	weights := s.add(mlx.TakeAlongAxis(scores, indices, -1))
	if c.Normalize && c.TopK > 1 {
		denom := s.add(mlx.Add(s.add(mlx.SumAxis(weights, -1, true)), s.scalar(1e-20)))
		weights = s.add(mlx.Divide(weights, denom))
	}
	return s.add(mlx.Multiply(weights, s.scalar(c.Scale))), indices
}

// ExpertWeights uses the reference checkpoint orientation [output,input].
// The caller owns the arrays and must keep them open during use.
type ExpertWeights struct{ Gate, Up, Down mlx.Array }

// Expert computes one float32 SwiGLU expert for x [tokens,dim]. routing must
// be [tokens,1]; use ones for an unweighted expert. A positive limit clips the
// up projection on both sides and the gate only from above, matching training.
// Routing weights are applied before the down projection. All inputs are
// borrowed; the returned array is owned. This does not quantize activations.
func Expert(x mlx.Array, w ExpertWeights, routing mlx.Array, limit float32) (mlx.Array, error) {
	return one(func(s *scope) mlx.Array {
		if limit < 0 || math.IsNaN(float64(limit)) || math.IsInf(float64(limit), 0) {
			s.err = fmt.Errorf("deepseek: invalid SwiGLU limit")
			return mlx.Array{}
		}
		s.check(x, "expert input", -1, -1)
		if s.err != nil {
			return mlx.Array{}
		}
		s.check(routing, "expert routing", x.Shape()[0], 1)
		if s.err != nil {
			return mlx.Array{}
		}
		return expert(s, x, w, routing, limit)
	})
}

func expert(s *scope, x mlx.Array, w ExpertWeights, routing mlx.Array, limit float32) mlx.Array {
	dim := x.Shape()[1]
	s.check(w.Gate, "expert gate", -1, dim)
	if s.err != nil {
		return mlx.Array{}
	}
	inter := w.Gate.Shape()[0]
	s.check(w.Up, "expert up", inter, dim)
	s.check(w.Down, "expert down", dim, inter)
	if s.err != nil {
		return mlx.Array{}
	}
	gate, up := s.linear(x, w.Gate), s.linear(x, w.Up)
	if limit > 0 {
		gate = s.add(mlx.Minimum(gate, s.scalar(limit)))
		up = s.add(mlx.Clip(up, s.scalar(-limit), s.scalar(limit)))
	}
	h := s.add(mlx.Multiply(s.add(mlx.SiLU(gate)), up))
	h = s.add(mlx.Multiply(h, routing))
	return s.linear(h, w.Down)
}

// MoE computes routed plus shared experts for x [tokens,dim]. It deliberately
// evaluates every expert and masks contributions: suitable for small numerical
// tests and autograd, NOT efficient large-model inference. It supports text
// routing only, not the separate vision-token correction bias.
func MoE(x, gateWeight, gateBias mlx.Array, experts []ExpertWeights, shared ExpertWeights, c RouterConfig, limit float32) (mlx.Array, error) {
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
		out := expert(s, x, shared, s.scalar(1), limit)
		for i, e := range experts {
			if s.err != nil {
				return mlx.Array{}
			}
			mask := s.add(mlx.Equal(indices, s.add(mlx.NewScalarInt(i))))
			selected := s.add(mlx.Where(mask, weights, s.scalar(0)))
			amplitude := s.add(mlx.SumAxis(selected, -1, true))
			out = s.add(mlx.Add(out, expert(s, x, e, amplitude, limit)))
		}
		return out
	})
}

// HyperConfig controls mHC mixing. Neither epsilon is inferred from the other.
type HyperConfig struct {
	Streams, Iterations  int
	NormEpsilon, Epsilon float32
}

// HyperMix returns pre/post [batch,seq,streams] and combination
// [batch,seq,streams,streams] coefficients for x [batch,seq,streams,dim].
// Projection is normalized over the entire flattened residual stream.
func HyperMix(x, projection, scale, base mlx.Array, c HyperConfig) (pre, post, combination mlx.Array, err error) {
	out, err := compute(func(s *scope) []mlx.Array {
		if c.Streams < 1 || c.Streams > 64 || c.Iterations < 1 || c.Iterations > 1000 || !positive(c.NormEpsilon) || !positive(c.Epsilon) {
			s.err = fmt.Errorf("deepseek: invalid hyper-connection configuration")
			return nil
		}
		s.check(x, "hyper input", -1, -1, c.Streams, -1)
		if s.err != nil {
			return nil
		}
		shape, hc := x.Shape(), c.Streams
		mix := (2 + hc) * hc
		s.check(projection, "hyper projection", mix, hc*shape[3])
		s.check(scale, "hyper scale", 3)
		s.check(base, "hyper base", mix)
		if s.err != nil {
			return nil
		}
		flat := s.add(mlx.Reshape(x, []int{shape[0], shape[1], hc * shape[3]}))
		rms := s.add(mlx.Sqrt(s.add(mlx.Add(s.add(mlx.MeanAxis(s.add(mlx.Square(flat)), -1, true)), s.scalar(c.NormEpsilon)))))
		mixes := s.add(mlx.Divide(s.linear(flat, projection), rms))
		affine := func(start, end, k int) mlx.Array {
			return s.add(mlx.Add(s.add(mlx.Multiply(s.span(mixes, -1, start, end), s.span(scale, 0, k, k+1))), s.span(base, 0, start, end)))
		}
		pre := s.add(mlx.Add(s.add(mlx.Sigmoid(affine(0, hc, 0))), s.scalar(c.Epsilon)))
		post := s.add(mlx.Multiply(s.add(mlx.Sigmoid(affine(hc, 2*hc, 1))), s.scalar(2)))
		comb := s.add(mlx.Reshape(affine(2*hc, mix, 2), []int{shape[0], shape[1], hc, hc}))
		comb = s.add(mlx.Add(s.add(mlx.SoftmaxAxis(comb, -1, true)), s.scalar(c.Epsilon)))
		normalize := func(axis int) {
			comb = s.add(mlx.Divide(comb, s.add(mlx.Add(s.add(mlx.SumAxis(comb, axis, true)), s.scalar(c.Epsilon)))))
		}
		normalize(-2)
		for i := 1; i < c.Iterations; i++ {
			normalize(-1)
			normalize(-2)
		}
		return []mlx.Array{pre, post, comb}
	})
	if err != nil {
		return pre, post, combination, err
	}
	return out[0], out[1], out[2], nil
}

// HyperPre collapses x [batch,seq,streams,dim] using the coefficients produced
// by the PREVIOUS sublayer, not the current sublayer's HyperMix call.
func HyperPre(x, pre mlx.Array) (mlx.Array, error) {
	return one(func(s *scope) mlx.Array {
		s.check(x, "hyper input", -1, -1, -1, -1)
		if s.err != nil {
			return mlx.Array{}
		}
		shape := x.Shape()
		s.check(pre, "pre mix", shape[:3]...)
		if s.err != nil {
			return mlx.Array{}
		}
		return s.add(mlx.SumAxis(s.add(mlx.Multiply(x, s.add(mlx.ExpandDims(pre, -1)))), 2, false))
	})
}

// HyperPost expands a sublayer output and mixes residual input streams. The
// combination's first stream axis is INPUT, its second is OUTPUT.
func HyperPost(x, residual, post, combination mlx.Array) (mlx.Array, error) {
	return one(func(s *scope) mlx.Array {
		s.check(residual, "residual", -1, -1, -1, -1)
		if s.err != nil {
			return mlx.Array{}
		}
		v := residual.Shape()
		s.check(x, "sublayer output", v[0], v[1], v[3])
		s.check(post, "post mix", v[:3]...)
		s.check(combination, "combination", v[0], v[1], v[2], v[2])
		if s.err != nil {
			return mlx.Array{}
		}
		expanded := s.add(mlx.Multiply(s.add(mlx.ExpandDims(post, -1)), s.add(mlx.ExpandDims(x, -2))))
		weighted := s.add(mlx.Multiply(s.add(mlx.ExpandDims(combination, -1)), s.add(mlx.ExpandDims(residual, -2))))
		return s.add(mlx.Add(expanded, s.add(mlx.SumAxis(weighted, 2, false))))
	})
}

// SparseAttention uses shared keys/values kv [batch,positions,dim] and q
// [batch,seq,heads,dim]. Indices are [batch,seq,slots] in row-major order;
// -1 means an unused slot, including entirely empty rows. The sink [heads]
// contributes to the denominator only. This is a float32 mathematical
// reference, not the official BF16 sparse kernel or an optimized CSA2 engine.
func SparseAttention(q, kv, sink mlx.Array, indices []int32, slots int, scale float32) (mlx.Array, error) {
	return one(func(s *scope) mlx.Array {
		s.check(q, "queries", -1, -1, -1, -1)
		if s.err != nil {
			return mlx.Array{}
		}
		v := q.Shape()
		s.check(kv, "key/value", v[0], -1, v[3])
		s.check(sink, "sink", v[2])
		if s.err != nil {
			return mlx.Array{}
		}
		positions := kv.Shape()[1]
		if positions > math.MaxInt32/v[0] {
			s.err = fmt.Errorf("deepseek: flattened key/value positions exceed int32 indexing")
			return mlx.Array{}
		}
		if slots <= 0 || len(indices)/v[0]/v[1] != slots || len(indices)%(v[0]*v[1]) != 0 || !positive(scale) {
			s.err = fmt.Errorf("deepseek: invalid sparse indices shape or scale")
			return mlx.Array{}
		}
		for _, id := range indices {
			if id < -1 || int(id) >= positions {
				s.err = fmt.Errorf("deepseek: sparse index %d out of bounds", id)
				return mlx.Array{}
			}
		}
		idx := s.add(mlx.NewInt32(indices, []int{v[0], v[1], slots}))
		return sparseAttentionTensor(s, q, kv, sink, idx, scale)
	})
}

// Internal callers construct valid indices on-device, avoiding host reads in
// the indexer. The public slice API validates caller-supplied indices above.
func sparseAttentionTensor(s *scope, q, kv, sink, indices mlx.Array, scale float32) mlx.Array {
	if s.err != nil {
		return mlx.Array{}
	}
	v, positions, slots := q.Shape(), kv.Shape()[1], indices.Shape()[2]
	valid := s.add(mlx.GreaterEqual(indices, s.add(mlx.NewScalarInt(0))))
	idx := s.add(mlx.Maximum(indices, s.add(mlx.NewScalarInt(0))))
	offsets := s.add(mlx.ArangeDType(0, float64(v[0]), 1, mlx.Int32))
	offsets = s.add(mlx.Reshape(offsets, []int{v[0], 1, 1}))
	offsets = s.add(mlx.Multiply(offsets, s.add(mlx.NewScalarInt(positions))))
	idx = s.add(mlx.Add(idx, offsets))
	mask := s.add(mlx.ExpandDims(valid, 2))
	flat := s.add(mlx.Reshape(kv, []int{v[0] * positions, v[3]}))
	gathered := s.add(mlx.TakeAxis(flat, idx, 0))
	keys := s.add(mlx.TransposeAxes(gathered, []int{0, 1, 3, 2}))
	logits := s.add(mlx.Multiply(s.add(mlx.Matmul(q, keys)), s.scalar(scale)))
	logits = s.add(mlx.Where(mask, logits, s.scalar(float32(math.Inf(-1)))))
	sinks := s.add(mlx.Reshape(sink, []int{1, 1, v[2], 1}))
	sinks = s.add(mlx.BroadcastTo(sinks, []int{v[0], v[1], v[2], 1}))
	withSink := s.add(mlx.ConcatenateAxis([]mlx.Array{logits, sinks}, -1))
	prob := s.add(mlx.SoftmaxAxis(withSink, -1, true))
	return s.add(mlx.Matmul(s.span(prob, -1, 0, slots), gathered))
}

// CompressComplete pools complete groups of consecutive tokens and applies RMS
// normalization, before RoPE. It rejects partial groups: Session manages their
// persistent decode state, while this function remains stateless.
// Ratio 1 ignores gateWeight, as in the reference.
func CompressComplete(x, kvWeight, gateWeight, norm mlx.Array, ratio int, epsilon float32) (mlx.Array, error) {
	return one(func(s *scope) mlx.Array {
		s.check(x, "compressor input", -1, -1, -1)
		s.check(kvWeight, "compressor projection", -1, -1)
		if s.err != nil {
			return mlx.Array{}
		}
		v, dim := x.Shape(), kvWeight.Shape()[0]
		if ratio < 1 || v[1]%ratio != 0 || !positive(epsilon) {
			s.err = fmt.Errorf("deepseek: compression requires complete groups and positive epsilon")
			return mlx.Array{}
		}
		s.check(kvWeight, "compressor projection", dim, v[2])
		s.check(norm, "compressor norm", dim)
		if ratio > 1 {
			s.check(gateWeight, "compressor gate", dim, v[2])
		}
		if s.err != nil {
			return mlx.Array{}
		}
		kv := s.linear(x, kvWeight)
		if ratio > 1 {
			shape := []int{v[0], v[1] / ratio, ratio, dim}
			kv = s.add(mlx.Reshape(kv, shape))
			score := s.add(mlx.Reshape(s.linear(x, gateWeight), shape))
			kv = s.add(mlx.SumAxis(s.add(mlx.Multiply(kv, s.add(mlx.SoftmaxAxis(score, 2, true)))), 2, false))
		}
		return s.add(mlx.RMSNorm(kv, norm, epsilon))
	})
}
