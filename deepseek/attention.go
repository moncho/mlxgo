package deepseek

import (
	"fmt"
	mlx "github.com/moncho/mlxgo"
	"math"
	"slices"
)

// Shared state is local to one forward, never global across sessions/models.
type attentionState struct {
	owner            *layerCache
	topk, candidates mlx.Array
}

func (session *Session) attention(s *scope, x mlx.Array, layer int, shared *attentionState) mlx.Array {
	if s.err != nil {
		return mlx.Array{}
	}
	c, w := session.model.config, session.model.weights
	cache := &session.layers[layer]
	p := fmt.Sprintf("layers.%d.attn.", layer)
	n := x.Shape()[1]
	start := session.offset
	r := c.Ratios[layer]
	qr := s.add(mlx.RMSNorm(s.linear(x, w[p+"wq_a.weight"]), w[p+"q_norm.weight"], c.Hyper.NormEpsilon))
	q := s.add(mlx.Reshape(s.linear(qr, w[p+"wq_b.weight"]), []int{1, n, c.Heads, c.HeadDim}))
	q = rotary(s, q, c, r, start, 1, false)
	kv := s.add(mlx.RMSNorm(s.linear(x, w[p+"wkv.weight"]), w[p+"kv_norm.weight"], c.Hyper.NormEpsilon))
	kv = rotary(s, kv, c, r, start, 1, false)
	windowLen := n
	if start > 0 {
		kv = s.add(mlx.ConcatenateAxis([]mlx.Array{cache.window, kv}, 1))
		windowLen = cache.windowLen + n
	}
	if start > 0 && windowLen > c.Window {
		kv = s.span(kv, 1, windowLen-c.Window, windowLen)
		windowLen = c.Window
	}
	window := kv
	if windowLen > c.Window {
		window = s.span(kv, 1, windowLen-c.Window, windowLen)
	}
	retain(s, &cache.window, window)
	cache.windowLen = min(windowLen, c.Window)
	winSlots := min(c.Window, windowLen)
	windowIDs := make([]int32, n*winSlots)
	for i := 0; i < n; i++ {
		for j := 0; j < winSlots; j++ {
			id := j
			if start == 0 {
				id = max(0, i-c.Window+1) + j
				if id > i {
					id = -1
				}
			}
			windowIDs[i*winSlots+j] = int32(id)
		}
	}
	indices := s.add(mlx.NewInt32(windowIDs, []int{1, n, winSlots}))
	if r > 0 {
		if source(c, layer) {
			shared.owner = cache
			previous := cache.compressedLen
			latent, complete := session.compress(s, x, layer)
			if complete > 0 && s.err == nil {
				key := s.add(mlx.RMSNorm(s.linear(latent, w[p+"indexer.wk.weight"]), w[p+"indexer.k_norm.weight"], c.Hyper.NormEpsilon))
				key = rotary(s, key, c, r, previous*r, r, false)
				appendCache(s, &cache.keys, key, previous)
				value := rotary(s, latent, c, r, previous*r, r, false)
				appendCache(s, &cache.compressed, value, previous)
				cache.compressedLen += complete
			}
		}
		if s.err != nil {
			return mlx.Array{}
		}
		if shared.owner == nil {
			s.err = fmt.Errorf("deepseek: missing compressed source")
			return mlx.Array{}
		}
		count := shared.owner.compressedLen
		if slices.Contains(c.IndexSources, layer) {
			if count == 0 {
				shared.topk = s.add(mlx.Zeros([]int{1, n, 0}, mlx.Int32))
			} else {
				shared.topk = session.index(s, x, qr, layer, shared)
			}
		}
		if count > 0 {
			kv = s.add(mlx.ConcatenateAxis([]mlx.Array{kv, shared.owner.compressed}, 1))
			valid := s.add(mlx.GreaterEqual(shared.topk, s.add(mlx.NewScalarInt(0))))
			shifted := s.add(mlx.Add(shared.topk, s.add(mlx.NewScalarInt(windowLen))))
			shifted = s.add(mlx.Where(valid, shifted, s.add(mlx.NewScalarInt(-1))))
			indices = s.add(mlx.ConcatenateAxis([]mlx.Array{indices, shifted}, -1))
		}
	}
	o := sparseAttentionTensor(s, q, kv, w[p+"attn_sink"], indices, float32(1/math.Sqrt(float64(c.HeadDim))))
	o = rotary(s, o, c, r, start, 1, true)
	o = s.add(mlx.Reshape(o, []int{1, n, c.Groups, 1, c.Heads * c.HeadDim / c.Groups}))
	wa := s.add(mlx.Reshape(w[p+"wo_a.weight"], []int{c.Groups, c.ORank, c.Heads * c.HeadDim / c.Groups}))
	wa = s.add(mlx.TransposeAxes(wa, []int{0, 2, 1}))
	o = s.add(mlx.Matmul(o, wa))
	o = s.add(mlx.Reshape(o, []int{1, n, c.Groups * c.ORank}))
	return s.linear(o, w[p+"wo_b.weight"])
}

func (session *Session) compress(s *scope, x mlx.Array, layer int) (mlx.Array, int) {
	c, w := session.model.config, session.model.weights
	cache := &session.layers[layer]
	r := c.Ratios[layer]
	p := fmt.Sprintf("layers.%d.attn.compressor.", layer)
	total := cache.pendingLen + x.Shape()[1]
	if cache.pendingLen > 0 {
		x = s.add(mlx.ConcatenateAxis([]mlx.Array{cache.pending, x}, 1))
	}
	cutoff := total - total%r
	var latent mlx.Array
	if cutoff > 0 {
		chunk := x
		if cutoff < total {
			chunk = s.span(x, 1, 0, cutoff)
		}
		latent = s.add(CompressComplete(chunk, w[p+"wkv.weight"], w[p+"wgate.weight"], w[p+"norm.weight"], r, c.Hyper.NormEpsilon))
	}
	if cutoff < total {
		retain(s, &cache.pending, s.span(x, 1, cutoff, total))
	} else {
		cache.pending.Close()
		cache.pending = mlx.Array{}
	}
	cache.pendingLen = total - cutoff
	return latent, cutoff / r
}

func (session *Session) index(s *scope, x, qr mlx.Array, layer int, shared *attentionState) mlx.Array {
	c, w := session.model.config, session.model.weights
	p := fmt.Sprintf("layers.%d.attn.indexer.", layer)
	n, count, r := x.Shape()[1], shared.owner.compressedLen, c.Ratios[layer]
	q := s.add(mlx.Reshape(s.linear(qr, w[p+"wq_b.weight"]), []int{1, n, c.IndexHeads, c.IndexDim}))
	q = rotary(s, q, c, r, session.offset, 1, false)
	keys := s.add(mlx.TransposeAxes(shared.owner.keys, []int{0, 2, 1}))
	keys = s.add(mlx.ExpandDims(keys, 1))
	logits := s.add(mlx.ReLU(s.add(mlx.Matmul(q, keys))))
	weights := s.add(mlx.Multiply(s.linear(x, w[p+"weights_proj.weight"]), s.scalar(float32(1/math.Sqrt(float64(c.IndexDim*c.IndexHeads))))))
	logits = s.add(mlx.SumAxis(s.add(mlx.Multiply(logits, s.add(mlx.ExpandDims(weights, -1)))), 2, false))
	visible := make([]int32, n)
	for i := range visible {
		visible[i] = int32((session.offset + i + 1) / r)
	}
	lens := s.add(mlx.NewInt32(visible, []int{1, n, 1}))
	positions := s.add(mlx.ArangeDType(0, float64(count), 1, mlx.Int32))
	allowed := s.add(mlx.Less(positions, lens))
	logits = s.add(mlx.Where(allowed, logits, s.scalar(float32(math.Inf(-1)))))
	if layer == c.CandidateSource {
		shared.candidates = candidateMask(s, logits, visible, c.CandidateBlocks, c.CandidateSize)
	} else if c.CandidateSource >= 0 && layer > c.CandidateSource {
		logits = s.add(mlx.Where(shared.candidates, logits, s.scalar(float32(math.Inf(-1)))))
	}
	ids := topIndices(s, logits, min(c.IndexTopK, count))
	ids = s.add(mlx.SortAxis(ids, -1))
	ids = s.add(mlx.AsType(ids, mlx.Int32))
	return s.add(mlx.Where(s.add(mlx.Less(ids, lens)), ids, s.add(mlx.NewScalarInt(-1))))
}

func topIndices(s *scope, scores mlx.Array, k int) mlx.Array {
	if s.err != nil {
		return mlx.Array{}
	}
	shape := scores.Shape()
	width := shape[len(shape)-1]
	order := s.add(mlx.ArgSortAxis(scores, -1))
	ids := make([]int32, k)
	for i := range ids {
		ids[i] = int32(width - i - 1)
	}
	return s.add(mlx.TakeAxis(order, s.add(mlx.NewInt32(ids, []int{k})), -1))
}

func candidateMask(s *scope, logits mlx.Array, visible []int32, top, block int) mlx.Array {
	if s.err != nil {
		return mlx.Array{}
	}
	n, width := len(visible), logits.Shape()[2]
	blocks := (width + block - 1) / block
	if pad := blocks*block - width; pad > 0 {
		padding := s.add(mlx.BroadcastTo(s.scalar(float32(math.Inf(-1))), []int{1, n, pad}))
		logits = s.add(mlx.ConcatenateAxis([]mlx.Array{logits, padding}, -1))
	}
	logits = s.add(mlx.Reshape(logits, []int{1, n, blocks, block}))
	maxIDs := s.add(mlx.ArgmaxAxis(logits, -1, true))
	scores := s.add(mlx.TakeAlongAxis(logits, maxIDs, -1))
	scores = s.add(mlx.Reshape(scores, []int{1, n, blocks}))
	last := make([]int32, n)
	for i, v := range visible {
		last[i] = -1
		if v > 0 {
			last[i] = (v - 1) / int32(block)
		}
	}
	bids := s.add(mlx.ArangeDType(0, float64(blocks), 1, mlx.Int32))
	pinned := s.add(mlx.Equal(bids, s.add(mlx.NewInt32(last, []int{1, n, 1}))))
	scores = s.add(mlx.Where(pinned, s.scalar(float32(math.Inf(1))), scores))
	selected := topIndices(s, scores, min(top, blocks))
	values := s.add(mlx.TakeAlongAxis(scores, selected, -1))
	selected = s.add(mlx.AsType(selected, mlx.Int32))
	selected = s.add(mlx.Where(s.add(mlx.Greater(values, s.scalar(float32(math.Inf(-1))))), selected, s.add(mlx.NewScalarInt(-1))))
	mask := s.add(mlx.Equal(s.add(mlx.ExpandDims(bids, -1)), s.add(mlx.ExpandDims(selected, -2))))
	mask = s.add(mlx.Greater(s.add(mlx.SumAxis(mask, -1, false)), s.add(mlx.NewScalarInt(0))))
	expand := make([]int32, width)
	for i := range expand {
		expand[i] = int32(i / block)
	}
	return s.add(mlx.TakeAxis(mask, s.add(mlx.NewInt32(expand, []int{width})), -1))
}

// Adjacent-pair RoPE applies to the tail, with compressed groups positioned at
// their first token. The inverse rotation is required before grouped output.
func rotary(s *scope, x mlx.Array, c Config, ratio, start, stride int, inverse bool) mlx.Array {
	if s.err != nil {
		return mlx.Array{}
	}
	shape := x.Shape()
	n, d := shape[1], shape[len(shape)-1]
	rd := c.RopeDim
	base, original := c.RopeTheta, 0
	if ratio > 0 {
		base, original = c.CompressTheta, c.OriginalSeq
	}
	cosines, sines := make([]float32, n*rd/2), make([]float32, n*rd/2)
	low, high := 0., 0.
	if original > 0 {
		corrected := func(beta float64) float64 {
			return float64(rd) * math.Log(float64(original)/(beta*2*math.Pi)) / (2 * math.Log(base))
		}
		low = math.Max(math.Floor(corrected(c.BetaFast)), 0)
		high = math.Min(math.Ceil(corrected(c.BetaSlow)), float64(rd-1))
	}
	for j := 0; j < rd/2; j++ {
		freq := 1 / math.Pow(base, float64(2*j)/float64(rd))
		if original > 0 {
			ramp := math.Max(0, math.Min(1, (float64(j)-low)/math.Max(high-low, 1e-3)))
			freq = freq/c.RopeFactor*ramp + freq*(1-ramp)
		}
		for i := 0; i < n; i++ {
			angle := float64(start+i*stride) * freq
			if inverse {
				angle = -angle
			}
			cosines[i*rd/2+j] = float32(math.Cos(angle))
			sines[i*rd/2+j] = float32(math.Sin(angle))
		}
	}
	tail := s.span(x, -1, d-rd, d)
	pairs := append(slices.Clone(shape[:len(shape)-1]), rd/2, 2)
	tail = s.add(mlx.Reshape(tail, pairs))
	pairShape := pairs[:len(pairs)-1]
	even := s.add(mlx.Reshape(s.span(tail, -1, 0, 1), pairShape))
	odd := s.add(mlx.Reshape(s.span(tail, -1, 1, 2), pairShape))
	freqShape := []int{1, n, rd / 2}
	if len(shape) == 4 {
		freqShape = []int{1, n, 1, rd / 2}
	}
	co := s.add(mlx.NewFloat32(cosines, freqShape))
	si := s.add(mlx.NewFloat32(sines, freqShape))
	real := s.add(mlx.Subtract(s.add(mlx.Multiply(even, co)), s.add(mlx.Multiply(odd, si))))
	imag := s.add(mlx.Add(s.add(mlx.Multiply(even, si)), s.add(mlx.Multiply(odd, co))))
	rotated := s.add(mlx.StackAxis([]mlx.Array{real, imag}, -1))
	tailShape := slices.Clone(shape)
	tailShape[len(tailShape)-1] = rd
	rotated = s.add(mlx.Reshape(rotated, tailShape))
	if d == rd {
		return rotated
	}
	return s.add(mlx.ConcatenateAxis([]mlx.Array{s.span(x, -1, 0, d-rd), rotated}, -1))
}
