package deepseek

import (
	"fmt"
	"math"
	"slices"
)

// Config describes the experimental float32 text backbone, not the released
// checkpoint format. Engram, vision, DSpark and quantization are not supported.
// Layer numbers are zero-based. A source publishes state to subsequent layers
// until another source replaces it; changing ratio requires a new KV source.
type Config struct {
	Format          string       `json:"format"`
	VocabSize       int          `json:"vocab_size"`
	Dim             int          `json:"dim"`
	InterDim        int          `json:"moe_inter_dim"`
	Layers          int          `json:"n_layers"`
	Heads           int          `json:"n_heads"`
	Experts         int          `json:"n_routed_experts"`
	Router          RouterConfig `json:"router"`
	Limit           float32      `json:"swiglu_limit"`
	QRank           int          `json:"q_lora_rank"`
	HeadDim         int          `json:"head_dim"`
	RopeDim         int          `json:"rope_head_dim"`
	Groups          int          `json:"o_groups"`
	ORank           int          `json:"o_lora_rank"`
	Window          int          `json:"window_size"`
	MaxSeq          int          `json:"max_seq_len"`
	Ratios          []int        `json:"compress_ratios"`
	KVSources       []int        `json:"kv_source_layers"`
	IndexSources    []int        `json:"index_source_layers"`
	RopeTheta       float64      `json:"rope_theta"`
	CompressTheta   float64      `json:"compress_rope_theta"`
	OriginalSeq     int          `json:"original_seq_len"`
	RopeFactor      float64      `json:"rope_factor"`
	BetaFast        float64      `json:"beta_fast"`
	BetaSlow        float64      `json:"beta_slow"`
	IndexHeads      int          `json:"index_n_heads"`
	IndexDim        int          `json:"index_head_dim"`
	IndexTopK       int          `json:"index_topk"`
	CandidateSource int          `json:"candidate_source_layer"`
	CandidateBlocks int          `json:"candidate_topk_blocks"`
	CandidateSize   int          `json:"candidate_block_size"`
	Hyper           HyperConfig  `json:"hyper"`
}

const Float32Format = "mlxgo.deepseek.float32.v1"

func (c Config) Validate() error {
	if c.Format != Float32Format {
		return fmt.Errorf("deepseek: unsupported model format %q (only %s)", c.Format, Float32Format)
	}
	for name, n := range map[string]int{"vocabulary": c.VocabSize, "dimension": c.Dim, "intermediate": c.InterDim, "layers": c.Layers, "heads": c.Heads, "experts": c.Experts, "query rank": c.QRank, "head dimension": c.HeadDim, "rotary dimension": c.RopeDim, "groups": c.Groups, "output rank": c.ORank, "window": c.Window, "context": c.MaxSeq, "index heads": c.IndexHeads, "index dimension": c.IndexDim, "index top-k": c.IndexTopK} {
		if n < 1 || n > 1<<20 {
			return fmt.Errorf("deepseek: invalid %s %d", name, n)
		}
	}
	if c.Layers > 128 || c.MaxSeq > 65536 || c.Window > c.MaxSeq || c.Heads%c.Groups != 0 || c.RopeDim%2 != 0 || c.RopeDim > c.HeadDim || c.RopeDim > c.IndexDim {
		return fmt.Errorf("deepseek: incompatible attention dimensions or context limits")
	}
	if c.Hyper.Streams < 1 || c.Hyper.Streams > 64 || c.Hyper.Iterations < 1 || c.Hyper.Iterations > 1000 || !positive(c.Hyper.NormEpsilon) || !positive(c.Hyper.Epsilon) {
		return fmt.Errorf("deepseek: invalid hyper-connection configuration")
	}
	if err := c.Router.validate(c.Experts); err != nil {
		return err
	}
	if c.Limit < 0 || math.IsNaN(float64(c.Limit)) || math.IsInf(float64(c.Limit), 0) {
		return fmt.Errorf("deepseek: invalid SwiGLU limit")
	}
	for _, x := range []float64{c.RopeTheta, c.CompressTheta, c.RopeFactor, c.BetaFast, c.BetaSlow} {
		if x <= 0 || math.IsInf(x, 0) || math.IsNaN(x) {
			return fmt.Errorf("deepseek: invalid rotary configuration")
		}
	}
	if c.RopeTheta <= 1 || c.CompressTheta <= 1 || c.OriginalSeq < 0 || c.BetaFast < c.BetaSlow {
		return fmt.Errorf("deepseek: invalid rotary extrapolation")
	}
	if len(c.Ratios) != c.Layers {
		return fmt.Errorf("deepseek: one compression ratio is required per layer")
	}
	for _, ids := range [][]int{c.KVSources, c.IndexSources} {
		for i, id := range ids {
			if id < 0 || id >= c.Layers || (i > 0 && ids[i-1] >= id) {
				return fmt.Errorf("deepseek: source layers must be sorted, unique and in range")
			}
		}
	}
	owner, index := -1, -1
	for i, r := range c.Ratios {
		kv, ix := slices.Contains(c.KVSources, i), slices.Contains(c.IndexSources, i)
		if r < 0 || r > c.MaxSeq || (r == 0 && (kv || ix)) || (kv && !ix) {
			return fmt.Errorf("deepseek: invalid compression/source at layer %d", i)
		}
		if kv {
			owner, index = i, -1
		}
		if ix {
			index = i
		}
		if r > 0 && (owner < 0 || c.Ratios[owner] != r || index < owner) {
			return fmt.Errorf("deepseek: layer %d has no compatible KV/index source", i)
		}
	}
	if c.CandidateSource < -1 || c.CandidateSource >= c.Layers {
		return fmt.Errorf("deepseek: invalid candidate source")
	}
	if c.CandidateSource >= 0 {
		if !slices.Contains(c.IndexSources, c.CandidateSource) || c.CandidateBlocks < 1 || c.CandidateSize < 1 || c.CandidateSize > c.MaxSeq {
			return fmt.Errorf("deepseek: invalid candidate selection")
		}
		if c.CandidateBlocks < 1+(c.IndexTopK-1+c.CandidateSize-1)/c.CandidateSize {
			return fmt.Errorf("deepseek: candidate capacity must cover index top-k even when the newest block has only one position")
		}
		for i := c.CandidateSource + 1; i < c.Layers; i++ {
			if c.Ratios[i] > 0 && (c.Ratios[i] != c.Ratios[c.CandidateSource] || slices.Contains(c.KVSources, i)) {
				return fmt.Errorf("deepseek: candidate consumers must share the candidate source's KV positions")
			}
		}
	}
	return nil
}

func (c Config) clone() Config {
	c.Ratios = slices.Clone(c.Ratios)
	c.KVSources = slices.Clone(c.KVSources)
	c.IndexSources = slices.Clone(c.IndexSources)
	return c
}

// ParameterShapes specifies this implementation's float32 checkpoint contract.
// Names follow the pinned reference's state_dict; values use [output,input].
func (c Config) ParameterShapes() (map[string][]int, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	p := map[string][]int{"embed.weight": {c.VocabSize, c.Dim}, "norm.weight": {c.Dim}, "head.weight": {c.VocabSize, c.Dim}}
	hc, mix := c.Hyper.Streams, (2+c.Hyper.Streams)*c.Hyper.Streams
	for i := 0; i < c.Layers; i++ {
		prefix := fmt.Sprintf("layers.%d.", i)
		add := func(name string, shape ...int) { p[prefix+name] = shape }
		for _, part := range []string{"attn", "ffn"} {
			add("hc_"+part+"_fn", mix, hc*c.Dim)
			add("hc_"+part+"_base", mix)
			add("hc_"+part+"_scale", 3)
			add(part+"_norm.weight", c.Dim)
		}
		add("attn.attn_sink", c.Heads)
		add("attn.wq_a.weight", c.QRank, c.Dim)
		add("attn.q_norm.weight", c.QRank)
		add("attn.wq_b.weight", c.Heads*c.HeadDim, c.QRank)
		add("attn.wkv.weight", c.HeadDim, c.Dim)
		add("attn.kv_norm.weight", c.HeadDim)
		add("attn.wo_a.weight", c.Groups*c.ORank, c.Heads*c.HeadDim/c.Groups)
		add("attn.wo_b.weight", c.Dim, c.Groups*c.ORank)
		if slices.Contains(c.KVSources, i) {
			add("attn.compressor.wkv.weight", c.HeadDim, c.Dim)
			add("attn.compressor.norm.weight", c.HeadDim)
			if c.Ratios[i] > 1 {
				add("attn.compressor.wgate.weight", c.HeadDim, c.Dim)
			}
			add("attn.indexer.wk.weight", c.IndexDim, c.HeadDim)
			add("attn.indexer.k_norm.weight", c.IndexDim)
		}
		if slices.Contains(c.IndexSources, i) {
			add("attn.indexer.wq_b.weight", c.IndexHeads*c.IndexDim, c.QRank)
			add("attn.indexer.weights_proj.weight", c.IndexHeads, c.Dim)
		}
		add("ffn.gate.weight", c.Experts, c.Dim)
		add("ffn.gate.bias", c.Experts)
		for e := -1; e < c.Experts; e++ {
			name := "ffn.shared_experts."
			if e >= 0 {
				name = fmt.Sprintf("ffn.experts.%d.", e)
			}
			add(name+"w1.weight", c.InterDim, c.Dim)
			add(name+"w3.weight", c.InterDim, c.Dim)
			add(name+"w2.weight", c.Dim, c.InterDim)
		}
	}
	return p, nil
}
