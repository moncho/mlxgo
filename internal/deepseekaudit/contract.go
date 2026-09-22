package deepseekaudit

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
)

// This is a shape contract, not a runnable Config. It does not invent Engram
// hashes or relax the runtime's context/precision restrictions.
type textConfig struct {
	Vocab        int   `json:"vocab_size"`
	Dim          int   `json:"hidden_size"`
	Inter        int   `json:"moe_intermediate_size"`
	Layers       int   `json:"num_hidden_layers"`
	Heads        int   `json:"num_attention_heads"`
	HeadDim      int   `json:"head_dim"`
	QRank        int   `json:"q_lora_rank"`
	ORank        int   `json:"o_lora_rank"`
	Groups       int   `json:"o_groups"`
	Experts      int   `json:"n_routed_experts"`
	Hyper        int   `json:"hc_mult"`
	Ratios       []int `json:"compress_ratios"`
	KVSources    []int `json:"kv_source_layer_ids"`
	IndexSources []int `json:"index_source_layer_ids"`
	IndexDim     int   `json:"index_head_dim"`
	IndexHeads   int   `json:"index_n_heads"`
	EngramLayers []int `json:"engram_layer_ids"`
	EngramRows   []int `json:"engram_num_embeddings"`
	EngramDim    int   `json:"engram_head_dim"`
	EngramHeads  int   `json:"engram_n_heads"`
	EngramNGram  int   `json:"engram_max_ngram_size"`
}

func readContract(b []byte) (map[string][]int, error) {
	var root struct {
		Text textConfig `json:"text_config"`
	}
	if err := json.Unmarshal(b, &root); err != nil {
		return nil, err
	}
	return shapes(root.Text), nil
}

// The pinned config is checksum-verified before this function is used. A test
// compares this independent schema with Config.ParameterShapes on tiny models.
func shapes(c textConfig) map[string][]int {
	p := map[string][]int{"embed.weight": {c.Vocab, c.Dim}, "norm.weight": {c.Dim}, "head.weight": {c.Vocab, c.Dim}}
	for layer := range c.Layers {
		add := func(name string, dims ...int) { p[fmt.Sprintf("layers.%d.%s", layer, name)] = dims }
		for _, part := range []string{"attn", "ffn"} {
			add("hc_"+part+"_fn", (c.Hyper+2)*c.Hyper, c.Hyper*c.Dim)
			add("hc_"+part+"_base", (c.Hyper+2)*c.Hyper)
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
		if slices.Contains(c.KVSources, layer) {
			add("attn.compressor.wkv.weight", c.HeadDim, c.Dim)
			add("attn.compressor.norm.weight", c.HeadDim)
			if c.Ratios[layer] > 1 {
				add("attn.compressor.wgate.weight", c.HeadDim, c.Dim)
			}
			add("attn.indexer.wk.weight", c.IndexDim, c.HeadDim)
			add("attn.indexer.k_norm.weight", c.IndexDim)
		}
		if slices.Contains(c.IndexSources, layer) {
			add("attn.indexer.wq_b.weight", c.IndexHeads*c.IndexDim, c.QRank)
			add("attn.indexer.weights_proj.weight", c.IndexHeads, c.Dim)
		}
		add("ffn.gate.weight", c.Experts, c.Dim)
		add("ffn.gate.bias", c.Experts)
		for expert := -1; expert < c.Experts; expert++ {
			prefix := "ffn.shared_experts."
			if expert >= 0 {
				prefix = fmt.Sprintf("ffn.experts.%d.", expert)
			}
			add(prefix+"w1.weight", c.Inter, c.Dim)
			add(prefix+"w2.weight", c.Dim, c.Inter)
			add(prefix+"w3.weight", c.Inter, c.Dim)
		}
		if i := slices.Index(c.EngramLayers, layer); i >= 0 {
			add("engram.embed.weight", c.EngramRows[i], c.EngramDim)
			add("engram.wkv.weight", (c.Hyper+1)*c.Dim, (c.EngramNGram-1)*c.EngramHeads*c.EngramDim)
			add("engram.q_weight", c.Hyper, c.Dim)
			add("engram.k_weight", c.Hyper, c.Dim)
		}
	}
	return p
}

func normalize(name string) string {
	name = strings.TrimPrefix(name, "model.")
	p := strings.Split(name, ".")
	for i, part := range p {
		switch part {
		case "self_attn":
			p[i] = "attn"
		case "mlp":
			if p[0] != "vision" {
				p[i] = "ffn"
			}
		case "weight_scale_inv":
			p[i] = "scale"
		case "e_score_correction_bias":
			p[i] = "bias"
		}
	}
	return strings.Join(p, ".")
}
