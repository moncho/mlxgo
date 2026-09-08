// Package qwen2 provides Qwen2 inference and LoRA training using MLX.
package qwen2

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

type Config struct {
	Architectures     []string        `json:"architectures"`
	ModelType         string          `json:"model_type"`
	HiddenSize        int             `json:"hidden_size"`
	IntermediateSize  int             `json:"intermediate_size"`
	NumLayers         int             `json:"num_hidden_layers"`
	NumHeads          int             `json:"num_attention_heads"`
	NumKVHeads        int             `json:"num_key_value_heads"`
	RMSNormEps        float32         `json:"rms_norm_eps"`
	RopeTheta         float32         `json:"rope_theta"`
	TieWordEmbeddings bool            `json:"tie_word_embeddings"`
	VocabSize         int             `json:"vocab_size"`
	MaxPositions      int             `json:"max_position_embeddings"`
	HiddenAct         string          `json:"hidden_act"`
	RopeScaling       json.RawMessage `json:"rope_scaling"`
	UseSlidingWindow  bool            `json:"use_sliding_window"`
	AttentionDropout  float32         `json:"attention_dropout"`
}

func LoadConfig(path string) (Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Config{}, err
	}
	var c Config
	if err := json.Unmarshal(data, &c); err != nil {
		return c, err
	}
	return c, c.Validate()
}

func (c Config) HeadDim() int {
	if c.NumHeads == 0 {
		return 0
	}
	return c.HiddenSize / c.NumHeads
}

func (c Config) Validate() error {
	if len(c.Architectures) != 1 || c.Architectures[0] != "Qwen2ForCausalLM" || c.ModelType != "qwen2" || !c.TieWordEmbeddings {
		return fmt.Errorf("qwen2: requires Qwen2ForCausalLM with tied embeddings")
	}
	if c.HiddenSize <= 0 || c.IntermediateSize <= 0 || c.NumLayers <= 0 || c.NumHeads <= 0 || c.NumKVHeads <= 0 || c.VocabSize <= 1 || c.MaxPositions <= 0 {
		return fmt.Errorf("qwen2: dimensions must be positive")
	}
	if c.HiddenSize%c.NumHeads != 0 || c.NumHeads%c.NumKVHeads != 0 || c.HeadDim()%2 != 0 {
		return fmt.Errorf("qwen2: incompatible head dimensions")
	}
	if c.HiddenAct != "silu" || len(c.RopeScaling) > 0 && string(c.RopeScaling) != "null" || c.UseSlidingWindow || c.AttentionDropout != 0 {
		return fmt.Errorf("qwen2: only SiLU, unscaled RoPE, full attention and zero dropout are supported")
	}
	for _, f := range []float32{c.RMSNormEps, c.RopeTheta} {
		if f <= 0 || math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
			return fmt.Errorf("qwen2: epsilon and theta must be positive and finite")
		}
	}
	return nil
}
