package qwen2

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestRejectUnsupportedConfig(t *testing.T) {
	base := map[string]any{"architectures": []string{"Qwen2ForCausalLM"}, "model_type": "qwen2", "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1, "rms_norm_eps": 1e-6, "rope_theta": 1e6, "tie_word_embeddings": true, "vocab_size": 32, "max_position_embeddings": 128, "hidden_act": "silu"}
	path := filepath.Join(t.TempDir(), "config.json")
	write := func(v any) {
		t.Helper()
		data, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write(base)
	if _, err := LoadConfig(path); err != nil {
		t.Fatal(err)
	}
	for key, bad := range map[string]any{"hidden_act": "gelu", "use_sliding_window": true, "rope_scaling": map[string]any{"factor": 2}, "num_attention_heads": 3, "num_key_value_heads": 3, "tie_word_embeddings": false, "max_position_embeddings": 0, "attention_dropout": .1, "architectures": []string{"LlamaForCausalLM"}} {
		t.Run(key, func(t *testing.T) {
			old, ok := base[key]
			base[key] = bad
			write(base)
			if _, err := LoadConfig(path); err == nil {
				t.Fatal("accepted unsupported config")
			}
			if ok {
				base[key] = old
			} else {
				delete(base, key)
			}
		})
	}
}
