package inference

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/moncho/mlxgo/deepseek"
	"github.com/moncho/mlxgo/qwen2"
)

func qwenConfig() qwen2.Config {
	return qwen2.Config{Architectures: []string{"Qwen2ForCausalLM"}, ModelType: "qwen2", HiddenSize: 32, IntermediateSize: 64, NumLayers: 2, NumHeads: 2, NumKVHeads: 1, RMSNormEps: 1e-6, RopeTheta: 1000000, TieWordEmbeddings: true, VocabSize: 32, MaxPositions: 512, HiddenAct: "silu"}
}

type fixture struct {
	Config     deepseek.Config `json:"config"`
	Parameters map[string]struct {
		Shape []int     `json:"shape"`
		Data  []float32 `json:"data"`
	} `json:"parameters"`
}

func readFixture(t *testing.T) fixture {
	t.Helper()
	b, err := os.ReadFile("../deepseek/testdata/model.json")
	if err != nil {
		t.Fatal(err)
	}
	var f fixture
	if err = json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func writeConfig(t *testing.T, dir string, v any) {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(dir, "config.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
}

func TestInspect(t *testing.T) {
	for _, tc := range []struct {
		name   string
		config any
		arch   string
		text   bool
	}{
		{"qwen", qwenConfig(), "qwen2", true},
		{"deepseek", readFixture(t).Config, "deepseek_v41", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			writeConfig(t, dir, tc.config)
			info, err := Inspect(dir)
			if err != nil {
				t.Fatal(err)
			}
			if info.Architecture != tc.arch || info.Text != tc.text || info.Adapters != tc.text || info.VocabSize != 32 {
				t.Fatalf("unexpected info: %+v", info)
			}
		})
	}
}

func TestInspectRejectsUnsupported(t *testing.T) {
	for _, tc := range []struct {
		name  string
		field string
		value any
	}{
		{"architecture", "model_type", "llama"},
		{"released deepseek", "model_type", "deepseek_v41"},
		{"quantization", "quantization", map[string]any{"bits": 4}},
		{"quantization config", "quantization_config", map[string]any{}},
		{"dtype", "dtype", "float32"},
		{"torch dtype", "torch_dtype", "float16"},
		{"mixed formats", "format", deepseek.Float32Format},
		{"sliding window", "use_sliding_window", true},
		{"rope scaling", "rope_scaling", map[string]any{"type": "linear", "factor": 2}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b, _ := json.Marshal(qwenConfig())
			var c map[string]any
			_ = json.Unmarshal(b, &c)
			c[tc.field] = tc.value
			dir := t.TempDir()
			writeConfig(t, dir, c)
			if _, err := Inspect(dir); !errors.Is(err, ErrUnsupported) {
				t.Fatalf("got %v", err)
			}
		})
	}
	t.Run("unknown deepseek field", func(t *testing.T) {
		b, _ := json.Marshal(readFixture(t).Config)
		var c map[string]any
		_ = json.Unmarshal(b, &c)
		c["engram"] = true
		dir := t.TempDir()
		writeConfig(t, dir, c)
		if _, err := Inspect(dir); !errors.Is(err, ErrUnsupported) {
			t.Fatalf("got %v", err)
		}
	})
	t.Run("shards", func(t *testing.T) {
		dir := t.TempDir()
		writeConfig(t, dir, qwenConfig())
		if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), []byte("{}"), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := Inspect(dir); !errors.Is(err, ErrUnsupported) {
			t.Fatalf("got %v", err)
		}
	})
	for _, data := range []string{"{", strings.Repeat(" ", 1<<20+1)} {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := Inspect(dir); err == nil {
			t.Fatal("accepted malformed/oversize config")
		}
	}
}

func TestOpenRejectsBeforeNativeWork(t *testing.T) {
	dir := t.TempDir()
	writeConfig(t, dir, readFixture(t).Config)
	if _, err := Open(dir, Options{Adapters: "adapter.safetensors"}); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("got %v", err)
	}
	writeConfig(t, dir, qwenConfig())
	if _, err := Open(dir, Options{}); err == nil || !strings.Contains(err.Error(), "tokenizer.json") {
		t.Fatalf("got %v", err)
	}
}

func TestZeroModel(t *testing.T) {
	for _, m := range []*Model{nil, {}} {
		if _, err := m.NewSession(); !errors.Is(err, ErrClosed) {
			t.Fatalf("got %v", err)
		}
		if err := m.Close(); err != nil {
			t.Fatal(err)
		}
	}
}
