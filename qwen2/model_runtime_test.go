//go:build mlx && mlxruntime

package qwen2

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/bpe"
)

func tinyConfig() Config {
	return Config{Architectures: []string{"Qwen2ForCausalLM"}, ModelType: "qwen2", HiddenSize: 32, IntermediateSize: 64, NumLayers: 2, NumHeads: 2, NumKVHeads: 1, RMSNormEps: 1e-6, RopeTheta: 1000000, TieWordEmbeddings: true, VocabSize: 32, MaxPositions: 512, HiddenAct: "silu"}
}

func tinyWeights(t *testing.T, c Config) *Weights {
	t.Helper()
	if err := mlx.RandomSeed(42); err != nil {
		t.Fatal(err)
	}
	w := &Weights{Layers: make([]Layer, c.NumLayers)}
	t.Cleanup(func() { _ = w.Close() })
	newArray := func(shape []int, ones bool) mlx.Array {
		var a mlx.Array
		var err error
		if ones {
			a, err = mlx.Ones(shape, mlx.Float32)
		} else {
			a, err = mlx.RandomNormal(shape, mlx.Float32, 0, 0.1)
		}
		if err != nil {
			t.Fatal(err)
		}
		return a
	}
	w.Embed = newArray([]int{c.VocabSize, c.HiddenSize}, false)
	w.Norm = newArray([]int{c.HiddenSize}, true)
	for i := range w.Layers {
		l := &w.Layers[i]
		kv := c.NumKVHeads * c.HeadDim()
		l.InputNorm = newArray([]int{c.HiddenSize}, true)
		l.PostAttnNorm = newArray([]int{c.HiddenSize}, true)
		l.Wq = newArray([]int{c.HiddenSize, c.HiddenSize}, false)
		l.Bq = newArray([]int{c.HiddenSize}, false)
		l.Wk = newArray([]int{c.HiddenSize, kv}, false)
		l.Bk = newArray([]int{kv}, false)
		l.Wv = newArray([]int{c.HiddenSize, kv}, false)
		l.Bv = newArray([]int{kv}, false)
		l.Wo = newArray([]int{c.HiddenSize, c.HiddenSize}, false)
		l.Wgate = newArray([]int{c.HiddenSize, c.IntermediateSize}, false)
		l.Wup = newArray([]int{c.HiddenSize, c.IntermediateSize}, false)
		l.Wdown = newArray([]int{c.IntermediateSize, c.HiddenSize}, false)
	}
	if err := mlx.Eval(w.Arrays()...); err != nil {
		t.Fatal(err)
	}
	return w
}

func floatData(t *testing.T, a mlx.Array) []float32 {
	t.Helper()
	f, err := mlx.AsType(a, mlx.Float32)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	d, err := f.Float32Data()
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func maxError(t *testing.T, a, b []float32, limit float64) float64 {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("length %d != %d", len(a), len(b))
	}
	worst := 0.0
	for i := range a {
		diff := math.Abs(float64(a[i] - b[i]))
		if math.IsNaN(diff) || math.IsInf(diff, 0) {
			t.Fatal("nonfinite difference")
		}
		worst = max(worst, diff)
	}
	if worst > limit {
		t.Fatalf("max error %g exceeds %g", worst, limit)
	}
	return worst
}

func TestCachedForwardMatchesFullSequence(t *testing.T) {
	if err := mlx.SetDefaultGPU(); err != nil {
		t.Fatal(err)
	}
	c := tinyConfig()
	w := tinyWeights(t, c)
	cache := NewKVCache(c.NumLayers)
	defer cache.Close()
	ids := []int32{1, 2, 3, 4, 5}
	for i := range ids {
		cached, err := Forward(w, c, ids[i:i+1], cache)
		if err != nil {
			t.Fatal(err)
		}
		full, err := Forward(w, c, ids[:i+1], nil)
		if err != nil {
			t.Fatal(err)
		}
		maxError(t, floatData(t, cached), floatData(t, full), 1e-4)
		_ = mlx.CloseArrays([]mlx.Array{cached, full})
		if cache.Offset != i+1 || !slices.Equal(cache.keys[0].Shape(), []int{1, 1, i + 1, 16}) {
			t.Fatal("incorrect cache growth")
		}
		if err := mlx.Eval(cache.Arrays()...); err != nil {
			t.Fatal(err)
		}
	}
	// A multi-token suffix must also respect the existing cache offset.
	a, err := Forward(w, c, []int32{6, 7}, cache)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	b, err := Forward(w, c, []int32{1, 2, 3, 4, 5, 6, 7}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	maxError(t, floatData(t, a), floatData(t, b), 1e-4)
}

func TestQwenGolden(t *testing.T) {
	dir := os.Getenv("MLXGO_QWEN2_DIR")
	if dir == "" {
		t.Skip("set MLXGO_QWEN2_DIR to run real checkpoint parity")
	}
	if err := mlx.SetDefaultGPU(); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("testdata/golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Prompt, Chat          string
		InputIDs              []int32 `json:"input_ids"`
		Tokens                []int32
		MaxLogitError         float64 `json:"max_logit_error"`
		MinimumMatchingTokens int     `json:"minimum_matching_tokens"`
	}
	if err := json.Unmarshal(data, &g); err != nil {
		t.Fatal(err)
	}
	if g.MaxLogitError <= 0 || g.MaxLogitError > .25 || g.MinimumMatchingTokens < 32 || len(g.Tokens) < 32 {
		t.Fatal("fixture weakens fixed acceptance limits")
	}
	c, err := LoadConfig(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	tok, err := bpe.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	if bpe.ChatTemplate(g.Prompt) != g.Chat || !slices.Equal(tok.Encode(g.Chat), g.InputIDs) {
		t.Fatal("chat/tokenizer mismatch")
	}
	w, err := Load(dir, c)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	cache := NewKVCache(c.NumLayers)
	defer cache.Close()
	logits, err := Forward(w, c, g.InputIDs, cache)
	if err != nil {
		t.Fatal(err)
	}
	defer logits.Close()
	ref, err := mlx.Load("testdata/prefill.npy")
	if err != nil {
		t.Fatal(err)
	}
	defer ref.Close()
	t.Logf("prefill max error: %g", maxError(t, floatData(t, logits), floatData(t, ref), g.MaxLogitError))
	for i, want := range g.Tokens {
		next, err := mlx.ArgmaxAxis(logits, -1, false)
		if err != nil {
			t.Fatal(err)
		}
		if err := mlx.Eval(append(cache.Arrays(), next)...); err != nil {
			t.Fatal(err)
		}
		ids, err := next.UInt32Data()
		if err != nil {
			t.Fatal(err)
		}
		_ = next.Close()
		_ = logits.Close()
		if int32(ids[0]) != want {
			t.Fatalf("token %d: got %d want %d", i, ids[0], want)
		}
		if i+1 < len(g.Tokens) {
			logits, err = Forward(w, c, []int32{want}, cache)
			if err != nil {
				t.Fatal(err)
			}
		}
	}
	t.Logf("matched %d reference tokens", len(g.Tokens))
}
