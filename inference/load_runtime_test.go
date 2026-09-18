//go:build mlx && mlxruntime

package inference

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/bpe"
	"github.com/moncho/mlxgo/qwen2"
)

func exportFixture(t *testing.T, mutate func(map[string]mlx.Array)) string {
	t.Helper()
	f := readFixture(t)
	dir := t.TempDir()
	writeConfig(t, dir, f.Config)
	params := make(map[string]mlx.Array, len(f.Parameters))
	defer func() {
		for _, a := range params {
			a.Close()
		}
	}()
	for name, p := range f.Parameters {
		a, err := mlx.NewFloat32(p.Data, p.Shape)
		if err != nil {
			t.Fatal(err)
		}
		params[name] = a
	}
	if mutate != nil {
		mutate(params)
	}
	if err := mlx.SaveSafetensors(filepath.Join(dir, "model.safetensors"), params, map[string]string{"synthetic": "true"}); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestBundleGenerationAndLifetime(t *testing.T) {
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			dir := exportFixture(t, nil)
			m, err := Open(dir, Options{})
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			prompt := []int32{1, 5, 2, 9, 4}
			want := []int32{21, 21, 21, 18}
			r, err := m.GenerateTokens(prompt, 4)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(r.Tokens, want) || r.PrefillTokens != 5 || r.PrefillSeconds <= 0 || r.DecodeSeconds <= 0 {
				t.Fatalf("%+v", r)
			}
			r, err = m.GenerateTokens(prompt, 4, 21)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(r.Tokens, []int32{21}) {
				t.Fatalf("stop token: %v", r.Tokens)
			}
			if _, err = m.Generate("hello", 4); !errors.Is(err, ErrTextUnsupported) {
				t.Fatalf("got %v", err)
			}
			for _, tc := range []struct {
				ids  []int32
				n    int
				stop []int32
			}{
				{nil, 1, nil}, {prompt, 0, nil}, {[]int32{-1}, 1, nil}, {[]int32{32}, 1, nil},
				{prompt, 29, nil}, {prompt, 1, []int32{32}},
			} {
				if _, err = m.GenerateTokens(tc.ids, tc.n, tc.stop...); err == nil {
					t.Fatalf("accepted %+v", tc)
				}
			}
			var wg sync.WaitGroup
			for range 8 {
				wg.Go(func() {
					for range 3 {
						r, err := m.GenerateTokens(prompt, 4)
						if err != nil || !slices.Equal(r.Tokens, want) {
							t.Errorf("concurrent generation: %v, %v", r.Tokens, err)
							return
						}
					}
				})
			}
			wg.Wait()
			s, err := m.NewSession()
			if err != nil {
				t.Fatal(err)
			}
			logits, err := s.Step(prompt)
			if err != nil {
				t.Fatal(err)
			}
			defer logits.Close()
			if s.Position() != 5 {
				t.Fatal("wrong position")
			}
			if err = m.Close(); err != nil {
				t.Fatal(err)
			}
			if _, err = s.Step([]int32{1}); !errors.Is(err, ErrClosed) {
				t.Fatalf("closed session: %v", err)
			}
			if _, err = m.NewSession(); !errors.Is(err, ErrClosed) {
				t.Fatalf("closed model: %v", err)
			}
			if _, err = m.GenerateTokens(prompt, 1); !errors.Is(err, ErrClosed) {
				t.Fatalf("closed generation: %v", err)
			}
			if err = s.Close(); err != nil {
				t.Fatal(err)
			}
			if err = m.Close(); err != nil {
				t.Fatal(err)
			}
			if _, err = logits.Float32Data(); err != nil {
				t.Fatalf("model closed caller-owned logits: %v", err)
			}
		})
	}
}

func TestBundleCloseDuringUse(t *testing.T) {
	if err := mlx.SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}
	m, err := Open(exportFixture(t, nil), Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	s, err := m.NewSession()
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	wg.Go(func() {
		a, err := s.Step([]int32{1, 5, 2})
		if err != nil && !errors.Is(err, ErrClosed) {
			t.Error(err)
		}
		a.Close()
	})
	wg.Go(func() {
		_ = s.Position()
		if err := m.Close(); err != nil {
			t.Error(err)
		}
	})
	wg.Wait()
}

func TestBundleRejectsBadWeights(t *testing.T) {
	if err := mlx.SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"missing", "shape", "dtype"} {
		t.Run(kind, func(t *testing.T) {
			dir := exportFixture(t, func(params map[string]mlx.Array) {
				old := params["embed.weight"]
				defer old.Close()
				switch kind {
				case "missing":
					delete(params, "embed.weight")
				case "shape":
					a, err := mlx.Zeros([]int{1}, mlx.Float32)
					if err != nil {
						t.Fatal(err)
					}
					params["embed.weight"] = a
				case "dtype":
					a, err := mlx.AsType(old, mlx.BFloat16)
					if err != nil {
						t.Fatal(err)
					}
					params["embed.weight"] = a
				}
			})
			m, err := Open(dir, Options{})
			if err == nil {
				m.Close()
				t.Fatal("accepted bad weights")
			}
		})
	}
}

// Tiny BF16 weights and the checked-in tokenizer exercise the Qwen loader in CI
// without downloading a checkpoint. Only the opt-in test below asserts language
// output from a pretrained model.
func exportQwenFixture(t *testing.T) (string, qwen2.Config) {
	t.Helper()
	c := qwenConfig()
	c.HiddenSize = 8
	c.IntermediateSize = 16
	c.NumLayers = 1
	c.VocabSize = 151936
	dir := t.TempDir()
	writeConfig(t, dir, c)
	b, err := os.ReadFile("../bpe/testdata/tokenizer.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(dir, "tokenizer.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
	if err = mlx.RandomSeed(42); err != nil {
		t.Fatal(err)
	}
	params := map[string]mlx.Array{}
	defer func() {
		for _, a := range params {
			a.Close()
		}
	}()
	add := func(name string, shape []int, norm bool) {
		var a mlx.Array
		var err error
		if norm {
			a, err = mlx.Ones(shape, mlx.BFloat16)
		} else {
			a, err = mlx.RandomNormal(shape, mlx.BFloat16, 0, .1)
		}
		if err != nil {
			t.Fatal(err)
		}
		params[name] = a
	}
	add("model.embed_tokens.weight", []int{c.VocabSize, c.HiddenSize}, false)
	add("model.norm.weight", []int{c.HiddenSize}, true)
	kv := c.NumKVHeads * c.HeadDim()
	for i := 0; i < c.NumLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d.", i)
		for _, part := range []string{"input_layernorm.weight", "post_attention_layernorm.weight"} {
			add(prefix+part, []int{c.HiddenSize}, true)
		}
		for _, p := range []struct {
			name    string
			out, in int
			bias    bool
		}{
			{"self_attn.q_proj", c.HiddenSize, c.HiddenSize, true},
			{"self_attn.k_proj", kv, c.HiddenSize, true}, {"self_attn.v_proj", kv, c.HiddenSize, true},
			{"self_attn.o_proj", c.HiddenSize, c.HiddenSize, false},
			{"mlp.gate_proj", c.IntermediateSize, c.HiddenSize, false}, {"mlp.up_proj", c.IntermediateSize, c.HiddenSize, false},
			{"mlp.down_proj", c.HiddenSize, c.IntermediateSize, false},
		} {
			add(prefix+p.name+".weight", []int{p.out, p.in}, false)
			if p.bias {
				add(prefix+p.name+".bias", []int{p.out}, false)
			}
		}
	}
	if err = mlx.SaveSafetensors(filepath.Join(dir, "model.safetensors"), params, nil); err != nil {
		t.Fatal(err)
	}
	return dir, c
}

func TestQwenBundleWithAdapters(t *testing.T) {
	if err := mlx.SetDefaultGPU(); err != nil {
		t.Fatal(err)
	}
	dir, c := exportQwenFixture(t)
	w, err := qwen2.Load(dir, c)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	tok, err := bpe.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	hash, err := qwen2.CheckpointHash(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	a, err := qwen2.NewAdapters(c, 2, 2, 42, hash)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	if err = a.Train(w, []qwen2.Example{{Inputs: []int32{1, 2}, Targets: []int32{7, 7}}}, qwen2.TrainOptions{Steps: 2, BatchSize: 1, LearningRate: .02, Seed: 42}); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "adapter.safetensors")
	if err = a.Save(path); err != nil {
		t.Fatal(err)
	}
	for _, adapter := range []string{"", path} {
		m, err := Open(dir, Options{Adapters: adapter})
		if err != nil {
			t.Fatal(err)
		}
		got, err := m.Generate("hello", 3)
		if err != nil {
			m.Close()
			t.Fatal(err)
		}
		var want qwen2.Result
		if adapter == "" {
			want, err = qwen2.Generate(w, c, tok, "hello", 3)
		} else {
			want, err = a.Generate(w, tok, "hello", 3)
		}
		if err != nil {
			m.Close()
			t.Fatal(err)
		}
		if !slices.Equal(got.Tokens, want.Tokens) || got.Text != want.Text {
			t.Fatalf("common=%+v legacy=%+v", got, want)
		}
		var wg sync.WaitGroup
		for range 8 {
			wg.Go(func() {
				got, err := m.Generate("hello", 3)
				if err != nil || !slices.Equal(got.Tokens, want.Tokens) || got.Text != want.Text {
					t.Errorf("concurrent Qwen generation: %+v, %v", got, err)
				}
			})
		}
		wg.Wait()
		m.Close()
	}
	// A valid adapter file for a different base must never be applied silently.
	a.BaseSHA256 = strings.Repeat("0", 64)
	if err = a.Save(path); err != nil {
		t.Fatal(err)
	}
	if m, err := Open(dir, Options{Adapters: path}); err == nil {
		m.Close()
		t.Fatal("accepted incompatible adapters")
	}
}

// This opt-in test compares the common loader with the established Qwen path
// and the pinned reference, then repeats with saved adapters.
func TestRealQwenLoaderParity(t *testing.T) {
	dir := os.Getenv("MLXGO_QWEN2_DIR")
	if dir == "" {
		t.Skip("set MLXGO_QWEN2_DIR to a real local checkpoint")
	}
	if err := mlx.SetDefaultGPU(); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile("../qwen2/testdata/golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var golden struct {
		Prompt string
		Tokens []int32
	}
	if err = json.Unmarshal(b, &golden); err != nil {
		t.Fatal(err)
	}
	c, err := qwen2.LoadConfig(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	tok, err := bpe.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	w, err := qwen2.Load(dir, c)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	paths := []string{""}
	if adapter := os.Getenv("MLXGO_QWEN2_ADAPTERS"); adapter != "" {
		paths = append(paths, adapter)
	}
	for i, path := range paths {
		t.Run(fmt.Sprintf("adapters=%t", i > 0), func(t *testing.T) {
			var want qwen2.Result
			var err error
			if path == "" {
				want, err = qwen2.Generate(w, c, tok, golden.Prompt, 32)
			} else {
				hash, e := qwen2.CheckpointHash(filepath.Join(dir, "model.safetensors"))
				if e != nil {
					t.Fatal(e)
				}
				a, e := qwen2.LoadAdapters(path, c, hash)
				if e != nil {
					t.Fatal(e)
				}
				defer a.Close()
				want, err = a.Generate(w, tok, golden.Prompt, 32)
			}
			if err != nil {
				t.Fatal(err)
			}
			m, err := Open(dir, Options{Adapters: path})
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			got, err := m.Generate(golden.Prompt, 32)
			if err != nil {
				t.Fatal(err)
			}
			if got.Text != want.Text || !slices.Equal(got.Tokens, want.Tokens) || got.PrefillTokens != want.PrefillTokens {
				t.Fatalf("common=%+v\nlegacy=%+v", got, want)
			}
			if path == "" && (len(golden.Tokens) < 32 || !slices.Equal(got.Tokens, golden.Tokens[:32])) {
				t.Fatalf("reference tokens differ: %v", got.Tokens)
			}
			t.Logf("matched %d tokens: %s", len(got.Tokens), got.Text)
		})
	}
}
