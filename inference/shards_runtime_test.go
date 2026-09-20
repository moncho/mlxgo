//go:build mlx && mlxruntime

package inference

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/bpe"
	"github.com/moncho/mlxgo/checkpoint"
	"github.com/moncho/mlxgo/qwen2"
)

// Use the native safetensors serializer to make real shards, not mock readers.
// Round-robin assignment exercises frequent shard changes during layer loading.
func shardBundle(t *testing.T, source string) string {
	t.Helper()
	dir := t.TempDir()
	for _, name := range []string{"config.json", "tokenizer.json"} {
		b, err := os.ReadFile(filepath.Join(source, name))
		if name == "tokenizer.json" && os.IsNotExist(err) {
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(filepath.Join(dir, name), b, 0600); err != nil {
			t.Fatal(err)
		}
	}
	m, err := checkpoint.Inspect(source)
	if err != nil {
		t.Fatal(err)
	}
	r, err := checkpoint.Open(source)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	names := m.Names()
	routing := map[string]string{}
	var total int64
	for part := 0; part < 3; part++ {
		file := fmt.Sprintf("model-%05d-of-00003.safetensors", part+1)
		params := map[string]mlx.Array{}
		for i := part; i < len(names); i += 3 {
			a, err := r.Get(names[i])
			if err != nil {
				for _, a := range params {
					a.Close()
				}
				t.Fatal(err)
			}
			params[names[i]] = a
			routing[names[i]] = file
			info, _ := m.Tensor(names[i])
			total += info.Bytes
		}
		err := mlx.SaveSafetensors(filepath.Join(dir, file), params, map[string]string{"format": "pt"})
		for _, a := range params {
			a.Close()
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	b, err := json.Marshal(map[string]any{"metadata": map[string]int64{"total_size": total}, "weight_map": routing})
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestShardedDeepSeekBundles(t *testing.T) {
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			for _, name := range []string{"model.json", "model_engram.json"} {
				t.Run(name, func(t *testing.T) {
					b, err := os.ReadFile(filepath.Join("../deepseek/testdata", name))
					if err != nil {
						t.Fatal(err)
					}
					var f fixture
					if err = json.Unmarshal(b, &f); err != nil {
						t.Fatal(err)
					}
					source := exportFixtureData(t, f, nil)
					dir := shardBundle(t, source)
					var want []int32
					for _, path := range []string{source, dir} {
						m, err := Open(path, Options{})
						if err != nil {
							t.Fatal(err)
						}
						r, err := m.GenerateTokens([]int32{1, 5, 2, 9, 4}, 4)
						m.Close()
						if err != nil {
							t.Fatal(err)
						}
						if want == nil {
							want = r.Tokens
						} else if !slices.Equal(r.Tokens, want) {
							t.Fatalf("sharded=%v single=%v", r.Tokens, want)
						}
					}
				})
			}
		})
	}
}

func TestShardedQwenTrainingAndAdapters(t *testing.T) {
	if err := mlx.SetDefaultGPU(); err != nil {
		t.Fatal(err)
	}
	source, c := exportQwenFixture(t)
	dir := shardBundle(t, source)
	legacy, err := qwen2.CheckpointHash(filepath.Join(source, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	single, err := qwen2.CheckpointHash(source)
	if err != nil || single != legacy {
		t.Fatal("single-file adapter identity changed", err)
	}
	hash, err := qwen2.CheckpointHash(dir)
	if err != nil {
		t.Fatal(err)
	}
	if hash == legacy {
		t.Fatal("sharded identity not distinct")
	}
	base, err := Open(source, Options{})
	if err != nil {
		t.Fatal(err)
	}
	want, err := base.Generate("hello", 3)
	base.Close()
	if err != nil {
		t.Fatal(err)
	}
	m, err := Open(dir, Options{})
	if err != nil {
		t.Fatal(err)
	}
	got, err := m.Generate("hello", 3)
	m.Close()
	if err != nil || !slices.Equal(got.Tokens, want.Tokens) {
		t.Fatal("Qwen sharding changed tokens", got.Tokens, want.Tokens, err)
	}
	w, err := qwen2.Load(dir, c)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
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
	tok, err := bpe.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	direct, err := a.Generate(w, tok, "hello", 3)
	if err != nil {
		t.Fatal(err)
	}
	m, err = Open(dir, Options{Adapters: path})
	if err != nil {
		t.Fatal(err)
	}
	got, err = m.Generate("hello", 3)
	m.Close()
	if err != nil || !slices.Equal(got.Tokens, direct.Tokens) || got.Text != direct.Text {
		t.Fatal("adapter reload mismatch", err)
	}
	a.BaseSHA256 = legacy
	if err = a.Save(path); err != nil {
		t.Fatal(err)
	}
	if m, err = Open(dir, Options{Adapters: path}); err == nil {
		m.Close()
		t.Fatal("accepted adapter for different checkpoint layout")
	}
}

// Opt-in because this writes a temporary copy of all real checkpoint bytes.
func TestRealShardedQwen(t *testing.T) {
	dir := os.Getenv("MLXGO_QWEN2_DIR")
	if dir == "" || os.Getenv("MLXGO_QWEN2_RESHARD") != "1" {
		t.Skip("set MLXGO_QWEN2_DIR and MLXGO_QWEN2_RESHARD=1")
	}
	if err := mlx.SetDefaultGPU(); err != nil {
		t.Fatal(err)
	}
	sharded := shardBundle(t, dir)
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
	m, err := Open(sharded, Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	r, err := m.Generate(golden.Prompt, 32)
	if err != nil {
		t.Fatal(err)
	}
	if len(golden.Tokens) < 32 || !slices.Equal(r.Tokens, golden.Tokens[:32]) {
		t.Fatalf("sharded checkpoint differs from golden: %v", r.Tokens)
	}
	t.Logf("matched 32 reference tokens from three real BF16 shards: %s", r.Text)
}
