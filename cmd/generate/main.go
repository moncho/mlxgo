package main

import (
	"flag"
	"fmt"
	"log"
	"path/filepath"

	"github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/bpe"
	"github.com/moncho/mlxgo/qwen2"
)

func main() {
	dir := flag.String("model", "models/Qwen2.5-0.5B-Instruct", "local Hugging Face model directory")
	prompt := flag.String("prompt", "Explain why the sky is blue in one sentence.", "user prompt")
	maxTokens := flag.Int("max-tokens", 64, "maximum generated tokens")
	adapters := flag.String("adapters", "", "optional mlxgo LoRA safetensors")
	flag.Parse()
	if err := run(*dir, *prompt, *maxTokens, *adapters); err != nil {
		log.Fatal(err)
	}
}

func run(dir, prompt string, maxTokens int, adapterPath string) error {
	if err := mlx.SetDefaultGPU(); err != nil {
		return err
	}
	c, err := qwen2.LoadConfig(filepath.Join(dir, "config.json"))
	if err != nil {
		return err
	}
	tok, err := bpe.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return err
	}
	w, err := qwen2.Load(dir, c)
	if err != nil {
		return err
	}
	defer w.Close()
	var r qwen2.Result
	if adapterPath == "" {
		r, err = qwen2.Generate(w, c, tok, prompt, maxTokens)
	} else {
		hash, e := qwen2.CheckpointHash(filepath.Join(dir, "model.safetensors"))
		if e != nil {
			return e
		}
		a, e := qwen2.LoadAdapters(adapterPath, c, hash)
		if e != nil {
			return e
		}
		defer a.Close()
		r, err = a.Generate(w, tok, prompt, maxTokens)
	}
	if err != nil {
		return err
	}
	fmt.Println(r.Text)
	if r.DecodeSeconds > 0 {
		fmt.Printf("\nGPU: prefill=%d tokens (%.3fs), decode=%.1f tokens/s\n", r.PrefillTokens, r.PrefillSeconds, float64(max(0, len(r.Tokens)-1))/r.DecodeSeconds)
	}
	return nil
}
