package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/bpe"
	"github.com/moncho/mlxgo/qwen2"
)

func main() {
	dir := flag.String("model", "models/Qwen2.5-0.5B-Instruct", "local base model directory")
	train := flag.String("train", "qwen2/testdata/train.jsonl", "training JSONL")
	valid := flag.String("valid", "qwen2/testdata/valid.jsonl", "held-out JSONL")
	out := flag.String("out", "checkpoints/qwen-lora.safetensors", "output adapters (overwrites)")
	steps := flag.Int("steps", 30, "optimizer steps")
	batch := flag.Int("batch-size", 2, "examples accumulated per optimizer step")
	rank := flag.Int("rank", 4, "LoRA rank")
	lr := flag.Float64("learning-rate", 0.001, "AdamW learning rate")
	maxGradNorm := flag.Float64("max-grad-norm", 1, "global gradient norm limit (0 disables clipping)")
	maxLength := flag.Int("max-length", 128, "maximum input tokens per example")
	flag.Parse()
	if err := run(*dir, *train, *valid, *out, *steps, *batch, *rank, float32(*lr), float32(*maxGradNorm), *maxLength); err != nil {
		log.Fatal(err)
	}
}

func run(dir, trainPath, validPath, out string, steps, batch, rank int, lr, maxGradNorm float32, maxLength int) error {
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
	train, err := qwen2.ReadExamples(trainPath, tok, maxLength)
	if err != nil {
		return err
	}
	valid, err := qwen2.ReadExamples(validPath, tok, maxLength)
	if err != nil {
		return err
	}
	hash, err := qwen2.CheckpointHash(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return err
	}
	w, err := qwen2.Load(dir, c)
	if err != nil {
		return err
	}
	defer w.Close()
	a, err := qwen2.NewAdapters(c, rank, float32(rank), 42, hash)
	if err != nil {
		return err
	}
	defer a.Close()
	before, err := a.Loss(w, valid)
	if err != nil {
		return err
	}
	fmt.Printf("held-out loss before=%.6f; train=%d validation=%d examples\n", before, len(train), len(valid))
	err = a.Train(w, train, qwen2.TrainOptions{Steps: steps, BatchSize: batch, LearningRate: lr, MaxGradNorm: maxGradNorm, Seed: 42, Report: func(step int, loss float32) {
		if step == 1 || step%5 == 0 || step == steps {
			fmt.Printf("step=%d train_loss=%.6f\n", step, loss)
		}
	}})
	if err != nil {
		return err
	}
	after, err := a.Loss(w, valid)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		return err
	}
	if err := a.Save(out); err != nil {
		return err
	}
	restored, err := qwen2.LoadAdapters(out, c, hash)
	if err != nil {
		return err
	}
	defer restored.Close()
	reloaded, err := restored.Loss(w, valid)
	if err != nil {
		return err
	}
	fmt.Printf("held-out loss: %.6f -> %.6f; reloaded=%.6f\nadapters: %s\n", before, after, reloaded, out)
	if reloaded != after {
		return fmt.Errorf("adapter reload changed validation loss")
	}
	if after >= before {
		return fmt.Errorf("held-out loss did not improve; adapters saved for inspection")
	}
	return nil
}
