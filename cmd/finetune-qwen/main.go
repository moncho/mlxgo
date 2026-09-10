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
	steps := flag.Int("steps", 30, "additional optimizer steps (also when resuming)")
	batch := flag.Int("batch-size", 2, "examples accumulated per optimizer step")
	rank := flag.Int("rank", 4, "LoRA rank")
	lr := flag.Float64("learning-rate", 0.001, "AdamW learning rate")
	maxGradNorm := flag.Float64("max-grad-norm", 1, "global gradient norm limit (0 disables clipping)")
	maxLength := flag.Int("max-length", 128, "maximum input tokens per example")
	resume := flag.String("resume", "", "full training checkpoint to resume")
	checkpoint := flag.String("checkpoint", "", "full training checkpoint output (atomic replacement)")
	every := flag.Int("checkpoint-every", 0, "save every N completed steps; 0 saves only at the end")
	flag.Parse()
	options := qwen2.TrainOptions{Steps: *steps, BatchSize: *batch, LearningRate: float32(*lr), MaxGradNorm: float32(*maxGradNorm), Seed: 42, ResumeFrom: *resume, CheckpointPath: *checkpoint, CheckpointEvery: *every}
	if err := run(*dir, *train, *valid, *out, *rank, *maxLength, options); err != nil {
		log.Fatal(err)
	}
}

func run(dir, trainPath, validPath, out string, rank, maxLength int, options qwen2.TrainOptions) error {
	if err := validatePaths(dir, trainPath, validPath, out, options); err != nil {
		return err
	}
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
	fmt.Printf("base-model held-out loss=%.6f; train=%d validation=%d examples\n", before, len(train), len(valid))
	localStep := 0
	options.Report = func(step int, loss float32) {
		localStep++
		if localStep == 1 || step%5 == 0 || localStep == options.Steps {
			fmt.Printf("step=%d train_loss=%.6f\n", step, loss)
		}
	}
	err = a.Train(w, train, options)
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

func validatePaths(dir, train, valid, out string, options qwen2.TrainOptions) error {
	inputs := []string{train, valid, filepath.Join(dir, "config.json"), filepath.Join(dir, "tokenizer.json"), filepath.Join(dir, "model.safetensors")}
	for _, output := range []string{out, options.CheckpointPath} {
		if output == "" {
			continue
		}
		for _, input := range inputs {
			if samePath(output, input) {
				return fmt.Errorf("output %s would overwrite input %s", output, input)
			}
		}
	}
	if samePath(out, options.CheckpointPath) || samePath(out, options.ResumeFrom) {
		return fmt.Errorf("adapter output and training checkpoint must use different paths")
	}
	return nil
}

func samePath(a, b string) bool {
	if a == "" || b == "" {
		return false
	}
	left, le := os.Stat(a)
	right, re := os.Stat(b)
	if le == nil && re == nil && os.SameFile(left, right) {
		return true
	}
	canonical := func(path string) string {
		abs, err := filepath.Abs(path)
		if err != nil {
			return path
		}
		if resolved, err := filepath.EvalSymlinks(abs); err == nil {
			return resolved
		}
		if parent, err := filepath.EvalSymlinks(filepath.Dir(abs)); err == nil {
			return filepath.Join(parent, filepath.Base(abs))
		}
		return abs
	}
	return canonical(a) == canonical(b)
}
