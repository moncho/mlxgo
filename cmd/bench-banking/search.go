package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"time"

	"github.com/moncho/mlxgo/internal/bankingbench"
	"github.com/moncho/mlxgo/qwen2"
)

type candidate struct {
	Path         string               `json:"validation_report"`
	Adapters     string               `json:"adapters,omitempty"`
	AdapterSHA   string               `json:"adapter_sha256,omitempty"`
	LearningRate float64              `json:"learning_rate,omitempty"`
	Epochs       int                  `json:"epochs,omitempty"`
	C            float64              `json:"logistic_c,omitempty"`
	Summary      bankingbench.Summary `json:"summary"`
}

func better(a, b candidate) bool {
	if a.Summary.MacroF1 != b.Summary.MacroF1 {
		return a.Summary.MacroF1 > b.Summary.MacroF1
	}
	if a.Summary.Correct != b.Summary.Correct {
		return a.Summary.Correct > b.Summary.Correct
	}
	if a.Epochs != b.Epochs {
		return a.Epochs < b.Epochs
	}
	if a.LearningRate != b.LearningRate {
		return a.LearningRate < b.LearningRate
	}
	return a.C < b.C
}

func writeNewJSON(path string, value any) error {
	b, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	_, writeErr := f.Write(append(b, '\n'))
	closeErr := f.Close()
	if writeErr != nil {
		return writeErr
	}
	return closeErr
}

// Search fixes its recipe before fitting and writes the validation selection
// before invoking any test evaluation. A fresh output directory is mandatory.
func search(o options, m bankingbench.Manifest, data map[string][]bankingbench.Record) error {
	if m.Version != bankingbench.FullVersion || o.out == "" {
		return fmt.Errorf("search requires a v3 full dataset and a new output directory")
	}
	if o.adapters == "" || o.trainedOn == "" {
		return fmt.Errorf("search requires previous -adapters and -trained-on for the paired test comparison")
	}
	if _, err := os.Stat(o.adapters); err != nil {
		return err
	}
	if _, _, err := bankingbench.Load(o.trainedOn); err != nil {
		return err
	}
	modelSHA, err := qwen2.CheckpointHash(o.model)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(o.out), 0o755); err != nil {
		return err
	}
	if err := os.Mkdir(o.out, 0o755); err != nil {
		return err
	}
	steps := (len(data["train"]) + 3) / 4
	lrs, cs := []float64{0.0001, 0.0003, 0.001}, []float64{1, 4, 16}
	protocol := map[string]any{"dataset": m, "model_sha256": modelSHA, "learning_rates": lrs, "epochs": []int{1, 2}, "steps_per_epoch": steps, "batch_size": 4, "rank_alpha": 8, "seed": 42, "max_grad_norm": 1, "max_length": 1024, "max_tokens": 24, "decoding": "greedy", "logistic_c": cs, "selection": "validation macro-F1, correct count, fewer epochs, lower learning rate/C", "created_utc": time.Now().UTC()}
	if err := writeNewJSON(filepath.Join(o.out, "protocol.json"), protocol); err != nil {
		return err
	}
	executable, err := os.Executable()
	if err != nil {
		return err
	}
	runChild := func(name, program string, args ...string) error {
		log, err := os.OpenFile(filepath.Join(o.out, name+".log"), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
		if err != nil {
			return err
		}
		defer log.Close()
		start := time.Now()
		fmt.Printf("%s start %s\n", start.UTC().Format(time.RFC3339), name)
		cmd := exec.Command(program, args...)
		cmd.Stdout, cmd.Stderr = log, log
		err = cmd.Run()
		fmt.Printf("%s finish %s elapsed=%s err=%v\n", time.Now().UTC().Format(time.RFC3339), name, time.Since(start), err)
		if err != nil {
			return fmt.Errorf("%s failed (see log): %w", name, err)
		}
		return nil
	}
	evaluate := func(name, system, split string, extra ...string) (report, error) {
		path := filepath.Join(o.out, name+".json")
		args := []string{"-data", o.data, "-model", o.model, "-system", system, "-split", split, "-out", path}
		args = append(args, extra...)
		if err := runChild(name, executable, args...); err != nil {
			return report{}, err
		}
		b, err := os.ReadFile(path)
		if err != nil {
			return report{}, err
		}
		var r report
		if err := json.Unmarshal(b, &r); err != nil {
			return r, err
		}
		if err := validateReport(r, m, data); err != nil {
			return r, err
		}
		printSummary(r)
		return r, nil
	}
	var bestLogistic, bestLoRA candidate
	for i, c := range cs {
		name := fmt.Sprintf("tfidf-c%g-valid", c)
		r, err := evaluate(name, "tfidf", "valid", "-logistic-c", fmt.Sprint(c))
		if err != nil {
			return err
		}
		candidate := candidate{Path: filepath.Join(o.out, name+".json"), C: c, Summary: r.Summary}
		if i == 0 || better(candidate, bestLogistic) {
			bestLogistic = candidate
		}
	}
	for i, lr := range lrs {
		previousCheckpoint := ""
		for epoch := 1; epoch <= 2; epoch++ {
			name := fmt.Sprintf("lora-lr%g-epoch%d", lr, epoch)
			adapter, checkpoint := filepath.Join(o.out, name+".safetensors"), filepath.Join(o.out, name+"-training.safetensors")
			args := []string{"run", "-tags", "mlx", "./cmd/finetune-qwen", "-model", o.model, "-train", filepath.Join(o.data, "train.jsonl"), "-valid", filepath.Join(o.data, "valid.jsonl"), "-steps", strconv.Itoa(steps), "-batch-size", "4", "-rank", "8", "-learning-rate", fmt.Sprint(lr), "-max-grad-norm", "1", "-max-length", "1024", "-checkpoint", checkpoint, "-checkpoint-every", strconv.Itoa(max(1, steps/4)), "-out", adapter}
			if previousCheckpoint != "" {
				args = append(args, "-resume", previousCheckpoint)
			}
			if err := runChild(name+"-train", "go", args...); err != nil {
				return err
			}
			r, err := evaluate(name+"-valid", "lora", "valid", "-adapters", adapter)
			if err != nil {
				return err
			}
			current := candidate{Path: filepath.Join(o.out, name+"-valid.json"), Adapters: adapter, AdapterSHA: r.Identity.Adapters, LearningRate: lr, Epochs: epoch, Summary: r.Summary}
			if i == 0 && epoch == 1 || better(current, bestLoRA) {
				bestLoRA = current
			}
			previousCheckpoint = checkpoint
		}
	}
	selection := map[string]any{"created_utc": time.Now().UTC(), "lora": bestLoRA, "tfidf": bestLogistic}
	if err := writeNewJSON(filepath.Join(o.out, "selection.json"), selection); err != nil {
		return err
	}
	fmt.Printf("Validation selection frozen: LoRA lr=%g epochs=%d; TF-IDF C=%g\n", bestLoRA.LearningRate, bestLoRA.Epochs, bestLogistic.C)
	if _, err := evaluate("nb-test", "nb", "test"); err != nil {
		return err
	}
	if _, err := evaluate("tfidf-test", "tfidf", "test", "-logistic-c", fmt.Sprint(bestLogistic.C)); err != nil {
		return err
	}
	if _, err := evaluate("previous-lora-test", "lora", "test", "-adapters", o.adapters, "-trained-on", o.trainedOn); err != nil {
		return err
	}
	if _, err := evaluate("selected-lora-test", "lora", "test", "-adapters", bestLoRA.Adapters); err != nil {
		return err
	}
	if _, err := evaluate("selected-labels-test", "lora", "test", "-adapters", bestLoRA.Adapters, "-decoding", "labels"); err != nil {
		return err
	}
	_, err = io.WriteString(os.Stdout, "Full-data experiment complete.\n")
	return err
}
