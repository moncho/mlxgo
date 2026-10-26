package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"time"

	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/bpe"
	"github.com/moncho/mlxgo/internal/bankingbench"
	"github.com/moncho/mlxgo/qwen2"
)

type options struct {
	logisticC                                             float64
	decoding, trainedOn                                   string
	mode, data, model, system, adapters, split, out, text string
	parent                                                string
	maxTokens, maxLength, train, valid, test, seed        int
}

type identity struct {
	Logistic  *bankingbench.LogisticInfo `json:"logistic,omitempty"`
	Model     string                     `json:"model_sha256,omitempty"`
	Tokenizer string                     `json:"tokenizer_sha256,omitempty"`
	Adapters  string                     `json:"adapters_sha256,omitempty"`
	Config    qwen2.Config               `json:"config"`
}

type report struct {
	Version     string                    `json:"version"`
	CreatedUTC  time.Time                 `json:"created_utc"`
	System      string                    `json:"system"`
	Decoding    string                    `json:"decoding,omitempty"`
	TrainingSHA string                    `json:"training_data_sha256,omitempty"`
	TrainingN   int                       `json:"training_examples,omitempty"`
	Split       string                    `json:"split"`
	Dataset     bankingbench.Manifest     `json:"dataset"`
	Identity    identity                  `json:"identity"`
	MaxTokens   int                       `json:"max_tokens"`
	GoVersion   string                    `json:"go_version"`
	Platform    string                    `json:"platform"`
	Device      string                    `json:"device"`
	PeakRSS     uint64                    `json:"peak_process_rss_bytes"`
	MemoryNote  string                    `json:"memory_note"`
	Summary     bankingbench.Summary      `json:"summary"`
	Predictions []bankingbench.Prediction `json:"predictions"`
}

func main() {
	var o options
	flag.StringVar(&o.mode, "mode", "eval", "prepare, expand, full, search, check, eval, compare, ablate, or classify")
	flag.Float64Var(&o.logisticC, "logistic-c", 4, "inverse L2 regularization for TF-IDF logistic regression")
	flag.StringVar(&o.decoding, "decoding", "greedy", "greedy or labels (finite-label constrained decoding)")
	flag.StringVar(&o.trainedOn, "trained-on", "", "declared adapter training dataset directory; defaults to -data")
	flag.StringVar(&o.parent, "parent", "benchmarks/banking77/data", "original v1 dataset for expand")
	flag.StringVar(&o.data, "data", "benchmarks/banking77/data", "prepared dataset directory")
	flag.StringVar(&o.model, "model", "models/Qwen2.5-0.5B-Instruct", "base model directory")
	flag.StringVar(&o.system, "system", "base", "nb, tfidf (logistic regression), base, or lora")
	flag.StringVar(&o.adapters, "adapters", "", "trained LoRA adapters (required for lora)")
	flag.StringVar(&o.split, "split", "valid", "valid or test; never used for fitting")
	flag.StringVar(&o.out, "out", "", "new report path (will not overwrite)")
	flag.StringVar(&o.text, "text", "", "customer message for classify")
	flag.IntVar(&o.maxTokens, "max-tokens", 24, "generation cap, identical for base and LoRA")
	flag.IntVar(&o.maxLength, "max-length", 1024, "preflight training token limit")
	flag.IntVar(&o.train, "train-per-class", 8, "training rows per intent for prepare")
	flag.IntVar(&o.valid, "valid-per-class", 8, "validation rows per intent for prepare")
	flag.IntVar(&o.test, "test-per-class", 5, "official test rows per intent for prepare (40 for full test)")
	flag.IntVar(&o.seed, "seed", 42, "deterministic split seed for prepare")
	flag.Parse()
	if err := run(o, flag.Args()); err != nil {
		log.Fatal(err)
	}
}

func run(o options, paths []string) error {
	if o.mode == "full" {
		return bankingbench.Full(o.parent, o.data, o.test)
	}
	if o.mode == "expand" {
		return bankingbench.Expand(o.parent, o.data, o.train, o.test)
	}
	if o.mode == "prepare" {
		return bankingbench.Prepare(o.data, bankingbench.Counts{Train: o.train, Valid: o.valid, Test: o.test}, o.seed)
	}
	m, data, err := bankingbench.Load(o.data)
	if err != nil {
		return err
	}
	if o.mode == "search" {
		return search(o, m, data)
	}
	if o.mode == "compare" {
		return compare(paths, m, data)
	}
	if o.mode == "ablate" {
		return ablate(paths, m, data)
	}
	if o.mode == "check" {
		tok, err := bpe.Load(filepath.Join(o.model, "tokenizer.json"))
		if err != nil {
			return err
		}
		for _, split := range []string{"train", "valid", "test"} {
			examples, err := qwen2.ReadExamples(filepath.Join(o.data, split+".jsonl"), tok, o.maxLength)
			if err != nil {
				return err
			}
			longest := 0
			for _, e := range examples {
				longest = max(longest, len(e.Inputs))
			}
			fmt.Printf("%s: %d examples, maximum %d tokens\n", split, len(examples), longest)
		}
		for _, label := range m.Labels {
			if len(tok.Encode(label))+1 > o.maxTokens {
				return fmt.Errorf("generation cap cannot fit label plus EOS: %s", label)
			}
		}
		fmt.Printf("77 intents; excluded %d training duplicates/overlaps; data hashes verified\n", m.ExcludedTrain)
		return nil
	}
	if o.mode != "eval" && o.mode != "classify" || !slices.Contains([]string{"nb", "tfidf", "base", "lora"}, o.system) || o.maxTokens <= 0 {
		return fmt.Errorf("invalid mode, system, or token cap")
	}
	if (o.system == "lora") != (o.adapters != "") {
		return fmt.Errorf("adapters are required for lora and forbidden for other systems")
	}
	if o.mode == "classify" && strings.TrimSpace(o.text) == "" {
		return fmt.Errorf("classify requires -text")
	}
	if o.decoding != "greedy" && o.decoding != "labels" || classical(o.system) && o.decoding != "greedy" {
		return fmt.Errorf("decoding must be greedy/labels; nb has no decoder")
	}
	trainingSHA, trainingN := "", 0
	if classical(o.system) || o.system == "lora" {
		trainingSHA, trainingN = m.Files["train.jsonl"], len(data["train"])
	}
	if o.trainedOn != "" {
		if o.system != "lora" {
			return fmt.Errorf("trained-on applies only to adapters")
		}
		tm, td, err := bankingbench.Load(o.trainedOn)
		if err != nil {
			return err
		}
		if !slices.Equal(tm.Labels, m.Labels) || tm.Files["train.csv"] != m.Files["train.csv"] {
			return fmt.Errorf("adapter training source or labels differ")
		}
		trainingSHA, trainingN = tm.Files["train.jsonl"], len(td["train"])
	}
	var f *os.File
	complete := false
	if o.mode == "eval" {
		if o.out == "" || o.split != "valid" && o.split != "test" {
			return fmt.Errorf("eval requires new -out and valid/test -split")
		}
		if err := os.MkdirAll(filepath.Dir(o.out), 0o755); err != nil {
			return err
		}
		f, err = os.OpenFile(o.out, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
		if err != nil {
			return err
		}
		defer func() {
			f.Close()
			if !complete {
				os.Remove(o.out)
			}
		}()
	}
	predict, cleanup, id, err := predictor(o, m, data["train"])
	if err != nil {
		return err
	}
	defer cleanup()
	if o.mode == "classify" {
		result, err := predict(o.text)
		if err != nil {
			return err
		}
		label := strings.TrimSpace(result.Text)
		if !slices.Contains(m.Labels, label) {
			return fmt.Errorf("model returned an invalid label: %q", result.Text)
		}
		return json.NewEncoder(os.Stdout).Encode(map[string]string{"intent": label})
	}
	// Identical, unscored warm-up; no evaluation example participates in fitting.
	if _, err := predict("I am waiting for my card to arrive."); err != nil {
		return err
	}
	r := report{Version: bankingbench.Version, CreatedUTC: time.Now().UTC(), System: o.system, Split: o.split, Dataset: m, Identity: id,
		Decoding: o.decoding, TrainingSHA: trainingSHA, TrainingN: trainingN,
		MaxTokens: o.maxTokens, GoVersion: runtime.Version(), Platform: runtime.GOOS + "/" + runtime.GOARCH, Device: "gpu",
		MemoryNote: "Process-lifetime peak RSS including model load and warm-up; not total Metal allocation. Zero means unavailable."}
	if classical(o.system) {
		r.Device = "cpu"
		r.MaxTokens = 0
		r.Decoding = "none"
	}
	for i, record := range data[o.split] {
		start := time.Now()
		result, err := predict(record.Text)
		if err != nil {
			return fmt.Errorf("record %s: %w", record.ID, err)
		}
		r.Predictions = append(r.Predictions, bankingbench.Prediction{ID: record.ID, Expected: record.Completion, Output: result.Text,
			Seconds: time.Since(start).Seconds(), Tokens: len(result.Tokens), TokenLimit: len(result.Tokens) == o.maxTokens})
		if (i+1)%77 == 0 || i+1 == len(data[o.split]) {
			fmt.Printf("%s %s: %d/%d\n", o.system, o.split, i+1, len(data[o.split]))
		}
	}
	r.Summary, err = bankingbench.Score(r.Predictions, m.Labels)
	if err != nil {
		return err
	}
	r.PeakRSS = peakRSS()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(r); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	complete = true
	printSummary(r)
	return nil
}

func predictor(o options, m bankingbench.Manifest, train []bankingbench.Record) (func(string) (qwen2.Result, error), func(), identity, error) {
	var id identity
	if o.system == "tfidf" {
		n, err := bankingbench.FitLogistic(train, m.Labels, o.logisticC)
		if err != nil {
			return nil, nil, id, err
		}
		id.Logistic = &n.Info
		fmt.Fprintf(os.Stderr, "TF-IDF fit: %+v\n", n.Info)
		return func(text string) (qwen2.Result, error) { return qwen2.Result{Text: n.Predict(text)}, nil }, func() {}, id, nil
	}
	if o.system == "nb" {
		n, err := bankingbench.Fit(train, m.Labels)
		if err != nil {
			return nil, nil, id, err
		}
		return func(text string) (qwen2.Result, error) { return qwen2.Result{Text: n.Predict(text)}, nil }, func() {}, id, nil
	}
	if err := mlx.SetDefaultGPU(); err != nil {
		return nil, nil, id, err
	}
	c, err := qwen2.LoadConfig(filepath.Join(o.model, "config.json"))
	if err != nil {
		return nil, nil, id, err
	}
	id.Config = c
	tok, err := bpe.Load(filepath.Join(o.model, "tokenizer.json"))
	if err != nil {
		return nil, nil, id, err
	}
	id.Model, err = qwen2.CheckpointHash(o.model)
	if err != nil {
		return nil, nil, id, err
	}
	id.Tokenizer, err = qwen2.CheckpointHash(filepath.Join(o.model, "tokenizer.json"))
	if err != nil {
		return nil, nil, id, err
	}
	w, err := qwen2.Load(o.model, c)
	if err != nil {
		return nil, nil, id, err
	}
	cleanup := func() { w.Close() }
	if o.system == "base" {
		if o.decoding == "labels" {
			predict, err := choicePredictor(w, c, tok, nil, m.Labels, o.maxTokens)
			if err != nil {
				cleanup()
				return nil, nil, id, err
			}
			return predict, cleanup, id, nil
		}
		return func(text string) (qwen2.Result, error) {
			return qwen2.Generate(w, c, tok, bankingbench.Prompt(text, m.Labels), o.maxTokens)
		}, cleanup, id, nil
	}
	a, err := qwen2.LoadAdapters(o.adapters, c, id.Model)
	if err != nil {
		cleanup()
		return nil, nil, id, err
	}
	id.Adapters, err = qwen2.CheckpointHash(o.adapters)
	if err != nil {
		a.Close()
		cleanup()
		return nil, nil, id, err
	}
	if o.decoding == "labels" {
		predict, err := choicePredictor(w, c, tok, a, m.Labels, o.maxTokens)
		if err != nil {
			a.Close()
			cleanup()
			return nil, nil, id, err
		}
		return predict, func() { a.Close(); cleanup() }, id, nil
	}
	return func(text string) (qwen2.Result, error) {
		return a.Generate(w, tok, bankingbench.Prompt(text, m.Labels), o.maxTokens)
	}, func() { a.Close(); cleanup() }, id, nil
}

func printSummary(r report) {
	s := r.Summary
	training := "unspecified"
	if r.TrainingN > 0 {
		training = fmt.Sprint(r.TrainingN)
	} else if r.System == "base" {
		training = "n/a"
	}
	fmt.Printf("%s/%s train=%s %s: accuracy=%d/%d (%.2f%%) macro-F1=%.4f invalid=%d cap_hits=%d mean=%.3fms peak_RSS=%.1fMiB\n",
		r.System, decodingMode(r), training, r.Split, s.Correct, s.Count, 100*s.Accuracy, s.MacroF1, s.Invalid, s.TokenLimit, 1000*s.Seconds/float64(s.Count), float64(r.PeakRSS)/(1<<20))
}

func validateReport(r report, m bankingbench.Manifest, data map[string][]bankingbench.Record) error {
	mode := decodingMode(r)
	if classical(r.System) && mode != "none" || !classical(r.System) && mode != "greedy" && mode != "labels" {
		return fmt.Errorf("invalid report decoding mode")
	}
	if r.Version != bankingbench.Version || !reflect.DeepEqual(r.Dataset, m) || (r.Split != "valid" && r.Split != "test") || !slices.Contains([]string{"nb", "tfidf", "base", "lora"}, r.System) {
		return fmt.Errorf("incompatible report dataset, version, split, or system")
	}
	if (r.System == "lora") != (r.Identity.Adapters != "") || !classical(r.System) && (r.Identity.Model == "" || r.Identity.Tokenizer == "" || r.MaxTokens <= 0) {
		return fmt.Errorf("incomplete model identity")
	}
	if (r.System == "tfidf") != (r.Identity.Logistic != nil) {
		return fmt.Errorf("incorrect logistic identity")
	}
	if info := r.Identity.Logistic; info != nil {
		if info.C <= 0 || math.IsNaN(info.C) || math.IsInf(info.C, 0) || info.Features < 0 || info.Iterations <= 0 || info.Status == "" || math.IsNaN(info.Objective) || math.IsInf(info.Objective, 0) || info.GradientMax < 0 || math.IsNaN(info.GradientMax) || math.IsInf(info.GradientMax, 0) {
			return fmt.Errorf("invalid logistic fitting metadata")
		}
	}
	records := data[r.Split]
	if len(r.Predictions) != len(records) {
		return fmt.Errorf("incorrect prediction count")
	}
	for i, p := range r.Predictions {
		if p.ID != records[i].ID || p.Expected != records[i].Completion {
			return fmt.Errorf("prediction references do not match the prepared dataset")
		}
	}
	s, err := bankingbench.Score(r.Predictions, m.Labels)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(s, r.Summary) {
		return fmt.Errorf("report summary does not match predictions")
	}
	if mode == "labels" && s.Invalid != 0 {
		return fmt.Errorf("constrained report contains invalid labels")
	}
	return nil
}

func compare(paths []string, m bankingbench.Manifest, data map[string][]bankingbench.Record) error {
	if len(paths) != 3 {
		return fmt.Errorf("compare requires three report paths: nb, base, and lora")
	}
	reports := map[string]report{}
	var split string
	for _, path := range paths {
		b, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		var r report
		if err := json.Unmarshal(b, &r); err != nil {
			return err
		}
		if err := validateReport(r, m, data); err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		if !slices.Contains([]string{"nb", "base", "lora"}, r.System) {
			return fmt.Errorf("compare requires nb, base, and lora reports")
		}
		if _, exists := reports[r.System]; exists || split != "" && split != r.Split {
			return fmt.Errorf("duplicate systems or mismatched evaluation splits")
		}
		split = r.Split
		reports[r.System] = r
	}
	b, t := reports["base"], reports["lora"]
	if b.Identity.Model != t.Identity.Model || b.Identity.Tokenizer != t.Identity.Tokenizer || !reflect.DeepEqual(b.Identity.Config, t.Identity.Config) || b.MaxTokens != t.MaxTokens || decodingMode(b) != decodingMode(t) {
		return fmt.Errorf("base and LoRA model or generation settings differ")
	}
	for _, system := range []string{"nb", "base", "lora"} {
		printSummary(reports[system])
	}
	fmt.Println("Single-run process timings, not a throughput benchmark. Few-shot subset, not the full BANKING77 benchmark.")
	return nil
}
