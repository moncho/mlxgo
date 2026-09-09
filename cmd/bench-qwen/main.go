package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/bpe"
	"github.com/moncho/mlxgo/internal/ticketbench"
	"github.com/moncho/mlxgo/qwen2"
)

type summary struct {
	Count             int     `json:"count"`
	Exact             int     `json:"exact"`
	ValidJSON         int     `json:"valid_json"`
	Schema            int     `json:"schema_valid"`
	TicketID          int     `json:"ticket_id_correct"`
	Queue             int     `json:"queue_correct"`
	Priority          int     `json:"priority_correct"`
	TokenLimit        int     `json:"hit_token_limit"`
	GeneratedTokens   int     `json:"generated_tokens"`
	GenerationSeconds float64 `json:"generation_seconds"`
	PrefillSeconds    float64 `json:"prefill_seconds"`
	DecodeSeconds     float64 `json:"decode_seconds"`
}

type prediction struct {
	ID         string            `json:"id"`
	Split      string            `json:"split"`
	Expected   string            `json:"expected"`
	Output     string            `json:"output"`
	Score      ticketbench.Score `json:"score"`
	TokenLimit bool              `json:"hit_token_limit"`
	Tokens     int               `json:"tokens"`
	Seconds    float64           `json:"seconds"`
}

type report struct {
	Version         string               `json:"version"`
	CreatedUTC      time.Time            `json:"created_utc"`
	ModelSHA256     string               `json:"model_sha256"`
	TokenizerSHA256 string               `json:"tokenizer_sha256"`
	AdaptersSHA256  string               `json:"adapters_sha256,omitempty"`
	Config          qwen2.Config         `json:"model_config"`
	Manifest        ticketbench.Manifest `json:"dataset"`
	MaxTokens       int                  `json:"max_tokens"`
	GoVersion       string               `json:"go_version"`
	Platform        string               `json:"platform"`
	Device          string               `json:"device"`
	PeakProcessRSS  uint64               `json:"peak_process_rss_bytes"`
	MemoryNote      string               `json:"memory_note"`
	Summaries       map[string]*summary  `json:"summaries"`
	Predictions     []prediction         `json:"predictions"`
}

func main() {
	mode := flag.String("mode", "eval", "prepare, eval, or compare")
	baseReport := flag.String("base", "checkpoints/tickets-base.json", "base report for comparison")
	tunedReport := flag.String("tuned", "checkpoints/tickets-tuned.json", "tuned report for comparison")
	dir := flag.String("model", "models/Qwen2.5-0.5B-Instruct", "local base model directory")
	dataDir := flag.String("data", "benchmarks/tickets", "benchmark dataset directory")
	split := flag.String("split", "test", "test or valid (retention is always included)")
	adapters := flag.String("adapters", "", "optional mlxgo LoRA safetensors")
	out := flag.String("out", "", "required evaluation JSON path; must not exist")
	maxTokens := flag.Int("max-tokens", 96, "maximum generated tokens per record")
	flag.Parse()
	var err error
	switch *mode {
	case "prepare":
		err = ticketbench.Prepare(*dataDir)
	case "eval":
		err = evaluate(*dir, *dataDir, *split, *adapters, *out, *maxTokens)
	case "compare":
		err = compare(*baseReport, *tunedReport)
	default:
		err = fmt.Errorf("unknown mode %q", *mode)
	}
	if err != nil {
		log.Fatal(err)
	}
}

func evaluate(dir, dataDir, split, adapterPath, out string, maxTokens int) error {
	if out == "" || maxTokens <= 0 || (split != "test" && split != "valid") {
		return fmt.Errorf("out, positive max-tokens, and split test or valid required")
	}
	manifest, data, err := ticketbench.LoadVerified(dataDir)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(out, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return err
	}
	complete := false
	defer func() {
		_ = f.Close()
		if !complete {
			_ = os.Remove(out)
		}
	}()
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
	modelHash, err := qwen2.CheckpointHash(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return err
	}
	tokHash, err := qwen2.CheckpointHash(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return err
	}
	w, err := qwen2.Load(dir, c)
	if err != nil {
		return err
	}
	defer w.Close()
	generate := func(prompt string) (qwen2.Result, error) { return qwen2.Generate(w, c, tok, prompt, maxTokens) }
	var adapterHash string
	if adapterPath != "" {
		a, err := qwen2.LoadAdapters(adapterPath, c, modelHash)
		if err != nil {
			return err
		}
		defer a.Close()
		generate = func(prompt string) (qwen2.Result, error) { return a.Generate(w, tok, prompt, maxTokens) }
		adapterHash, err = qwen2.CheckpointHash(adapterPath)
		if err != nil {
			return err
		}
	}
	// Use the same unscored warm-up for both model variants, in separate processes.
	if _, err := generate("Reply with only the word ready."); err != nil {
		return err
	}
	r := report{
		Version: ticketbench.Version, CreatedUTC: time.Now().UTC(),
		ModelSHA256: modelHash, TokenizerSHA256: tokHash, AdaptersSHA256: adapterHash,
		Config: c, Manifest: manifest, MaxTokens: maxTokens,
		GoVersion: runtime.Version(), Platform: runtime.GOOS + "/" + runtime.GOARCH,
		Device: "gpu", MemoryNote: "Process-lifetime peak RSS including model loading and warm-up, not total Metal allocation; 0 means unavailable.",
		Summaries: map[string]*summary{},
	}
	for _, name := range []string{split, "retention"} {
		s := &summary{}
		r.Summaries[name] = s
		for _, record := range data[name] {
			start := time.Now()
			result, err := generate(record.Prompt)
			elapsed := time.Since(start).Seconds()
			if err != nil {
				return fmt.Errorf("record %s: %w", record.ID, err)
			}
			var score ticketbench.Score
			if name == "retention" {
				score.Exact = ticketbench.ScoreRetention(result.Text, record.Completion)
			} else {
				score, err = ticketbench.ScoreExtraction(result.Text, record.Completion)
				if err != nil {
					return err
				}
			}
			limit := len(result.Tokens) == maxTokens
			s.Count++
			s.Exact += boolInt(score.Exact)
			s.ValidJSON += boolInt(score.ValidJSON)
			s.Schema += boolInt(score.Schema)
			s.TicketID += boolInt(score.TicketID)
			s.Queue += boolInt(score.Queue)
			s.Priority += boolInt(score.Priority)
			s.TokenLimit += boolInt(limit)
			s.GeneratedTokens += len(result.Tokens)
			s.GenerationSeconds += elapsed
			s.PrefillSeconds += result.PrefillSeconds
			s.DecodeSeconds += result.DecodeSeconds
			r.Predictions = append(r.Predictions, prediction{record.ID, name, record.Completion, result.Text, score, limit, len(result.Tokens), elapsed})
			if s.Count%12 == 0 || s.Count == len(data[name]) {
				fmt.Printf("%s %d/%d exact=%d valid_json=%d\n", name, s.Count, len(data[name]), s.Exact, s.ValidJSON)
			}
		}
	}
	r.PeakProcessRSS = peakRSS()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(r); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	complete = true
	fmt.Printf("report: %s\n", out)
	return nil
}

func boolInt(value bool) int {
	if value {
		return 1
	}
	return 0
}
