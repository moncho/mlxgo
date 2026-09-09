package main

import (
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"sort"
)

func loadReport(path string) (report, error) {
	var r report
	data, err := os.ReadFile(path)
	if err != nil {
		return r, err
	}
	if err := json.Unmarshal(data, &r); err != nil {
		return r, err
	}
	if r.Version == "" || r.ModelSHA256 == "" || r.TokenizerSHA256 == "" || r.MaxTokens <= 0 || len(r.Summaries) == 0 {
		return r, fmt.Errorf("%s: incomplete report", path)
	}
	for split, s := range r.Summaries {
		if s == nil || s.Count <= 0 || s.Count != r.Manifest.Counts[split] || s.Exact < 0 || s.Exact > s.Count || s.GenerationSeconds <= 0 {
			return r, fmt.Errorf("%s: invalid summary %s", path, split)
		}
	}
	return r, nil
}

func compatible(base, tuned report) error {
	if base.AdaptersSHA256 != "" || tuned.AdaptersSHA256 == "" {
		return fmt.Errorf("expected a base report without adapters and a tuned report with adapters")
	}
	if base.Version != tuned.Version || base.ModelSHA256 != tuned.ModelSHA256 || base.TokenizerSHA256 != tuned.TokenizerSHA256 || base.MaxTokens != tuned.MaxTokens || !reflect.DeepEqual(base.Config, tuned.Config) || !reflect.DeepEqual(base.Manifest, tuned.Manifest) {
		return fmt.Errorf("reports use different datasets, base models, tokenizers, configurations, or generation settings")
	}
	if len(base.Summaries) != len(tuned.Summaries) {
		return fmt.Errorf("reports contain different splits")
	}
	for split, b := range base.Summaries {
		t, ok := tuned.Summaries[split]
		if !ok || b == nil || t == nil || b.Count != t.Count {
			return fmt.Errorf("reports contain different records for split %s", split)
		}
	}
	if len(base.Predictions) != len(tuned.Predictions) {
		return fmt.Errorf("reports contain different prediction counts")
	}
	references := map[string]string{}
	for _, p := range base.Predictions {
		key := p.Split + "/" + p.ID
		if _, exists := references[key]; exists {
			return fmt.Errorf("duplicate prediction %s", key)
		}
		references[key] = p.Expected
	}
	for _, p := range tuned.Predictions {
		key := p.Split + "/" + p.ID
		if expected, ok := references[key]; !ok || expected != p.Expected {
			return fmt.Errorf("reports contain different reference records: %s", key)
		}
		delete(references, key)
	}
	return nil
}

func compare(basePath, tunedPath string) error {
	base, err := loadReport(basePath)
	if err != nil {
		return err
	}
	tuned, err := loadReport(tunedPath)
	if err != nil {
		return err
	}
	if err := compatible(base, tuned); err != nil {
		return err
	}
	fmt.Println("Metric                         Base             Tuned")
	var splits []string
	for split := range base.Summaries {
		splits = append(splits, split)
	}
	sort.Strings(splits)
	for _, split := range splits {
		b, t := base.Summaries[split], tuned.Summaries[split]
		fmt.Printf("%-28s %3d/%-3d (%5.1f%%) %3d/%-3d (%5.1f%%)\n", split+" exact", b.Exact, b.Count, 100*float64(b.Exact)/float64(b.Count), t.Exact, t.Count, 100*float64(t.Exact)/float64(t.Count))
		if split != "retention" {
			for _, metric := range []struct {
				name          string
				before, after int
			}{
				{"valid JSON", b.ValidJSON, t.ValidJSON}, {"schema valid", b.Schema, t.Schema},
				{"ticket ID", b.TicketID, t.TicketID}, {"queue", b.Queue, t.Queue}, {"priority", b.Priority, t.Priority},
				{"token limit hit", b.TokenLimit, t.TokenLimit},
			} {
				fmt.Printf("%-28s %3d/%-3d          %3d/%-3d\n", split+" "+metric.name, metric.before, b.Count, metric.after, t.Count)
			}
		}
		fmt.Printf("%-28s %8.3fs         %8.3fs\n", split+" mean latency", b.GenerationSeconds/float64(b.Count), t.GenerationSeconds/float64(t.Count))
		fmt.Printf("%-28s %8.1f          %8.1f\n", split+" output tokens/sec", float64(b.GeneratedTokens)/b.GenerationSeconds, float64(t.GeneratedTokens)/t.GenerationSeconds)
	}
	fmt.Printf("%-28s %8.1f MiB      %8.1f MiB\n", "Peak process RSS", float64(base.PeakProcessRSS)/(1<<20), float64(tuned.PeakProcessRSS)/(1<<20))
	fmt.Println("Timing includes prefill and termination. RSS includes loading/warm-up, not all GPU allocations.")
	if b, ok := base.Summaries["retention"]; ok && tuned.Summaries["retention"].Exact < b.Exact {
		fmt.Println("WARNING: unrelated-prompt retention regressed. Inspect predictions before using these adapters.")
	}
	return nil
}
