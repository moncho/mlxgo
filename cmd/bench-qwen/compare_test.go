package main

import (
	"path/filepath"
	"reflect"
	"testing"

	"github.com/moncho/mlxgo/internal/ticketbench"
)

func TestCompatibleReports(t *testing.T) {
	fresh := func() (report, report) {
		base := report{
			Version: ticketbench.Version, ModelSHA256: "model", TokenizerSHA256: "tokenizer", MaxTokens: 96,
			Manifest:    ticketbench.Manifest{Version: ticketbench.Version},
			Summaries:   map[string]*summary{"test": {Count: 1}},
			Predictions: []prediction{{ID: "one", Split: "test", Expected: "answer"}},
		}
		tuned := base
		tuned.AdaptersSHA256 = "adapters"
		tuned.Summaries = map[string]*summary{"test": {Count: 1}}
		tuned.Predictions = append([]prediction{}, base.Predictions...)
		return base, tuned
	}
	base, tuned := fresh()
	if err := compatible(base, tuned); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*report){
		func(r *report) { r.Version = "other" },
		func(r *report) { r.ModelSHA256 = "other" },
		func(r *report) { r.TokenizerSHA256 = "other" },
		func(r *report) { r.AdaptersSHA256 = "" },
		func(r *report) { r.MaxTokens++ },
		func(r *report) { r.Manifest.Version = "other" },
		func(r *report) { r.Summaries["test"].Count++ },
		func(r *report) { r.Predictions[0].Expected = "other" },
		func(r *report) { r.Predictions[0].ID = "other" },
		func(r *report) { r.Predictions = nil },
	} {
		base, tuned := fresh()
		mutate(&tuned)
		if err := compatible(base, tuned); err == nil {
			t.Fatal("accepted incompatible reports")
		}
	}
}

func TestRecordedReportsMatchScorerAndDataset(t *testing.T) {
	dir := filepath.Join("..", "..", "benchmarks", "tickets")
	manifest, data, err := ticketbench.LoadVerified(dir)
	if err != nil {
		t.Fatal(err)
	}
	references := map[string]ticketbench.Record{}
	for split, records := range data {
		for _, record := range records {
			references[split+"/"+record.ID] = record
		}
	}
	for _, name := range []string{"base", "tuned"} {
		r, err := loadReport(filepath.Join(dir, "results", name+".json"))
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(r.Manifest, manifest) {
			t.Fatal("recorded dataset manifest mismatch")
		}
		seen := map[string]bool{}
		counts := map[string]summary{}
		for _, p := range r.Predictions {
			key := p.Split + "/" + p.ID
			record, ok := references[key]
			if !ok || seen[key] || record.Completion != p.Expected {
				t.Fatalf("invalid or repeated prediction %s", key)
			}
			seen[key] = true
			var score ticketbench.Score
			if p.Split == "retention" {
				score.Exact = ticketbench.ScoreRetention(p.Output, p.Expected)
			} else {
				score, err = ticketbench.ScoreExtraction(p.Output, p.Expected)
				if err != nil {
					t.Fatal(err)
				}
			}
			if score != p.Score || p.TokenLimit != (p.Tokens == r.MaxTokens) || p.Tokens < 0 || p.Tokens > r.MaxTokens {
				t.Fatalf("stale score or token count for %s", key)
			}
			s := counts[p.Split]
			s.Count++
			s.Exact += boolInt(score.Exact)
			s.ValidJSON += boolInt(score.ValidJSON)
			s.Schema += boolInt(score.Schema)
			s.TicketID += boolInt(score.TicketID)
			s.Queue += boolInt(score.Queue)
			s.Priority += boolInt(score.Priority)
			s.TokenLimit += boolInt(p.TokenLimit)
			s.GeneratedTokens += p.Tokens
			s.GenerationSeconds += p.Seconds
			counts[p.Split] = s
		}
		if len(counts) != len(r.Summaries) {
			t.Fatal("split count mismatch")
		}
		for split, got := range counts {
			want := *r.Summaries[split]
			want.PrefillSeconds, want.DecodeSeconds = 0, 0
			if got != want {
				t.Fatalf("%s %s summary mismatch: %+v vs %+v", name, split, got, want)
			}
		}
	}
}
