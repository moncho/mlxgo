package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/moncho/mlxgo/internal/bankingbench"
)

func sampleReport(t *testing.T, system string) (report, bankingbench.Manifest, map[string][]bankingbench.Record) {
	t.Helper()
	m := bankingbench.Manifest{Version: bankingbench.Version, Labels: []string{"a", "b"}}
	data := map[string][]bankingbench.Record{"test": {{ID: "1", Completion: "a"}, {ID: "2", Completion: "b"}}}
	r := report{Version: bankingbench.Version, System: system, Split: "test", Dataset: m, MaxTokens: 24,
		Identity:    identity{Model: "model", Tokenizer: "tokenizer"},
		Predictions: []bankingbench.Prediction{{ID: "1", Expected: "a", Output: "a"}, {ID: "2", Expected: "b", Output: "a"}}}
	if system == "lora" {
		r.Identity.Adapters = "adapter"
	}
	if system == "nb" {
		r.Identity = identity{}
		r.MaxTokens = 0
	}
	var err error
	r.Summary, err = bankingbench.Score(r.Predictions, m.Labels)
	if err != nil {
		t.Fatal(err)
	}
	return r, m, data
}

func TestReportValidation(t *testing.T) {
	r, m, data := sampleReport(t, "base")
	if err := validateReport(r, m, data); err != nil {
		t.Fatal(err)
	}
	r.Summary.Correct++
	if err := validateReport(r, m, data); err == nil {
		t.Fatal("accepted false summary")
	}
	r, m, data = sampleReport(t, "lora")
	r.Identity.Adapters = ""
	if err := validateReport(r, m, data); err == nil {
		t.Fatal("accepted missing adapter identity")
	}
	r, m, data = sampleReport(t, "base")
	r.Predictions[0].Expected = "b"
	if err := validateReport(r, m, data); err == nil {
		t.Fatal("accepted changed references")
	}
}

func TestCompareCompatibility(t *testing.T) {
	dir := t.TempDir()
	var paths []string
	var m bankingbench.Manifest
	var data map[string][]bankingbench.Record
	for _, system := range []string{"nb", "base", "lora"} {
		r, manifest, records := sampleReport(t, system)
		m, data = manifest, records
		b, _ := json.Marshal(r)
		path := filepath.Join(dir, system+".json")
		if err := os.WriteFile(path, b, 0o644); err != nil {
			t.Fatal(err)
		}
		paths = append(paths, path)
	}
	if err := compare(paths, m, data); err != nil {
		t.Fatal(err)
	}
	if err := compare([]string{paths[0], paths[0], paths[2]}, m, data); err == nil {
		t.Fatal("accepted duplicate systems")
	}
	r, _, _ := sampleReport(t, "lora")
	r.MaxTokens++
	b, _ := json.Marshal(r)
	os.WriteFile(paths[2], b, 0o644)
	if err := compare(paths, m, data); err == nil {
		t.Fatal("accepted different generation settings")
	}
}

func TestLogisticReportValidation(t *testing.T) {
	r, m, data := sampleReport(t, "nb")
	r.System = "tfidf"
	r.Identity.Logistic = &bankingbench.LogisticInfo{C: 4, Features: 12, Iterations: 10, Status: "GradientThreshold", Objective: .2, GradientMax: 1e-7}
	if err := validateReport(r, m, data); err != nil {
		t.Fatal(err)
	}
	r.Identity.Logistic.C = 0
	if err := validateReport(r, m, data); err == nil {
		t.Fatal("accepted invalid regularization")
	}
	r.Identity.Logistic = nil
	if err := validateReport(r, m, data); err == nil {
		t.Fatal("accepted missing fitting identity")
	}
}
