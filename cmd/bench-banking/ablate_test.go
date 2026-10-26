package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestAblationCompatibility(t *testing.T) {
	free, m, data := sampleReport(t, "lora")
	free.Decoding, free.TrainingSHA, free.TrainingN = "greedy", "old-training", 616
	constrained := free
	constrained.Decoding = "labels"
	dir := t.TempDir()
	paths := []string{filepath.Join(dir, "free.json"), filepath.Join(dir, "labels.json")}
	write := func(path string, r report) {
		t.Helper()
		b, err := json.Marshal(r)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write(paths[0], free)
	write(paths[1], constrained)
	if err := ablate(paths, m, data); err != nil {
		t.Fatal(err)
	}
	if err := ablate([]string{paths[0], paths[0]}, m, data); err == nil {
		t.Fatal("accepted duplicate ablation variants")
	}
	constrained.Identity.Tokenizer = "different"
	write(paths[1], constrained)
	if err := ablate(paths, m, data); err == nil {
		t.Fatal("accepted changed tokenizer")
	}
	constrained = free
	constrained.TrainingSHA = ""
	write(paths[1], constrained)
	if err := ablate(paths, m, data); err == nil {
		t.Fatal("accepted undeclared adapter training data")
	}
}
