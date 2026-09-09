package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/moncho/mlxgo/internal/ticketbench"
)

func TestEvaluateRejectsInvalidOptions(t *testing.T) {
	for _, tc := range []struct {
		split, out string
		tokens     int
	}{
		{"test", "", 96}, {"train", "unused", 96}, {"test", "unused", 0},
	} {
		if err := evaluate("missing", "missing", tc.split, "", tc.out, tc.tokens); err == nil {
			t.Fatal("invalid options accepted")
		}
	}
}

func TestEvaluateDoesNotOverwrite(t *testing.T) {
	dir := t.TempDir()
	if err := ticketbench.Prepare(dir); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "report.json")
	if err := os.WriteFile(path, []byte("keep"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := evaluate("missing", dir, "test", "", path, 96); err == nil {
		t.Fatal("accepted existing report")
	}
	data, err := os.ReadFile(path)
	if err != nil || string(data) != "keep" {
		t.Fatalf("existing report changed: %q %v", data, err)
	}
}
