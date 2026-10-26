package bankingbench

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestFullSplitIsolation(t *testing.T) {
	labels := []string{"a", "b"}
	train, test := rows("train", labels, 12), rows("test", labels, 10)
	pilot, expanded := Counts{2, 2, 2}, Counts{6, 2, 3}
	first, _, _ := Split(train, test, labels, pilot, 42)
	second, _, _ := ExpandedSplit(train, test, labels, pilot, expanded, 42)
	full, _, err := FullSplit(train, test, labels, pilot, expanded, Counts{0, 2, 3}, 42)
	if err != nil {
		t.Fatal(err)
	}
	if len(full["train"]) != 20 || !slices.Equal(full["valid"], first["valid"]) {
		t.Fatal("lost full training or frozen validation")
	}
	for _, r := range second["train"] {
		if !slices.Contains(full["train"], r) {
			t.Fatal("lost existing training row")
		}
	}
	for _, r := range full["test"] {
		for _, old := range append(first["test"], second["test"]...) {
			if r.ID == old.ID || normalized(r.Text) == normalized(old.Text) {
				t.Fatal("reused inspected test row")
			}
		}
	}
	if _, _, err := FullSplit(train, test, labels, pilot, expanded, Counts{0, 2, 6}, 42); err == nil {
		t.Fatal("accepted exhausted test source")
	}
	// A different source ID with inspected text must be excluded too.
	duplicate := first["test"][0]
	duplicate.ID = "test-extra"
	duplicate.Text = "  " + duplicate.Text + "  "
	test = append(test, duplicate)
	full, _, err = FullSplit(train, test, labels, pilot, expanded, Counts{0, 2, 3}, 42)
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range full["test"] {
		if r.ID == duplicate.ID {
			t.Fatal("reused inspected text through another source ID")
		}
	}
}

func TestFullRoundTrip(t *testing.T) {
	dir, _, _ := fixture(t)
	expanded, full := filepath.Join(t.TempDir(), "expanded"), filepath.Join(t.TempDir(), "full")
	if err := Expand(dir, expanded, 2, 1); err != nil {
		t.Fatal(err)
	}
	if err := Full(expanded, full, 1); err != nil {
		t.Fatal(err)
	}
	m, data, err := Load(full)
	if err != nil {
		t.Fatal(err)
	}
	if m.Version != FullVersion || m.PerClass.Train != 0 || len(data["train"]) != 154 {
		t.Fatal("incorrect full manifest")
	}
	if err := Full(expanded, full, 1); err == nil {
		t.Fatal("overwrote full dataset")
	}
	if err := os.WriteFile(filepath.Join(full, "previous-manifest.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Load(full); err == nil {
		t.Fatal("accepted changed predecessor")
	}
}
