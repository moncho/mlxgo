package bankingbench

import (
	"path/filepath"
	"slices"
	"testing"
)

func TestExpansionFreezesValidationAndExcludesOldTest(t *testing.T) {
	labels := []string{"a", "b"}
	train, test := rows("train", labels, 12), rows("test", labels, 8)
	parent := Counts{2, 2, 2}
	old, _, err := Split(train, test, labels, parent, 42)
	if err != nil {
		t.Fatal(err)
	}
	grown, _, err := ExpandedSplit(train, test, labels, parent, Counts{6, 2, 3}, 42)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(old["valid"], grown["valid"]) || len(grown["train"]) != 12 || len(grown["test"]) != 6 {
		t.Fatal("incorrect expansion or changed validation")
	}
	for _, r := range old["train"] {
		if !slices.Contains(grown["train"], r) {
			t.Fatal("lost original training example")
		}
	}
	for _, r := range grown["train"] {
		if slices.Contains(old["valid"], r) {
			t.Fatal("validation leaked into training")
		}
	}
	for _, r := range grown["test"] {
		if slices.Contains(old["test"], r) {
			t.Fatal("reused inspected test example")
		}
	}
	if _, _, err := ExpandedSplit(train, test, labels, parent, Counts{6, 3, 3}, 42); err == nil {
		t.Fatal("accepted changed validation count")
	}
}

func TestExpandRoundTrip(t *testing.T) {
	dir, m, data := fixture(t)
	out := filepath.Join(t.TempDir(), "expanded")
	if err := Expand(dir, out, 32, 1); err != nil {
		t.Fatal(err)
	}
	expanded, got, err := Load(out)
	if err != nil {
		t.Fatal(err)
	}
	if expanded.Version != ExpandedVersion || expanded.TrainCounts[m.Labels[0]] != 2 || len(got["train"]) != 154 || !slices.Equal(data["valid"], got["valid"]) {
		t.Fatal("expansion failed to cap at available rows or retain validation")
	}
	if err := Expand(dir, out, 32, 1); err == nil {
		t.Fatal("overwrote existing dataset")
	}
}
