package main

import (
	"path/filepath"
	"testing"

	"github.com/moncho/mlxgo/internal/bankingbench"
)

func TestValidationSelection(t *testing.T) {
	base := candidate{Epochs: 2, LearningRate: .0003, Summary: bankingbench.Summary{MacroF1: .8, Correct: 480}}
	a := base
	a.Summary.MacroF1 = .81
	a.Summary.Correct = 479
	if !better(a, base) {
		t.Fatal("macro F1 must take priority")
	}
	a = base
	a.Epochs = 1
	if !better(a, base) {
		t.Fatal("prefer fewer updates on ties")
	}
	a = base
	a.LearningRate = .0001
	if !better(a, base) || better(base, base) {
		t.Fatal("unstable tie selection")
	}
}

func TestSearchArtifactsNeverOverwrite(t *testing.T) {
	path := filepath.Join(t.TempDir(), "selection.json")
	if err := writeNewJSON(path, map[string]int{"fixed": 1}); err != nil {
		t.Fatal(err)
	}
	if err := writeNewJSON(path, map[string]int{"fixed": 2}); err == nil {
		t.Fatal("overwrote frozen selection")
	}
}
