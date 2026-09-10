package qwen2

import (
	"errors"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

func TestTrainingOrderMatchesLegacy(t *testing.T) {
	for _, n := range []int{1, 3, 17} {
		rng := rand.New(rand.NewSource(42))
		order := newTrainingOrder(42, n)
		for epoch := range 8 {
			for _, want := range rng.Perm(n) {
				if got := order.next(); got != want || order.epoch != epoch {
					t.Fatalf("order changed: got %d want %d", got, want)
				}
			}
		}
	}
}

func TestTrainingDataHash(t *testing.T) {
	base := []Example{{Inputs: []int32{1, 2}, Targets: []int32{2, 3}}}
	want := trainingDataHash(base)
	for _, data := range [][]Example{
		{{Inputs: []int32{1, 2}, Targets: []int32{2, 4}}},
		{{Inputs: []int32{1, 2}, Targets: []int32{2, 3}, LossStart: 1}},
		{{Inputs: []int32{1}, Targets: []int32{2}}, {Inputs: []int32{2}, Targets: []int32{3}}},
	} {
		if trainingDataHash(data) == want {
			t.Fatal("dataset change not hashed")
		}
	}
}

func TestAtomicTrainingSave(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "checkpoint.safetensors")
	if err := os.WriteFile(path, []byte("previous"), 0o644); err != nil {
		t.Fatal(err)
	}
	wantErr := errors.New("injected save error")
	err := atomicTrainingSave(path, func(temp string) error {
		if err := os.WriteFile(temp, []byte("partial"), 0o644); err != nil {
			return err
		}
		return wantErr
	})
	if !errors.Is(err, wantErr) {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil || string(data) != "previous" {
		t.Fatalf("lost previous checkpoint: %q %v", data, err)
	}
	files, err := os.ReadDir(dir)
	if err != nil || len(files) != 1 {
		t.Fatal("temporary file leaked", err)
	}
	if err := atomicTrainingSave(path, func(temp string) error { return os.WriteFile(temp, []byte("complete"), 0o644) }); err != nil {
		t.Fatal(err)
	}
	data, err = os.ReadFile(path)
	if err != nil || string(data) != "complete" {
		t.Fatalf("replacement failed: %q %v", data, err)
	}
}
