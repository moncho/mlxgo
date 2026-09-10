package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/moncho/mlxgo/qwen2"
)

func TestCheckpointPaths(t *testing.T) {
	dir := t.TempDir()
	state := filepath.Join(dir, "state.safetensors")
	out := filepath.Join(dir, "adapter.safetensors")
	train := filepath.Join(dir, "train.jsonl")
	valid := filepath.Join(dir, "valid.jsonl")
	o := qwen2.TrainOptions{ResumeFrom: state, CheckpointPath: state}
	if err := validatePaths(dir, train, valid, out, o); err != nil {
		t.Fatal(err)
	}
	for _, output := range []string{state, train, valid, filepath.Join(dir, "model.safetensors")} {
		if err := validatePaths(dir, train, valid, output, o); err == nil {
			t.Fatal("accepted input overwrite", output)
		}
	}
	if err := os.WriteFile(state, []byte("state"), 0o644); err != nil {
		t.Fatal(err)
	}
	alias := filepath.Join(dir, "alias")
	if err := os.Symlink(state, alias); err != nil {
		t.Fatal(err)
	}
	if err := validatePaths(dir, train, valid, alias, o); err == nil {
		t.Fatal("accepted symlink overwrite")
	}
}
