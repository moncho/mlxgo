package main

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"
)

func TestModes(t *testing.T) {
	for _, tc := range []struct {
		remote     bool
		path, save string
	}{{false, "", ""}, {true, "input", ""}, {false, "input", "output"}} {
		if err := run(context.Background(), tc.remote, tc.path, tc.save, "", io.Discard, io.Discard); err == nil {
			t.Fatal("accepted ambiguous mode")
		}
	}
}

func TestNoOverwrite(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "keep.json")
	if err := os.WriteFile(path, []byte("keep"), 0600); err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{path, filepath.Join(dir, "alias")} {
		if p != path {
			if err := os.Symlink(path, p); err != nil {
				t.Fatal(err)
			}
		}
		if err := writeNew(p, func(w io.Writer) error { _, e := io.WriteString(w, "overwrite"); return e }); err == nil {
			t.Fatal("overwrote existing file")
		}
	}
	b, _ := os.ReadFile(path)
	if string(b) != "keep" {
		t.Fatal("modified existing data")
	}
	failed := filepath.Join(dir, "failed.json")
	if err := writeNew(failed, func(io.Writer) error { return errors.New("failed") }); err == nil {
		t.Fatal("lost write error")
	}
	if _, err := os.Stat(failed); !os.IsNotExist(err) {
		t.Fatal("left partial output")
	}
}
