package mlx

import (
	"errors"
	"testing"
)

func TestBatchRunsFunction(t *testing.T) {
	sentinel := errors.New("sentinel")
	called := false

	err := Batch(func() error {
		called = true
		return sentinel
	})
	if !called {
		t.Fatal("Batch did not call function")
	}
	if !errors.Is(err, sentinel) {
		t.Fatalf("Batch error = %v, want %v", err, sentinel)
	}
}

func TestBatchRejectsNil(t *testing.T) {
	if err := Batch(nil); err == nil {
		t.Fatal("expected nil Batch function to fail")
	}
}
