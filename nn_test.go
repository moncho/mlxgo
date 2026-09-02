package mlx

import "testing"

func TestSGDValidatesInputs(t *testing.T) {
	if _, err := SGDWithLearningRate(nil, nil, Array{}); err == nil {
		t.Fatal("expected empty params to fail")
	}
	if _, err := SGDWithLearningRate([]Array{{}}, nil, Array{}); err == nil {
		t.Fatal("expected mismatched params and grads to fail")
	}
}

func TestCloseArraysAcceptsEmptyInput(t *testing.T) {
	if err := CloseArrays(nil); err != nil {
		t.Fatalf("CloseArrays(nil) returned error: %v", err)
	}
}
