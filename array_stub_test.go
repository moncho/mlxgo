//go:build !mlx

package mlx

import "testing"

func TestStubExplainsHowToEnableMLX(t *testing.T) {
	_, err := NewFloat32([]float32{1}, []int{1})
	if err == nil {
		t.Fatal("expected the stub build to return an error")
	}
}

func TestStubLog1p(t *testing.T) {
	if _, err := Log1p(Array{}); err != errBuiltWithoutMLX {
		t.Fatalf("Log1p: got %v, want unavailable error", err)
	}
}

func TestStubSort(t *testing.T) {
	for _, fn := range []func(Array, int) (Array, error){SortAxis, ArgSortAxis} {
		if _, err := fn(Array{}, 0); err != errBuiltWithoutMLX {
			t.Fatalf("expected unavailable error, got %v", err)
		}
	}
}

func TestStubQuantizationPrimitives(t *testing.T) {
	for _, fn := range []func(Array, Array) (Array, error){BitwiseAnd, BitwiseOr, LeftShift, RightShift} {
		if _, err := fn(Array{}, Array{}); err != errBuiltWithoutMLX {
			t.Fatalf("got %v", err)
		}
	}
	if _, err := View(Array{}, UInt32); err != errBuiltWithoutMLX {
		t.Fatalf("got %v", err)
	}
	if _, err := MaxAxis(Array{}, 0, true); err != errBuiltWithoutMLX {
		t.Fatalf("got %v", err)
	}
}
