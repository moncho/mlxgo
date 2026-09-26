//go:build !mlx

package mlx

import (
	"errors"
	"testing"
)

func TestQuantizedStubs(t *testing.T) {
	if _, err := GetMemoryUsage(); !errors.Is(err, errBuiltWithoutMLX) {
		t.Fatal(err)
	}
	if err := ResetPeakMemory(); !errors.Is(err, errBuiltWithoutMLX) {
		t.Fatal(err)
	}
	if _, err := NewUInt8([]byte{1}, []int{1}); !errors.Is(err, errBuiltWithoutMLX) {
		t.Fatal(err)
	}
	if _, err := MXFP8Matmul(Array{}, Array{}, Array{}); !errors.Is(err, errBuiltWithoutMLX) {
		t.Fatal(err)
	}
}
