//go:build !mlx

package mlx

import (
	"errors"
	"testing"
)

func TestStubSynchronize(t *testing.T) {
	if err := Synchronize(); !errors.Is(err, errBuiltWithoutMLX) {
		t.Fatalf("unexpected error: %v", err)
	}
}
