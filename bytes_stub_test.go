//go:build !mlx

package mlx

import (
	"errors"
	"testing"
)

func TestStubUInt8Data(t *testing.T) {
	if _, err := (Array{}).UInt8Data(); !errors.Is(err, errBuiltWithoutMLX) {
		t.Fatal(err)
	}
}
