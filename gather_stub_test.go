//go:build !mlx

package mlx

import (
	"errors"
	"testing"
)

func TestGatherStubs(t *testing.T) {
	if _, err := GatherMM(Array{}, Array{}, Array{}, Array{}, false); !errors.Is(err, errBuiltWithoutMLX) {
		t.Fatal(err)
	}
	if _, err := GatherMXFP4(Array{}, Array{}, Array{}, Array{}, Array{}, false); !errors.Is(err, errBuiltWithoutMLX) {
		t.Fatal(err)
	}
}
