package mlx

import (
	"math"
	"testing"
)

func TestClipGradNormInvalidArguments(t *testing.T) {
	for _, limit := range []float32{-1, float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		if out, _, err := ClipGradNorm([]Array{{}}, limit); err == nil || out != nil {
			t.Fatalf("limit %g: got %v, %v", limit, out, err)
		}
	}
	if out, _, err := ClipGradNorm(nil, 1); err == nil || out != nil {
		t.Fatalf("empty gradients: got %v, %v", out, err)
	}
}
