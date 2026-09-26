//go:build !mlx

package quant

import (
	mlx "github.com/moncho/mlxgo"
	"testing"
)

func TestActivationArrayNeedsNativeMLX(t *testing.T) {
	if _, _, err := QuantizeActivationArray(mlx.Array{}, FP8Activation32); err == nil {
		t.Fatal("stub quantization succeeded")
	}
	if _, err := DequantizeActivationArray(mlx.Array{}, mlx.Array{}, FP8Activation32, Float32); err == nil {
		t.Fatal("stub dequantization succeeded")
	}
}
