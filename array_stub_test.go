//go:build !mlx

package mlx

import "testing"

func TestStubExplainsHowToEnableMLX(t *testing.T) {
	_, err := NewFloat32([]float32{1}, []int{1})
	if err == nil {
		t.Fatal("expected the stub build to return an error")
	}
}
