//go:build !mlx

package mlx

import "testing"

func TestStubLiveArrays(t *testing.T) {
	if got := LiveArrays(); got != 0 {
		t.Fatalf("stub count: %d", got)
	}
}
