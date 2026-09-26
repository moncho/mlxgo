//go:build mlx && mlxruntime

package mlx

import "testing"

func TestRuntimeMemoryUsage(t *testing.T) {
	if err := SetDefaultGPU(); err != nil {
		t.Fatal(err)
	}
	defer SetDefaultCPU()
	before, err := GetMemoryUsage()
	if err != nil {
		t.Fatal(err)
	}
	x, err := Ones([]int{1024 * 1024}, Float32)
	if err != nil {
		t.Fatal(err)
	}
	defer x.Close()
	if err := x.Eval(); err != nil {
		t.Fatal(err)
	}
	if err := ResetPeakMemory(); err != nil {
		t.Fatal(err)
	}
	y, err := Multiply(x, x)
	if err != nil {
		t.Fatal(err)
	}
	defer y.Close()
	if err := y.Eval(); err != nil {
		t.Fatal(err)
	}
	after, err := GetMemoryUsage()
	if err != nil {
		t.Fatal(err)
	}
	if after.ActiveBytes < before.ActiveBytes+8<<20 || after.PeakBytes < after.ActiveBytes {
		t.Fatalf("allocator did not track arrays: before=%+v after=%+v", before, after)
	}
	x.Close()
	y.Close()
	closed, err := GetMemoryUsage()
	if err != nil {
		t.Fatal(err)
	}
	if closed.ActiveBytes > after.ActiveBytes-8<<20 {
		t.Fatalf("allocator did not release arrays: after=%+v closed=%+v", after, closed)
	}
}
