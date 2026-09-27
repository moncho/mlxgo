//go:build mlx && mlxruntime

package mlx

import (
	"sync"
	"testing"
)

func TestRuntimeLiveArrays(t *testing.T) {
	base := LiveArrays()
	a := mustNewFloat32(t, []float32{1}, []int{1})
	b := mustNewFloat32(t, []float32{2}, []int{1})
	c := mustNewFloat32(t, []float32{3}, []int{1})
	defer CloseArrays([]Array{a, b, c})
	if got := LiveArrays(); got != base+3 {
		t.Fatalf("created: %d, baseline %d", got, base)
	}
	a.Close()
	b.Close()
	if got := LiveArrays(); got != base+1 {
		t.Fatalf("closed two: %d, baseline %d", got, base)
	}
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			copy := c
			if err := copy.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	wg.Wait()
	if got := LiveArrays(); got != base {
		t.Fatalf("closed all: %d, baseline %d", got, base)
	}
	var empty Array
	empty.Close()
	c.Close()
	if got := LiveArrays(); got != base {
		t.Fatalf("idempotent close: %d, baseline %d", got, base)
	}
}

func TestRuntimeLiveArraysErrorAndBorrow(t *testing.T) {
	base := LiveArrays()
	a := mustNewFloat32(t, []float32{1, 2}, []int{2})
	b := mustNewFloat32(t, []float32{1, 2, 3}, []int{3})
	defer CloseArrays([]Array{a, b})
	handle, err := a.handleValue()
	if err != nil {
		t.Fatal(err)
	}
	borrowed := borrowedArray(handle)
	borrowed.Close()
	if _, err := Add(a, b); err == nil {
		t.Fatal("expected shape error")
	}
	if got := LiveArrays(); got != base+2 {
		t.Fatalf("error/borrow changed count: %d, baseline %d", got, base)
	}
	a.Close()
	b.Close()
	if got := LiveArrays(); got != base {
		t.Fatalf("cleanup: %d, baseline %d", got, base)
	}
}
