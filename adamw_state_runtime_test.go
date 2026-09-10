//go:build mlx && mlxruntime

package mlx

import (
	"slices"
	"testing"
)

func TestAdamWStateOwnershipAndResume(t *testing.T) {
	p := mustNewFloat32(t, []float32{1, 2}, []int{2})
	defer p.Close()
	g := mustNewFloat32(t, []float32{.2, -.3}, []int{2})
	defer g.Close()
	o, err := NewAdamW(.01, .1)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Close()
	zero, err := o.State()
	if err != nil {
		t.Fatal(err)
	}
	z, err := NewAdamWFromState(zero, []Array{p})
	if err != nil {
		t.Fatal(err)
	}
	_ = z.Close()
	next, err := o.Update([]Array{p}, []Array{g})
	if err != nil {
		t.Fatal(err)
	}
	defer CloseArrays(next)
	state, err := o.State()
	if err != nil {
		t.Fatal(err)
	}
	resumed, err := NewAdamWFromState(state, next)
	if err != nil {
		t.Fatal(err)
	}
	defer resumed.Close()
	_ = CloseArrays(state.First)
	_ = CloseArrays(state.Second)
	want, err := o.Update(next, []Array{g})
	if err != nil {
		t.Fatal(err)
	}
	defer CloseArrays(want)
	_ = o.Close()
	got, err := resumed.Update(next, []Array{g})
	if err != nil {
		t.Fatal(err)
	}
	defer CloseArrays(got)
	a, err := want[0].Float32Data()
	if err != nil {
		t.Fatal(err)
	}
	b, err := got[0].Float32Data()
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(a, b) {
		t.Fatalf("resumed optimizer differs %v vs %v", a, b)
	}
	if _, err := o.State(); err == nil {
		t.Fatal("closed optimizer snapshot accepted")
	}
}
