//go:build mlx && mlxruntime

package mlx

import (
	"math"
	"testing"
)

func TestAdamWAgainstScalarReference(t *testing.T) {
	if err := SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}
	o, err := NewAdamW(.01, .1)
	if err != nil {
		t.Fatal(err)
	}
	defer o.Close()
	p := mustNewFloat32(t, []float32{1}, []int{1})
	defer func() { _ = p.Close() }()
	want, m, v := 1.0, 0.0, 0.0
	for i, g := range []float64{.2, -.3, .1, 0} {
		grad := mustNewFloat32(t, []float32{float32(g)}, []int{1})
		next, err := o.Update([]Array{p}, []Array{grad})
		_ = grad.Close()
		if err != nil {
			t.Fatal(err)
		}
		_ = p.Close()
		p = next[0]
		m = .9*m + .1*g
		v = .999*v + .001*g*g
		want = want*(1-.01*.1) - .01*(m/(1-math.Pow(.9, float64(i+1))))/(math.Sqrt(v/(1-math.Pow(.999, float64(i+1))))+1e-8)
		d, err := p.Float32Data()
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(float64(d[0])-want) > 1e-6 {
			t.Fatalf("step %d got %g want %g", i, d[0], want)
		}
	}
}
