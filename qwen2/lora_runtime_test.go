//go:build mlx && mlxruntime

package qwen2

import (
	"math"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/moncho/mlxgo"
)

func TestLoRAGradientFiniteDifference(t *testing.T) {
	c := tinyConfig()
	w := tinyWeights(t, c)
	a, err := NewAdapters(c, 2, 2, 42, strings.Repeat("0", 64))
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	e := Example{Inputs: []int32{1, 2, 3}, Targets: []int32{2, 3, 4}, LossStart: 1}
	masked := Example{Inputs: slices.Clone(e.Inputs), Targets: []int32{19, 3, 4}, LossStart: 1}
	firstLoss, err := a.Loss(w, []Example{e})
	if err != nil {
		t.Fatal(err)
	}
	maskedLoss, err := a.Loss(w, []Example{masked})
	if err != nil {
		t.Fatal(err)
	}
	if firstLoss != maskedLoss {
		t.Fatal("masked prompt target changed the loss")
	}
	vg, err := mlx.NewValueAndGrad(func(p []mlx.Array) ([]mlx.Array, error) {
		l, err := a.exampleLoss(w, e, p)
		if err != nil {
			return nil, err
		}
		return []mlx.Array{l}, nil
	}, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer vg.Close()
	values, grads, err := vg.Apply(a.Params...)
	if err != nil {
		t.Fatal(err)
	}
	defer mlx.CloseArrays(values)
	defer mlx.CloseArrays(grads)
	g := floatData(t, grads[0])
	at := 0
	for i := range g {
		if math.Abs(float64(g[i])) > math.Abs(float64(g[at])) {
			at = i
		}
	}
	if math.Abs(float64(g[at])) < 1e-5 {
		t.Fatal("unexpected zero gradient")
	}
	original := a.Params[1]
	data := floatData(t, original)
	eval := func(delta float32) float32 {
		t.Helper()
		copyData := slices.Clone(data)
		copyData[at] += delta
		p, err := mlx.NewFloat32(copyData, original.Shape())
		if err != nil {
			t.Fatal(err)
		}
		a.Params[1] = p
		loss, err := a.Loss(w, []Example{e})
		a.Params[1] = original
		_ = p.Close()
		if err != nil {
			t.Fatal(err)
		}
		return loss
	}
	numeric := (eval(.002) - eval(-.002)) / .004
	if math.Abs(float64(numeric-g[at])) > max(.002, math.Abs(float64(g[at]))*.03) {
		t.Fatalf("gradient=%g finite difference=%g", g[at], numeric)
	}
}

func TestLoRATrainsAndPreservesBase(t *testing.T) {
	if err := mlx.SetDefaultGPU(); err != nil {
		t.Fatal(err)
	}
	c := tinyConfig()
	w := tinyWeights(t, c)
	base := make([][]float32, len(w.Arrays()))
	for i, p := range w.Arrays() {
		base[i] = floatData(t, p)
	}
	a, err := NewAdapters(c, 4, 4, 42, strings.Repeat("0", 64))
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	ids := []int32{1, 2, 3}
	x, err := Forward(w, c, ids, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer x.Close()
	y, err := a.Forward(w, ids, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer y.Close()
	maxError(t, floatData(t, x), floatData(t, y), 0)
	train := []Example{{Inputs: []int32{1, 2, 3}, Targets: []int32{7, 7, 7}}, {Inputs: []int32{2, 3}, Targets: []int32{7, 7}}}
	valid := []Example{{Inputs: []int32{4, 5, 6}, Targets: []int32{7, 7, 7}}}
	before, err := a.Loss(w, valid)
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Train(w, train, TrainOptions{Steps: 60, BatchSize: 2, LearningRate: .02, Seed: 42}); err != nil {
		t.Fatal(err)
	}
	after, err := a.Loss(w, valid)
	if err != nil {
		t.Fatal(err)
	}
	if after >= before*.8 {
		t.Fatalf("held-out loss did not improve enough: %g -> %g", before, after)
	}
	for i, p := range w.Arrays() {
		if !slices.Equal(base[i], floatData(t, p)) {
			t.Fatalf("base tensor %d changed", i)
		}
	}
	path := filepath.Join(t.TempDir(), "adapters.safetensors")
	if err := a.Save(path); err != nil {
		t.Fatal(err)
	}
	restored, err := LoadAdapters(path, c, a.BaseSHA256)
	if err != nil {
		t.Fatal(err)
	}
	defer restored.Close()
	p, err := a.Forward(w, ids, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	q, err := restored.Forward(w, ids, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
	maxError(t, floatData(t, p), floatData(t, q), 0)
	if wrong, err := LoadAdapters(path, c, strings.Repeat("1", 64)); err == nil {
		_ = wrong.Close()
		t.Fatal("accepted a different base checkpoint")
	}
	t.Logf("held-out loss %g -> %g; frozen base and reload verified", before, after)
}
