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

func TestLoRAClipsAccumulatedGradient(t *testing.T) {
	if err := mlx.SetDefaultGPU(); err != nil {
		t.Fatal(err)
	}
	c := tinyConfig()
	w := tinyWeights(t, c)
	a, err := NewAdapters(c, 2, 2, 42, strings.Repeat("0", 64))
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	data := []Example{
		{Inputs: []int32{1, 2, 3}, Targets: []int32{2, 3, 4}, LossStart: 1},
		{Inputs: []int32{5, 6, 7}, Targets: []int32{6, 7, 8}},
	}
	argnums := make([]int, len(a.Params))
	before := make([][]float32, len(a.Params))
	mean := make([][]float64, len(a.Params))
	for i, p := range a.Params {
		argnums[i] = i
		before[i] = floatData(t, p)
		mean[i] = make([]float64, len(before[i]))
	}
	var current Example
	vg, err := mlx.NewValueAndGrad(func(params []mlx.Array) ([]mlx.Array, error) {
		l, err := a.exampleLoss(w, current, params)
		if err != nil {
			return nil, err
		}
		return []mlx.Array{l}, nil
	}, argnums...)
	if err != nil {
		t.Fatal(err)
	}
	defer vg.Close()
	var lossSum float64
	tokens := 0
	for _, e := range data {
		current = e
		values, grads, err := vg.Apply(a.Params...)
		if err != nil {
			t.Fatal(err)
		}
		defer mlx.CloseArrays(values)
		defer mlx.CloseArrays(grads)
		n := len(e.Targets) - e.LossStart
		tokens += n
		lossSum += float64(floatData(t, values[0])[0]) * float64(n)
		for i, g := range grads {
			for j, x := range floatData(t, g) {
				mean[i][j] += float64(x) * float64(n)
			}
		}
	}
	var squaredNorm float64
	for i := range mean {
		for j := range mean[i] {
			mean[i][j] /= float64(tokens)
			squaredNorm += mean[i][j] * mean[i][j]
		}
	}
	const limit, rate, decay = 1e-8, .01, .1
	norm := math.Sqrt(squaredNorm)
	if norm <= limit*100 {
		t.Fatalf("fixture does not exercise clipping: norm=%g", norm)
	}
	var reported float32
	if err := a.Train(w, data, TrainOptions{
		Steps: 1, BatchSize: 2, LearningRate: rate, WeightDecay: decay,
		MaxGradNorm: limit, Seed: 42,
		Report: func(step int, loss float32) {
			if step != 1 {
				t.Fatalf("unexpected step %d", step)
			}
			reported = loss
		},
	}); err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(reported)-lossSum/float64(tokens)) > 1e-5 {
		t.Fatalf("reported loss=%g want %g", reported, lossSum/float64(tokens))
	}
	for i, p := range a.Params {
		for j, got := range floatData(t, p) {
			g := mean[i][j] * limit / norm
			// The first AdamW step's bias-corrected moments are g and g*g.
			want := float64(before[i][j])*(1-rate*decay) - rate*g/(math.Abs(g)+1e-8)
			if math.Abs(float64(got)-want) > 2e-6 {
				t.Fatalf("param %d element %d: got %g want %g", i, j, got, want)
			}
		}
	}
	for _, limit := range []float32{-1, float32(math.NaN()), float32(math.Inf(1))} {
		old := slices.Clone(a.Params)
		err := a.Train(w, data, TrainOptions{Steps: 1, BatchSize: 2, LearningRate: rate, MaxGradNorm: limit})
		if err == nil || !strings.Contains(err.Error(), "max gradient norm") {
			t.Fatalf("invalid limit %g: %v", limit, err)
		}
		for i, p := range a.Params {
			if !slices.Equal(floatData(t, p), floatData(t, old[i])) {
				t.Fatal("invalid options changed parameters")
			}
		}
	}
}
