//go:build mlx && mlxruntime

package qwen2

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"slices"
	"testing"

	"github.com/moncho/mlxgo"
)

type litGPTTensor struct {
	Shape  []int     `json:"shape"`
	Values []float32 `json:"values"`
}

type litGPTTrainingFixture struct {
	Format string          `json:"format"`
	Source json.RawMessage `json:"source"`
	Config Config          `json:"config"`
	LoRA   struct {
		Rank  int     `json:"rank"`
		Alpha float32 `json:"alpha"`
	} `json:"lora"`
	Optimizer struct {
		LearningRate float32 `json:"learning_rate"`
		WeightDecay  float32 `json:"weight_decay"`
		Beta1        float32 `json:"beta1"`
		Beta2        float32 `json:"beta2"`
		Epsilon      float32 `json:"epsilon"`
		MaxGradNorm  float32 `json:"max_grad_norm"`
	} `json:"optimizer"`
	Weights  map[string]litGPTTensor `json:"weights"`
	Adapters []litGPTTensor          `json:"adapters"`
	Examples []struct {
		Inputs    []int32      `json:"inputs"`
		Targets   []int32      `json:"targets"`
		LossStart int          `json:"loss_start"`
		Logits    litGPTTensor `json:"logits"`
		Loss      float32      `json:"loss"`
	} `json:"examples"`
	Loss            float32        `json:"loss"`
	Gradients       []litGPTTensor `json:"gradients"`
	UpdatedAdapters []litGPTTensor `json:"updated_adapters"`
	PostStepLoss    float32        `json:"post_step_loss"`
}

func TestLitGPTTrainingReference(t *testing.T) {
	data, err := os.ReadFile("testdata/litgpt_training.json")
	if err != nil {
		t.Fatal(err)
	}
	var f litGPTTrainingFixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	c, o := f.Config, f.Optimizer
	if f.Format != "mlxgo-litgpt-training-v1" {
		t.Fatalf("unsupported fixture format %q", f.Format)
	}
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	if c.HiddenSize != 8 || c.IntermediateSize != 12 || c.NumLayers != 2 || c.NumHeads != 2 || c.NumKVHeads != 1 || c.VocabSize != 16 || c.MaxPositions != 32 || f.LoRA.Rank != 2 || f.LoRA.Alpha != 4 {
		t.Fatal("fixture must exercise the fixed tiny GQA model and scaled rank-2 adapters")
	}
	// Train uses these AdamW defaults. Keep acceptance limits in the test rather
	// than allowing a regenerated fixture to relax them.
	if o.LearningRate != .01 || o.WeightDecay != .1 || o.Beta1 != .9 || o.Beta2 != .999 || o.Epsilon != 1e-8 || o.MaxGradNorm != 0 {
		t.Fatal("fixture optimizer does not match the training reference settings")
	}
	if len(f.Weights) != 2+12*c.NumLayers || len(f.Adapters) != 4*c.NumLayers || len(f.Gradients) != len(f.Adapters) || len(f.UpdatedAdapters) != len(f.Adapters) || len(f.Examples) != 2 {
		t.Fatal("incomplete training fixture")
	}
	examples := make([]Example, len(f.Examples))
	for i, e := range f.Examples {
		examples[i] = Example{Inputs: e.Inputs, Targets: e.Targets, LossStart: e.LossStart}
		checkLitGPTTensor(t, fmt.Sprintf("example %d logits", i), e.Logits, []int{1, len(e.Inputs), c.VocabSize})
	}
	if err := validateExamples(c, examples); err != nil {
		t.Fatal(err)
	}
	if len(examples[0].Inputs) == len(examples[1].Inputs) || len(examples[0].Targets)-examples[0].LossStart == len(examples[1].Targets)-examples[1].LossStart || examples[0].LossStart+examples[1].LossStart == 0 {
		t.Fatal("fixture must exercise variable lengths, masking, and unequal completion-token counts")
	}
	defer mlx.SetDefaultCPU()
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			w := litGPTWeights(t, f)
			base := make([][]float32, len(w.Arrays()))
			for i, p := range w.Arrays() {
				base[i] = slices.Clone(floatData(t, p))
			}
			a, err := NewAdapters(c, f.LoRA.Rank, f.LoRA.Alpha, 42, fmt.Sprintf("%x", sha256.Sum256(data)))
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = a.Close() })
			for i, shape := range a.shapes() {
				name := fmt.Sprintf("adapter %d", i)
				checkLitGPTTensor(t, name, f.Adapters[i], shape)
				checkLitGPTTensor(t, name+" gradient", f.Gradients[i], shape)
				checkLitGPTTensor(t, name+" updated", f.UpdatedAdapters[i], shape)
				if !slices.ContainsFunc(f.Adapters[i].Values, func(v float32) bool { return v != 0 }) || !slices.ContainsFunc(f.Gradients[i].Values, func(v float32) bool { return v != 0 }) {
					t.Fatalf("%s must have nonzero initial values and gradients", name)
				}
				p, err := mlx.NewFloat32(f.Adapters[i].Values, shape)
				if err != nil {
					t.Fatal(err)
				}
				_ = a.Params[i].Close()
				a.Params[i] = p
			}
			for i, e := range examples {
				logits, err := forward(w, c, e.Inputs, nil, a.hook(a.Params), true)
				if err != nil {
					t.Fatal(err)
				}
				checkLitGPTArray(t, fmt.Sprintf("example %d logits", i), logits, f.Examples[i].Logits, 2e-6, 2e-5)
				_ = logits.Close()
				loss, err := a.Loss(w, []Example{e})
				if err != nil {
					t.Fatal(err)
				}
				checkLitGPTValues(t, fmt.Sprintf("example %d loss", i), []float32{loss}, []float32{f.Examples[i].Loss}, 2e-6, 2e-5)
			}
			loss, err := a.Loss(w, examples)
			if err != nil {
				t.Fatal(err)
			}
			checkLitGPTValues(t, "token-weighted loss", []float32{loss}, []float32{f.Loss}, 2e-6, 2e-5)
			checkLitGPTGradients(t, w, a, examples, f.Gradients)
			reports := 0
			if err := a.Train(w, examples, TrainOptions{
				Steps: 1, BatchSize: len(examples), LearningRate: o.LearningRate,
				WeightDecay: o.WeightDecay, MaxGradNorm: o.MaxGradNorm, Seed: 42,
				Report: func(step int, loss float32) {
					reports++
					if step != 1 {
						t.Fatalf("reported step %d, want 1", step)
					}
					checkLitGPTValues(t, "reported loss", []float32{loss}, []float32{f.Loss}, 2e-6, 2e-5)
				},
			}); err != nil {
				t.Fatal(err)
			}
			if reports != 1 {
				t.Fatalf("reported %d steps, want 1", reports)
			}
			for i, p := range a.Params {
				checkLitGPTArray(t, fmt.Sprintf("updated adapter %d", i), p, f.UpdatedAdapters[i], 2e-6, 2e-5)
			}
			loss, err = a.Loss(w, examples)
			if err != nil {
				t.Fatal(err)
			}
			checkLitGPTValues(t, "post-step loss", []float32{loss}, []float32{f.PostStepLoss}, 2e-6, 2e-5)
			for i, p := range w.Arrays() {
				if !slices.Equal(base[i], floatData(t, p)) {
					t.Fatalf("frozen base tensor %d changed", i)
				}
			}
		})
	}
}

func litGPTWeights(t testing.TB, f litGPTTrainingFixture) *Weights {
	t.Helper()
	c := f.Config
	w := &Weights{Layers: make([]Layer, c.NumLayers)}
	t.Cleanup(func() { _ = w.Close() })
	load := func(name string, shape []int) mlx.Array {
		t.Helper()
		tensor, ok := f.Weights[name]
		if !ok {
			t.Fatalf("missing weight %s", name)
		}
		checkLitGPTTensor(t, name, tensor, shape)
		a, err := mlx.NewFloat32(tensor.Values, shape)
		if err != nil {
			t.Fatal(err)
		}
		return a
	}
	w.Embed = load("embed", []int{c.VocabSize, c.HiddenSize})
	w.Norm = load("norm", []int{c.HiddenSize})
	kv := c.NumKVHeads * c.HeadDim()
	for i := range w.Layers {
		l := &w.Layers[i]
		for _, item := range []struct {
			dst   *mlx.Array
			name  string
			shape []int
		}{
			{&l.InputNorm, "input_norm", []int{c.HiddenSize}},
			{&l.PostAttnNorm, "post_attention_norm", []int{c.HiddenSize}},
			{&l.Wq, "wq", []int{c.HiddenSize, c.HiddenSize}},
			{&l.Bq, "bq", []int{c.HiddenSize}},
			{&l.Wk, "wk", []int{c.HiddenSize, kv}},
			{&l.Bk, "bk", []int{kv}},
			{&l.Wv, "wv", []int{c.HiddenSize, kv}},
			{&l.Bv, "bv", []int{kv}},
			{&l.Wo, "wo", []int{c.HiddenSize, c.HiddenSize}},
			{&l.Wgate, "wgate", []int{c.HiddenSize, c.IntermediateSize}},
			{&l.Wup, "wup", []int{c.HiddenSize, c.IntermediateSize}},
			{&l.Wdown, "wdown", []int{c.IntermediateSize, c.HiddenSize}},
		} {
			*item.dst = load(fmt.Sprintf("layers.%d.%s", i, item.name), item.shape)
		}
	}
	return w
}

func checkLitGPTGradients(t *testing.T, w *Weights, a *Adapters, examples []Example, want []litGPTTensor) {
	t.Helper()
	argnums := make([]int, len(a.Params))
	mean := make([][]float64, len(a.Params))
	for i := range argnums {
		argnums[i] = i
		mean[i] = make([]float64, len(want[i].Values))
	}
	var current Example
	vg, err := mlx.NewValueAndGrad(func(params []mlx.Array) ([]mlx.Array, error) {
		loss, err := a.exampleLoss(w, current, params)
		if err != nil {
			return nil, err
		}
		return []mlx.Array{loss}, nil
	}, argnums...)
	if err != nil {
		t.Fatal(err)
	}
	defer vg.Close()
	tokens := 0
	for _, e := range examples {
		current = e
		values, grads, err := vg.Apply(a.Params...)
		if err != nil {
			t.Fatal(err)
		}
		defer mlx.CloseArrays(values)
		defer mlx.CloseArrays(grads)
		if len(grads) != len(want) {
			t.Fatalf("got %d gradients, want %d", len(grads), len(want))
		}
		n := len(e.Targets) - e.LossStart
		tokens += n
		for i, g := range grads {
			if !slices.Equal(g.Shape(), want[i].Shape) {
				t.Fatalf("gradient %d shape %v, want %v", i, g.Shape(), want[i].Shape)
			}
			for j, x := range floatData(t, g) {
				mean[i][j] += float64(x) * float64(n)
			}
		}
	}
	for i, values := range mean {
		got := make([]float32, len(values))
		for j, x := range values {
			got[j] = float32(x / float64(tokens))
		}
		checkLitGPTValues(t, fmt.Sprintf("gradient %d", i), got, want[i].Values, 2e-6, 2e-4)
	}
}

func checkLitGPTTensor(t testing.TB, name string, tensor litGPTTensor, shape []int) {
	t.Helper()
	n := 1
	for _, d := range shape {
		n *= d
	}
	if !slices.Equal(tensor.Shape, shape) || len(tensor.Values) != n {
		t.Fatalf("%s: shape %v and %d values, want %v and %d", name, tensor.Shape, len(tensor.Values), shape, n)
	}
	for i, v := range tensor.Values {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("%s element %d is nonfinite", name, i)
		}
	}
}

func checkLitGPTArray(t *testing.T, name string, got mlx.Array, want litGPTTensor, absolute, relative float64) {
	t.Helper()
	if !slices.Equal(got.Shape(), want.Shape) {
		t.Fatalf("%s: shape %v, want %v", name, got.Shape(), want.Shape)
	}
	checkLitGPTValues(t, name, floatData(t, got), want.Values, absolute, relative)
}

func checkLitGPTValues(t *testing.T, name string, got, want []float32, absolute, relative float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", name, len(got), len(want))
	}
	for i, x := range got {
		y := want[i]
		difference := math.Abs(float64(x) - float64(y))
		limit := absolute + relative*math.Abs(float64(y))
		if math.IsNaN(difference) || math.IsInf(difference, 0) || difference > limit {
			t.Fatalf("%s element %d: got %.9g, want %.9g (error %g, limit %g)", name, i, x, y, difference, limit)
		}
	}
}
