//go:build mlx && mlxruntime

package qwen2

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/moncho/mlxgo"
)

// This can compare artifacts produced by separate CLI processes without loading
// base weights. It checks every adapter and moment value, including signed zero.
func TestTrainingCheckpointFilesEqual(t *testing.T) {
	full, resumed := os.Getenv("MLXGO_CHECKPOINT_FULL"), os.Getenv("MLXGO_CHECKPOINT_RESUMED")
	if full == "" || resumed == "" {
		t.Skip("set MLXGO_CHECKPOINT_FULL and MLXGO_CHECKPOINT_RESUMED")
	}
	a, b := checkpointMeta(t, full), checkpointMeta(t, resumed)
	am, err := json.Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	bm, err := json.Marshal(b)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(am, bm) {
		t.Fatal("checkpoint metadata differs")
	}
	f, err := mlx.LoadSafetensors(full)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	r, err := mlx.LoadSafetensors(resumed)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	count := 0
	for i := range 4 * a.Config.NumLayers {
		for _, prefix := range []string{"lora.", "adam.first.", "adam.second."} {
			name := fmt.Sprint(prefix, i)
			p, err := f.Get(name)
			if err != nil {
				t.Fatal(err)
			}
			defer p.Close()
			q, err := r.Get(name)
			if err != nil {
				t.Fatal(err)
			}
			defer q.Close()
			x, y := floatData(t, p), floatData(t, q)
			if !slices.Equal(p.Shape(), q.Shape()) || len(x) != len(y) {
				t.Fatal("shape mismatch", name)
			}
			for j := range x {
				if math.Float32bits(x[j]) != math.Float32bits(y[j]) {
					t.Fatal("bit mismatch", name, j)
				}
			}
			count++
		}
	}
	t.Logf("%d tensors bit-identical at step %d; optimizer and data-order metadata identical", count, a.Step)
}

func checkpointMeta(t *testing.T, path string) trainingMetadata {
	t.Helper()
	f, err := mlx.LoadSafetensors(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	text, ok, err := f.Metadata("training")
	if err != nil || !ok {
		t.Fatalf("metadata: %v", err)
	}
	var m trainingMetadata
	if err := json.Unmarshal([]byte(text), &m); err != nil {
		t.Fatal(err)
	}
	return m
}

func TestTrainingCheckpointExactResume(t *testing.T) {
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			c := tinyConfig()
			w := tinyWeights(t, c)
			newAdapters := func() *Adapters {
				a, err := NewAdapters(c, 2, 2, 42, strings.Repeat("0", 64))
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { _ = a.Close() })
				return a
			}
			data := []Example{
				{Inputs: []int32{1, 2, 3}, Targets: []int32{2, 3, 4}, LossStart: 1},
				{Inputs: []int32{4, 5}, Targets: []int32{5, 6}},
				{Inputs: []int32{6, 7, 8}, Targets: []int32{7, 8, 9}},
				{Inputs: []int32{9}, Targets: []int32{10}},
				{Inputs: []int32{10, 11}, Targets: []int32{11, 12}},
			}
			full, partial := newAdapters(), newAdapters()
			fullPath, splitPath := filepath.Join(t.TempDir(), "full.safetensors"), filepath.Join(t.TempDir(), "split.safetensors")
			options := TrainOptions{Steps: 9, BatchSize: 3, LearningRate: .003, WeightDecay: .01, MaxGradNorm: .1, Seed: 42, CheckpointPath: fullPath}
			var fullLoss, splitLoss []float32
			options.Report = func(step int, loss float32) { fullLoss = append(fullLoss, loss) }
			if err := full.Train(w, data, options); err != nil {
				t.Fatal(err)
			}
			options.Steps, options.CheckpointEvery, options.CheckpointPath = 4, 2, splitPath
			// Only inspect a periodic checkpoint after it exists.
			options.Report = func(step int, loss float32) {
				if step%2 == 0 && checkpointMeta(t, splitPath).Step != step {
					t.Fatal("periodic save missed step", step)
				}
				splitLoss = append(splitLoss, loss)
			}
			if err := partial.Train(w, data, options); err != nil {
				t.Fatal(err)
			}
			_ = partial.Close()
			resumed := newAdapters()
			options.Steps, options.ResumeFrom = 5, splitPath
			options.Report = func(step int, loss float32) {
				if step != len(splitLoss)+1 {
					t.Fatal("report step reset", step)
				}
				splitLoss = append(splitLoss, loss)
			}
			if err := resumed.Train(w, data, options); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(fullLoss, splitLoss) {
				t.Fatalf("loss trajectory differs: %v vs %v", fullLoss, splitLoss)
			}
			for i := range full.Params {
				if !slices.Equal(floatData(t, full.Params[i]), floatData(t, resumed.Params[i])) {
					t.Fatal("resumed parameter differs", i)
				}
			}
			for _, prefix := range []string{"adam.first.", "adam.second."} {
				f, err := mlx.LoadSafetensors(fullPath)
				if err != nil {
					t.Fatal(err)
				}
				defer f.Close()
				s, err := mlx.LoadSafetensors(splitPath)
				if err != nil {
					t.Fatal(err)
				}
				defer s.Close()
				for i := range full.Params {
					p, err := f.Get(fmt.Sprint(prefix, i))
					if err != nil {
						t.Fatal(err)
					}
					defer p.Close()
					q, err := s.Get(fmt.Sprint(prefix, i))
					if err != nil {
						t.Fatal(err)
					}
					defer q.Close()
					if !slices.Equal(floatData(t, p), floatData(t, q)) {
						t.Fatal("moment mismatch", prefix, i)
					}
				}
			}
			m := checkpointMeta(t, splitPath)
			if m.Step != 9 || m.Epoch != 5 || m.Cursor != 2 {
				t.Fatalf("bad final position %+v", m)
			}
		})
	}
}

func TestTrainingCheckpointRejectsMismatch(t *testing.T) {
	c := tinyConfig()
	w := tinyWeights(t, c)
	a, err := NewAdapters(c, 2, 2, 42, strings.Repeat("0", 64))
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	data := []Example{{Inputs: []int32{1, 2}, Targets: []int32{2, 3}}}
	path := filepath.Join(t.TempDir(), "state.safetensors")
	o := TrainOptions{Steps: 1, BatchSize: 1, LearningRate: .003, Seed: 42, CheckpointPath: path}
	if err := a.Train(w, data, o); err != nil {
		t.Fatal(err)
	}
	o.ResumeFrom, o.CheckpointPath = path, ""
	before := make([][]float32, len(a.Params))
	for i, p := range a.Params {
		before[i] = floatData(t, p)
	}
	assertUnchanged := func() {
		t.Helper()
		for i, p := range a.Params {
			if !slices.Equal(before[i], floatData(t, p)) {
				t.Fatal("failed load changed parameters")
			}
		}
	}
	for _, mutate := range []func(*TrainOptions){
		func(o *TrainOptions) { o.BatchSize++ }, func(o *TrainOptions) { o.LearningRate *= 2 },
		func(o *TrainOptions) { o.WeightDecay = .1 }, func(o *TrainOptions) { o.MaxGradNorm = 1 }, func(o *TrainOptions) { o.Seed++ },
	} {
		bad := o
		mutate(&bad)
		if err := a.Train(w, data, bad); err == nil {
			t.Fatal("accepted setting mismatch")
		}
		assertUnchanged()
	}
	changed := []Example{{Inputs: []int32{1, 2}, Targets: []int32{2, 4}}}
	if err := a.Train(w, changed, o); err == nil {
		t.Fatal("accepted changed dataset")
	}
	assertUnchanged()
	adapterOnly := filepath.Join(t.TempDir(), "adapter.safetensors")
	if err := a.Save(adapterOnly); err != nil {
		t.Fatal(err)
	}
	bad := o
	bad.ResumeFrom = adapterOnly
	if err := a.Train(w, data, bad); err == nil {
		t.Fatal("accepted adapter-only file")
	}
	assertUnchanged()
	for _, corruption := range []string{"version", "go-version", "step", "cursor", "order", "base", "rank", "missing", "shape", "dtype", "nan", "adapter-nan", "negative"} {
		t.Run(corruption, func(t *testing.T) {
			f, err := mlx.LoadSafetensors(path)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			m := checkpointMeta(t, path)
			arrays := map[string]mlx.Array{}
			var owned []mlx.Array
			defer func() { _ = mlx.CloseArrays(owned) }()
			for i := range a.Params {
				for _, prefix := range []string{"lora.", "adam.first.", "adam.second."} {
					name := fmt.Sprint(prefix, i)
					p, err := f.Get(name)
					if err != nil {
						t.Fatal(err)
					}
					owned = append(owned, p)
					arrays[name] = p
				}
			}
			switch corruption {
			case "version":
				m.Format = "unknown"
			case "go-version":
				m.GoVersion = "different"
			case "step":
				m.Step = math.MaxInt
			case "cursor":
				m.Cursor++
			case "order":
				m.Order[0] = -1
			case "base":
				m.BaseSHA256 = strings.Repeat("1", 64)
			case "rank":
				m.Rank++
			case "missing":
				delete(arrays, "adam.first.0")
			default:
				key := "adam.second.0"
				if corruption == "adapter-nan" {
					key = "lora.0"
				}
				values := floatData(t, arrays[key])
				shape := arrays[key].Shape()
				if corruption == "nan" || corruption == "adapter-nan" {
					values[0] = float32(math.NaN())
				}
				if corruption == "negative" {
					values[0] = -1
				}
				if corruption == "shape" {
					values, shape = []float32{0}, []int{1}
				}
				p, err := mlx.NewFloat32(values, shape)
				if err != nil {
					t.Fatal(err)
				}
				owned = append(owned, p)
				if corruption == "dtype" {
					p, err = mlx.AsType(p, mlx.Float16)
					if err != nil {
						t.Fatal(err)
					}
					owned = append(owned, p)
				}
				arrays[key] = p
			}
			encoded, err := json.Marshal(m)
			if err != nil {
				t.Fatal(err)
			}
			broken := filepath.Join(t.TempDir(), "bad.safetensors")
			if err := mlx.SaveSafetensors(broken, arrays, map[string]string{"training": string(encoded)}); err != nil {
				t.Fatal(err)
			}
			bad := o
			bad.ResumeFrom = broken
			if err := a.Train(w, data, bad); err == nil {
				t.Fatal("accepted corrupt checkpoint")
			}
			assertUnchanged()
		})
	}
	truncated := filepath.Join(t.TempDir(), "truncated.safetensors")
	if err := os.WriteFile(truncated, []byte("broken"), 0o644); err != nil {
		t.Fatal(err)
	}
	bad.ResumeFrom = truncated
	if err := a.Train(w, data, bad); err == nil {
		t.Fatal("accepted truncated checkpoint")
	}
	assertUnchanged()
}
