//go:build mlx && mlxruntime

package qwen2

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"testing"
	"time"

	"github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/checkpoint"
)

// TestQwenTrainingRuntime is an opt-in measurement of a local checkpoint, not a
// performance assertion. It uses the production training loop and a repeated,
// synthetic token sequence so another implementation can reproduce its work.
func TestQwenTrainingRuntime(t *testing.T) {
	const prefix = "MLXGO_TRAIN_BENCH_"
	modelDir := os.Getenv(prefix + "MODEL")
	if modelDir == "" {
		t.Skip("set MLXGO_TRAIN_BENCH_MODEL to measure a local Qwen checkpoint")
	}
	output := os.Getenv(prefix + "OUTPUT")
	if output == "" {
		t.Fatal("MLXGO_TRAIN_BENCH_OUTPUT is required")
	}
	positive := func(name string, fallback int) int {
		t.Helper()
		text := os.Getenv(prefix + name)
		if text == "" {
			return fallback
		}
		n, err := strconv.Atoi(text)
		if err != nil || n <= 0 {
			t.Fatalf("%s%s must be a positive integer", prefix, name)
		}
		return n
	}
	length := positive("LENGTH", 512)
	rounds := positive("ROUNDS", 3)
	warmup := positive("WARMUP", 3)
	steps := positive("STEPS", 5)
	device := os.Getenv(prefix + "DEVICE")
	if device == "" {
		device = "gpu"
	}
	var err error
	switch device {
	case "cpu":
		err = mlx.SetDefaultCPU()
	case "gpu":
		err = mlx.SetDefaultGPU()
	default:
		t.Fatal("MLXGO_TRAIN_BENCH_DEVICE must be cpu or gpu")
	}
	if err != nil {
		t.Fatal(err)
	}
	defer mlx.SetDefaultCPU()
	configPath := filepath.Join(modelDir, "config.json")
	c, err := LoadConfig(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if length > c.MaxPositions {
		t.Fatal("benchmark length exceeds the model context length")
	}
	configBytes, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	baseHash, err := CheckpointHash(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	w, err := Load(modelDir, c)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	// The shared comparison is FP32 throughout. Keep this conversion local to
	// the measurement rather than changing the checkpoint loader's BF16 contract.
	destinations := []*mlx.Array{&w.Embed, &w.Norm}
	for i := range w.Layers {
		l := &w.Layers[i]
		destinations = append(destinations, &l.InputNorm, &l.PostAttnNorm,
			&l.Wq, &l.Bq, &l.Wk, &l.Bk, &l.Wv, &l.Bv, &l.Wo,
			&l.Wgate, &l.Wup, &l.Wdown)
	}
	for _, dst := range destinations {
		x, err := mlx.AsType(*dst, mlx.Float32)
		if err != nil {
			t.Fatal(err)
		}
		if err := x.Eval(); err != nil {
			_ = x.Close()
			t.Fatal(err)
		}
		_ = dst.Close()
		*dst = x
	}
	tokens := make([]int32, length+1)
	for i := range tokens {
		tokens[i] = int32((int64(i)*73 + 19) % int64(c.VocabSize))
	}
	data := []Example{{Inputs: tokens[:length], Targets: tokens[1:], LossStart: length / 2}}
	adapterPath := os.Getenv(prefix + "ADAPTERS")
	newAdapters := func() *Adapters {
		t.Helper()
		var a *Adapters
		var err error
		if adapterPath == "" {
			a, err = NewAdapters(c, 8, 16, 42, baseHash)
		} else {
			a, err = LoadAdapters(adapterPath, c, baseHash)
		}
		if err != nil {
			t.Fatal(err)
		}
		if a.Rank != 8 || a.Alpha != 16 {
			_ = a.Close()
			t.Fatal("benchmark requires rank 8, alpha 16 adapters")
		}
		return a
	}
	initial := newAdapters()
	if path := os.Getenv(prefix + "SAVE_ADAPTERS"); path != "" {
		if err := initial.Save(path); err != nil {
			_ = initial.Close()
			t.Fatal(err)
		}
		adapterPath = path
	}
	count := func(arrays []mlx.Array) int64 {
		var total int64
		for _, a := range arrays {
			n := int64(1)
			for _, dim := range a.Shape() {
				n *= int64(dim)
			}
			total += n
		}
		return total
	}
	baseParameters := count(w.Arrays())
	adapterParameters := count(initial.Params)
	_ = initial.Close()
	adapterHash := ""
	if adapterPath != "" {
		adapterHash, err = checkpoint.FileHash(adapterPath)
		if err != nil {
			t.Fatal(err)
		}
	}
	options := TrainOptions{BatchSize: 1, LearningRate: 1e-4, WeightDecay: .01, Seed: 42}
	// Verify the starting loss and one optimizer update outside the timed rounds.
	probe := newAdapters()
	options.Steps = 1
	var initialLoss float32
	options.Report = func(_ int, loss float32) { initialLoss = loss }
	if err := probe.Train(w, data, options); err != nil {
		_ = probe.Close()
		t.Fatal(err)
	}
	postStepLoss, err := probe.Loss(w, data)
	if err != nil {
		_ = probe.Close()
		t.Fatal(err)
	}
	if path := os.Getenv(prefix + "VERIFY_OUT"); path != "" {
		if err := probe.Save(path); err != nil {
			_ = probe.Close()
			t.Fatal(err)
		}
	}
	_ = probe.Close()

	msPerStep := make([]float64, rounds)
	finalLosses := make([]float32, rounds)
	peakBytes := make([]uint64, rounds)
	for round := range rounds {
		a := newAdapters()
		if err := mlx.Synchronize(); err != nil {
			_ = a.Close()
			t.Fatal(err)
		}
		if err := mlx.ResetPeakMemory(); err != nil {
			_ = a.Close()
			t.Fatal(err)
		}
		var start time.Time
		var elapsed time.Duration
		options.Steps = warmup + steps
		options.Report = func(step int, loss float32) {
			if step == warmup || step == warmup+steps {
				if err := mlx.Synchronize(); err != nil {
					t.Fatal(err)
				}
				if step == warmup {
					start = time.Now()
				} else {
					elapsed = time.Since(start)
					finalLosses[round] = loss
				}
			}
		}
		if err := a.Train(w, data, options); err != nil {
			_ = a.Close()
			t.Fatal(err)
		}
		if start.IsZero() || elapsed <= 0 {
			_ = a.Close()
			t.Fatal("training did not finish its measured steps")
		}
		usage, err := mlx.GetMemoryUsage()
		_ = a.Close()
		if err != nil {
			t.Fatal(err)
		}
		peakBytes[round] = usage.PeakBytes
		msPerStep[round] = float64(elapsed) / float64(time.Millisecond) / float64(steps)
		t.Logf("%s length %d round %d: %.3f ms/step, loss %.9g", device, length, round+1, msPerStep[round], finalLosses[round])
	}
	sorted := slices.Clone(msPerStep)
	slices.Sort(sorted)
	median := sorted[len(sorted)/2]
	if len(sorted)%2 == 0 {
		median = (sorted[len(sorted)/2-1] + median) / 2
	}
	result := map[string]any{
		"format": "mlxgo-qwen-training-runtime-v1", "implementation": "Go/MLX", "device": device,
		"dtype": "float32", "checkpoint_dtype": "bfloat16", "checkpoint_sha256": baseHash,
		"config_sha256": fmt.Sprintf("%x", sha256.Sum256(configBytes)), "config": c,
		"base_parameters": baseParameters, "adapter_parameters": adapterParameters,
		"initial_adapters_sha256": adapterHash, "adapter_seed": 42,
		"lora": map[string]any{"rank": 8, "alpha": 16, "projections": []string{"q", "v"}},
		"optimizer": map[string]any{"name": "AdamW", "learning_rate": 1e-4, "weight_decay": .01,
			"beta1": .9, "beta2": .999, "epsilon": 1e-8, "max_grad_norm": 0, "finite_gradient_check": true},
		"sequence_length": length, "batch_size": 1, "input_tokens_per_step": length,
		"supervised_tokens_per_step": length - length/2,
		"dataset":                    "one repeated sequence: ids[i] = (i*73 + 19) % vocab_size; shifted targets; first floor(length/2) targets masked",
		"rounds":                     rounds, "warmup_steps": warmup, "measured_steps": steps,
		"initial_loss": initialLoss, "post_one_update_loss": postStepLoss,
		"ms_per_step": msPerStep, "median_ms_per_step": median, "final_losses": finalLosses,
		"input_tokens_per_second":      float64(length) * 1000 / median,
		"supervised_tokens_per_second": float64(length-length/2) * 1000 / median,
		"mlx_peak_active_bytes":        peakBytes,
		"memory_scope":                 "MLX active allocations during each round including warmup; excludes allocator cache and process RSS",
		"timing":                       "model loading, FP32 conversion, adapter setup, and verification excluded; one optimizer persists through warmup and measurement; device synchronized at both timing boundaries",
		"veclib_maximum_threads":       os.Getenv("VECLIB_MAXIMUM_THREADS"),
	}
	encoded, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(output, append(encoded, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
}
