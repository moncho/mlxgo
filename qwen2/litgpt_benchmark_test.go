//go:build mlx && mlxruntime

package qwen2

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"testing"

	"github.com/moncho/mlxgo"
)

// BenchmarkLitGPTTrainingRuntime measures optimizer steps on the same float32
// fixture as TestLitGPTTrainingReference. Model loading and 20 warmup steps are
// excluded; warmup and measured steps share one optimizer and training order.
func BenchmarkLitGPTTrainingRuntime(b *testing.B) {
	b.StopTimer()
	data, err := os.ReadFile("testdata/litgpt_training.json")
	if err != nil {
		b.Fatal(err)
	}
	var f litGPTTrainingFixture
	if err := json.Unmarshal(data, &f); err != nil {
		b.Fatal(err)
	}
	if f.Format != "mlxgo-litgpt-training-v1" || len(f.Examples) != 2 || f.LoRA.Rank != 2 || f.LoRA.Alpha != 4 {
		b.Fatal("benchmark requires the fixed LitGPT training fixture")
	}
	if err := f.Config.Validate(); err != nil {
		b.Fatal(err)
	}
	examples := make([]Example, len(f.Examples))
	for i, e := range f.Examples {
		examples[i] = Example{Inputs: e.Inputs, Targets: e.Targets, LossStart: e.LossStart}
	}
	defer mlx.SetDefaultCPU()
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		b.Run(device.name, func(b *testing.B) {
			b.StopTimer()
			if err := device.set(); err != nil {
				b.Fatal(err)
			}
			w := litGPTWeights(b, f)
			a, err := NewAdapters(f.Config, f.LoRA.Rank, f.LoRA.Alpha, 42, fmt.Sprintf("%x", sha256.Sum256(data)))
			if err != nil {
				b.Fatal(err)
			}
			b.Cleanup(func() { _ = a.Close() })
			if len(f.Adapters) != len(a.Params) {
				b.Fatal("incomplete adapter fixture")
			}
			for i, shape := range a.shapes() {
				checkLitGPTTensor(b, fmt.Sprintf("adapter %d", i), f.Adapters[i], shape)
				p, err := mlx.NewFloat32(f.Adapters[i].Values, shape)
				if err != nil {
					b.Fatal(err)
				}
				_ = a.Params[i].Close()
				a.Params[i] = p
			}
			const warmup = 20
			finalStep := warmup + b.N
			var finalLoss float32
			finished := false
			o := f.Optimizer
			if err := a.Train(w, examples, TrainOptions{
				Steps: finalStep, BatchSize: len(examples), LearningRate: o.LearningRate,
				WeightDecay: o.WeightDecay, MaxGradNorm: o.MaxGradNorm, Seed: 42,
				Report: func(step int, loss float32) {
					switch step {
					case warmup:
						if err := mlx.Synchronize(); err != nil {
							b.Fatal(err)
						}
						b.ResetTimer()
						b.StartTimer()
					case finalStep:
						if err := mlx.Synchronize(); err != nil {
							b.Fatal(err)
						}
						b.StopTimer()
						finalLoss, finished = loss, true
					}
				},
			}); err != nil {
				b.Fatal(err)
			}
			if !finished {
				b.Fatal("training did not report its final step")
			}
			if err := finiteLoss(finalLoss); err != nil {
				b.Fatal(err)
			}
			b.ReportMetric(float64(finalLoss), "loss")
		})
	}
}
