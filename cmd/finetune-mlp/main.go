//go:build mlx

// Command finetune-mlp demonstrates pretraining, checkpoint loading, and full
// parameter fine-tuning on two related synthetic regression tasks.
package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"

	"github.com/moncho/mlxgo"
)

var parameterNames = []string{"hidden.weight", "hidden.bias", "output.weight", "output.bias"}
var parameterShapes = [][]int{{1, 16}, {1, 16}, {16, 1}, {1, 1}}

type model struct {
	params []mlx.Array
}

type dataset struct {
	x, y mlx.Array
}

type metrics struct {
	source, before, after, reloadMaxError float32
}

func main() {
	out := flag.String("out", "checkpoints/finetune-mlp", "directory for pretrained and fine-tuned safetensors")
	device := flag.String("device", "cpu", "execution device: cpu or gpu")
	flag.Parse()
	if err := run(*out, *device, os.Stdout); err != nil {
		log.Fatal(err)
	}
}

func run(out, device string, log io.Writer) error {
	switch device {
	case "cpu":
		if err := mlx.SetDefaultCPU(); err != nil {
			return err
		}
	case "gpu":
		if err := mlx.SetDefaultGPU(); err != nil {
			return err
		}
	default:
		return fmt.Errorf("unknown device %q: use cpu or gpu", device)
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		return err
	}
	m, err := experiment(out, log)
	if err != nil {
		return err
	}
	// Fixed acceptance limits make this example a useful runtime smoke test.
	if m.source > 0.02 || m.after > 0.02 || m.after >= m.before*0.1 {
		return fmt.Errorf("learning check failed: source=%g target_before=%g target_after=%g", m.source, m.before, m.after)
	}
	if m.reloadMaxError > 1e-6 {
		return fmt.Errorf("checkpoint predictions differ by %g", m.reloadMaxError)
	}
	fmt.Fprintf(log, "PASS: held-out target MSE improved %.1fx; reload max error=%g\n", m.before/m.after, m.reloadMaxError)
	return nil
}

func experiment(out string, log io.Writer) (result metrics, err error) {
	// Training uses grid points; evaluation uses only the midpoints between them.
	sourceTrain, err := newDataset(false, false)
	if err != nil {
		return result, err
	}
	defer sourceTrain.close()
	sourceTest, err := newDataset(true, false)
	if err != nil {
		return result, err
	}
	defer sourceTest.close()
	targetTrain, err := newDataset(false, true)
	if err != nil {
		return result, err
	}
	defer targetTrain.close()
	targetTest, err := newDataset(true, true)
	if err != nil {
		return result, err
	}
	defer targetTest.close()

	pretrained, err := newModel()
	if err != nil {
		return result, err
	}
	defer pretrained.close()
	fmt.Fprintln(log, "MLP: 1 -> 16 tanh -> 1 (49 parameters), float32, full-batch SGD")
	if err := pretrained.train(sourceTrain, 600, "pretrain", log); err != nil {
		return result, err
	}
	result.source, _, err = pretrained.evaluate(sourceTest)
	if err != nil {
		return result, err
	}
	pretrainedPath := filepath.Join(out, "pretrained.safetensors")
	if err := pretrained.save(pretrainedPath, "pretrained"); err != nil {
		return result, err
	}
	// Release the original parameters: fine-tuning must start from the file.
	pretrained.close()
	adapted, err := loadModel(pretrainedPath)
	if err != nil {
		return result, err
	}
	defer adapted.close()
	result.before, _, err = adapted.evaluate(targetTest)
	if err != nil {
		return result, err
	}
	if err := adapted.train(targetTrain, 250, "finetune", log); err != nil {
		return result, err
	}
	var predictions []float32
	result.after, predictions, err = adapted.evaluate(targetTest)
	if err != nil {
		return result, err
	}
	finalPath := filepath.Join(out, "finetuned.safetensors")
	if err := adapted.save(finalPath, "finetuned"); err != nil {
		return result, err
	}
	adapted.close()
	reloaded, err := loadModel(finalPath)
	if err != nil {
		return result, err
	}
	defer reloaded.close()
	_, restored, err := reloaded.evaluate(targetTest)
	if err != nil {
		return result, err
	}
	for i := range predictions {
		result.reloadMaxError = max(result.reloadMaxError, float32(math.Abs(float64(predictions[i]-restored[i]))))
	}
	fmt.Fprintf(log, "held-out MSE: source=%.6f target_before=%.6f target_after=%.6f\n", result.source, result.before, result.after)
	fmt.Fprintf(log, "checkpoints: %s, %s\n", pretrainedPath, finalPath)
	return result, nil
}

func newDataset(heldOut, target bool) (*dataset, error) {
	n, offset := 64, 0.0
	if heldOut {
		n, offset = 63, 0.5
	}
	x, y := make([]float32, n), make([]float32, n)
	for i := range x {
		x[i] = float32(-1 + 2*(float64(i)+offset)/63)
		y[i] = float32(math.Sin(1.5 * float64(x[i])))
		if target {
			y[i] = 0.6*y[i] + 0.5
		}
	}
	xa, err := mlx.NewFloat32(x, []int{n, 1})
	if err != nil {
		return nil, err
	}
	ya, err := mlx.NewFloat32(y, []int{n, 1})
	if err != nil {
		_ = xa.Close()
		return nil, err
	}
	return &dataset{x: xa, y: ya}, nil
}

func (d *dataset) close() { _ = mlx.CloseArrays([]mlx.Array{d.x, d.y}) }

func newModel() (*model, error) {
	m := &model{}
	rng := rand.New(rand.NewSource(42))
	for i, shape := range parameterShapes {
		data := make([]float32, shape[0]*shape[1])
		if i%2 == 0 {
			limit := math.Sqrt(6 / float64(shape[0]+shape[1]))
			for j := range data {
				data[j] = float32((2*rng.Float64() - 1) * limit)
			}
		}
		a, err := mlx.NewFloat32(data, shape)
		if err != nil {
			m.close()
			return nil, err
		}
		m.params = append(m.params, a)
	}
	return m, nil
}

func (m *model) close() {
	_ = mlx.CloseArrays(m.params)
	m.params = nil
}

func forward(x mlx.Array, params []mlx.Array) (mlx.Array, error) {
	h, err := mlx.Linear(x, params[0], params[1])
	if err != nil {
		return mlx.Array{}, err
	}
	defer h.Close()
	activated, err := mlx.Tanh(h)
	if err != nil {
		return mlx.Array{}, err
	}
	defer activated.Close()
	return mlx.Linear(activated, params[2], params[3])
}

func (m *model) train(d *dataset, steps int, phase string, log io.Writer) error {
	vg, err := mlx.NewValueAndGrad(func(params []mlx.Array) ([]mlx.Array, error) {
		pred, err := forward(d.x, params)
		if err != nil {
			return nil, err
		}
		defer pred.Close()
		loss, err := mlx.MSELoss(pred, d.y)
		if err != nil {
			return nil, err
		}
		return []mlx.Array{loss}, nil
	}, 0, 1, 2, 3)
	if err != nil {
		return err
	}
	defer vg.Close()
	rate, err := mlx.NewScalarFloat32(0.05)
	if err != nil {
		return err
	}
	defer rate.Close()
	for step := 0; step < steps; step++ {
		var loss float32
		err := mlx.Batch(func() error {
			values, grads, err := vg.Apply(m.params...)
			if err != nil {
				return err
			}
			defer mlx.CloseArrays(values)
			defer mlx.CloseArrays(grads)
			next, err := mlx.SGDWithLearningRate(m.params, grads, rate)
			if err != nil {
				return err
			}
			// Evaluate each step before dropping old weights to bound the lazy graph.
			if err := mlx.Eval(append(slices.Clone(next), values...)...); err != nil {
				_ = mlx.CloseArrays(next)
				return err
			}
			data, err := values[0].Float32Data()
			if err != nil {
				_ = mlx.CloseArrays(next)
				return err
			}
			loss = data[0]
			m.close()
			m.params = next
			return finite(loss)
		})
		if err != nil {
			return fmt.Errorf("%s step %d: %w", phase, step, err)
		}
		if step == 0 || (step+1)%100 == 0 || step+1 == steps {
			fmt.Fprintf(log, "%s step=%03d train_mse=%.6f\n", phase, step+1, loss)
		}
	}
	return nil
}

func (m *model) evaluate(d *dataset) (loss float32, predictions []float32, err error) {
	err = mlx.Batch(func() error {
		pred, err := forward(d.x, m.params)
		if err != nil {
			return err
		}
		defer pred.Close()
		mse, err := mlx.MSELoss(pred, d.y)
		if err != nil {
			return err
		}
		defer mse.Close()
		data, err := mse.Float32Data()
		if err != nil {
			return err
		}
		loss = data[0]
		if err := finite(loss); err != nil {
			return err
		}
		predictions, err = pred.Float32Data()
		return err
	})
	return
}

func finite(v float32) error {
	if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
		return fmt.Errorf("non-finite loss: %g", v)
	}
	return nil
}

func (m *model) save(path, phase string) error {
	weights := make(map[string]mlx.Array, len(m.params))
	for i, name := range parameterNames {
		weights[name] = m.params[i]
	}
	return mlx.SaveSafetensors(path, weights, map[string]string{
		"architecture": "mlp-1-16-1-tanh", "phase": phase,
	})
}

func loadModel(path string) (*model, error) {
	file, err := mlx.LoadSafetensors(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	architecture, ok, err := file.Metadata("architecture")
	if err != nil {
		return nil, err
	}
	if !ok || architecture != "mlp-1-16-1-tanh" {
		return nil, fmt.Errorf("unsupported checkpoint architecture %q", architecture)
	}
	m := &model{}
	for i, name := range parameterNames {
		a, err := file.Get(name)
		if err != nil {
			m.close()
			return nil, err
		}
		m.params = append(m.params, a)
		dtype, err := a.DType()
		if err != nil || dtype != mlx.Float32 || !slices.Equal(a.Shape(), parameterShapes[i]) {
			m.close()
			return nil, fmt.Errorf("invalid checkpoint tensor %s: expected float32 %v (dtype error: %v)", name, parameterShapes[i], err)
		}
	}
	return m, nil
}
