package qwen2

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"

	"github.com/moncho/mlxgo"
)

const trainingFormat = "mlxgo-qwen2-training-v1"

type trainingMetadata struct {
	Format, GoVersion                      string
	Config                                 Config
	BaseSHA256, DataSHA256                 string
	Rank                                   int
	Alpha                                  float32
	Step, BatchSize                        int
	LearningRate, WeightDecay, MaxGradNorm float32
	Seed                                   int64
	Epoch, Cursor                          int
	Order                                  []int
}

func trainingDataHash(data []Example) string {
	h := sha256.New()
	var buf [8]byte
	write := func(v int) { binary.LittleEndian.PutUint64(buf[:], uint64(v)); _, _ = h.Write(buf[:]) }
	write(len(data))
	for _, e := range data {
		write(e.LossStart)
		for _, ids := range [][]int32{e.Inputs, e.Targets} {
			write(len(ids))
			for _, id := range ids {
				write(int(id))
			}
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

func (a *Adapters) saveTraining(path string, optimizer *mlx.AdamW, order *trainingOrder, dataHash string, options TrainOptions) error {
	state, err := optimizer.State()
	if err != nil {
		return err
	}
	defer mlx.CloseArrays(state.First)
	defer mlx.CloseArrays(state.Second)
	meta := trainingMetadata{
		Format: trainingFormat, GoVersion: runtime.Version(), Config: a.config,
		BaseSHA256: a.BaseSHA256, DataSHA256: dataHash, Rank: a.Rank, Alpha: a.Alpha,
		Step: state.Step, BatchSize: options.BatchSize, LearningRate: state.LearningRate,
		WeightDecay: state.WeightDecay, MaxGradNorm: options.MaxGradNorm, Seed: options.Seed,
		Epoch: order.epoch, Cursor: order.cursor, Order: order.order,
	}
	encoded, err := json.Marshal(meta)
	if err != nil {
		return err
	}
	arrays := map[string]mlx.Array{}
	for i, p := range a.Params {
		arrays[fmt.Sprintf("lora.%d", i)] = p
		arrays[fmt.Sprintf("adam.first.%d", i)] = state.First[i]
		arrays[fmt.Sprintf("adam.second.%d", i)] = state.Second[i]
	}
	return atomicTrainingSave(path, func(temp string) error {
		return mlx.SaveSafetensors(temp, arrays, map[string]string{"training": string(encoded)})
	})
}

// Write in the destination directory so a failed save cannot truncate the last
// checkpoint and rename stays on the same filesystem. This is not a backup.
func atomicTrainingSave(path string, write func(string) error) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".mlxgo-training-*.safetensors")
	if err != nil {
		return err
	}
	temp := f.Name()
	defer os.Remove(temp)
	if err := f.Close(); err != nil {
		return err
	}
	if err := write(temp); err != nil {
		return err
	}
	f, err = os.OpenFile(temp, os.O_RDWR, 0)
	if err != nil {
		return err
	}
	err = f.Sync()
	closeErr := f.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	return os.Rename(temp, path)
}

func (m trainingMetadata) validate(a *Adapters, hash string, n int, o TrainOptions) error {
	if m.Format != trainingFormat || m.GoVersion != runtime.Version() {
		return fmt.Errorf("qwen2: incompatible training checkpoint format or Go version")
	}
	cfg, _ := json.Marshal(a.config)
	saved, _ := json.Marshal(m.Config)
	if string(cfg) != string(saved) || m.BaseSHA256 != a.BaseSHA256 || m.Rank != a.Rank || m.Alpha != a.Alpha {
		return fmt.Errorf("qwen2: training checkpoint model or adapter configuration mismatch")
	}
	if m.DataSHA256 != hash || len(m.Order) != n {
		return fmt.Errorf("qwen2: training checkpoint dataset mismatch")
	}
	if m.BatchSize != o.BatchSize || m.LearningRate != o.LearningRate || m.WeightDecay != o.WeightDecay || m.MaxGradNorm != o.MaxGradNorm || m.Seed != o.Seed {
		return fmt.Errorf("qwen2: training checkpoint settings mismatch")
	}
	if m.Step <= 0 || m.Step > math.MaxInt/o.BatchSize || o.Steps > math.MaxInt/o.BatchSize-m.Step {
		return fmt.Errorf("qwen2: invalid training checkpoint step")
	}
	consumed := m.Step * o.BatchSize
	if m.Epoch != (consumed-1)/n || m.Cursor != (consumed-1)%n+1 {
		return fmt.Errorf("qwen2: inconsistent training checkpoint data position")
	}
	seen := make([]bool, n)
	for _, i := range m.Order {
		if i < 0 || i >= n || seen[i] {
			return fmt.Errorf("qwen2: invalid training checkpoint permutation")
		}
		seen[i] = true
	}
	return nil
}

func (a *Adapters) restoreTraining(path, dataHash string, n int, options TrainOptions) (*mlx.AdamW, *trainingOrder, int, error) {
	f, err := mlx.LoadSafetensors(path)
	if err != nil {
		return nil, nil, 0, err
	}
	defer f.Close()
	text, ok, err := f.Metadata("training")
	if err != nil {
		return nil, nil, 0, err
	}
	if !ok {
		return nil, nil, 0, fmt.Errorf("qwen2: not a full training checkpoint (missing training metadata)")
	}
	var meta trainingMetadata
	if err := json.Unmarshal([]byte(text), &meta); err != nil {
		return nil, nil, 0, err
	}
	if err := meta.validate(a, dataHash, n, options); err != nil {
		return nil, nil, 0, err
	}
	var params, first, second []mlx.Array
	defer func() { _ = mlx.CloseArrays(params); _ = mlx.CloseArrays(first); _ = mlx.CloseArrays(second) }()
	for i, shape := range a.shapes() {
		for _, item := range []struct {
			name string
			dest *[]mlx.Array
		}{
			{fmt.Sprintf("lora.%d", i), &params},
			{fmt.Sprintf("adam.first.%d", i), &first},
			{fmt.Sprintf("adam.second.%d", i), &second},
		} {
			p, err := f.Get(item.name)
			if err != nil {
				return nil, nil, 0, err
			}
			*item.dest = append(*item.dest, p)
			dt, err := p.DType()
			if err != nil {
				return nil, nil, 0, err
			}
			if dt != mlx.Float32 || !slices.Equal(shape, p.Shape()) {
				return nil, nil, 0, fmt.Errorf("qwen2: invalid tensor %s", item.name)
			}
			if item.dest == &params {
				values, err := p.Float32Data()
				if err != nil {
					return nil, nil, 0, err
				}
				for _, v := range values {
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
						return nil, nil, 0, fmt.Errorf("qwen2: nonfinite adapter tensor %s", item.name)
					}
				}
			}
		}
	}
	optimizer, err := mlx.NewAdamWFromState(mlx.AdamWState{
		Step: meta.Step, LearningRate: meta.LearningRate, WeightDecay: meta.WeightDecay, First: first, Second: second,
	}, params)
	if err != nil {
		return nil, nil, 0, err
	}
	order := newTrainingOrder(meta.Seed, n)
	for i := 0; i < meta.Epoch; i++ {
		order.order = order.rng.Perm(n)
	}
	if !slices.Equal(order.order, meta.Order) {
		_ = optimizer.Close()
		return nil, nil, 0, fmt.Errorf("qwen2: checkpoint shuffle sequence mismatch")
	}
	order.epoch, order.cursor = meta.Epoch, meta.Cursor
	_ = mlx.CloseArrays(a.Params)
	a.Params, params = params, nil
	return optimizer, order, meta.Step, nil
}
