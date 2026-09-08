package qwen2

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"

	"github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/bpe"
)

// Example contains shifted next-token pairs. LossStart masks the prompt: only
// Targets[LossStart:] contribute. Variable-length examples need no padding.
type Example struct {
	Inputs, Targets []int32
	LossStart       int
}

// ReadExamples reads JSONL records with prompt/completion fields. It rejects
// overlong records instead of silently truncating targets or changing the task.
func ReadExamples(path string, tok *bpe.Tokenizer, maxLength int) ([]Example, error) {
	if tok == nil || maxLength <= 0 {
		return nil, fmt.Errorf("qwen2: tokenizer and positive max length required")
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 4096), 4<<20)
	var examples []Example
	line := 0
	for scanner.Scan() {
		line++
		var record struct {
			Prompt     string `json:"prompt"`
			Completion string `json:"completion"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, line, err)
		}
		if record.Prompt == "" || record.Completion == "" {
			return nil, fmt.Errorf("%s:%d: prompt and completion required", path, line)
		}
		// Encode the completion separately so the assistant boundary cannot merge
		// with it; the same explicit boundary is used for the loss mask.
		prefix := tok.Encode(bpe.ChatTemplate(record.Prompt))
		end, ok := tok.SpecialID("<|im_end|>")
		if !ok {
			return nil, fmt.Errorf("qwen2: missing im_end")
		}
		ids := append(append(prefix, tok.Encode(record.Completion)...), end)
		if len(ids)-1 > maxLength {
			return nil, fmt.Errorf("%s:%d: %d tokens exceeds max length %d", path, line, len(ids)-1, maxLength)
		}
		examples = append(examples, Example{Inputs: ids[:len(ids)-1], Targets: ids[1:], LossStart: len(prefix) - 1})
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(examples) == 0 {
		return nil, fmt.Errorf("qwen2: empty dataset %s", path)
	}
	return examples, nil
}

func validateExamples(c Config, data []Example) error {
	if len(data) == 0 {
		return fmt.Errorf("qwen2: empty dataset")
	}
	for i, e := range data {
		if len(e.Inputs) == 0 || len(e.Inputs) != len(e.Targets) || len(e.Inputs) > c.MaxPositions || e.LossStart < 0 || e.LossStart >= len(e.Inputs) {
			return fmt.Errorf("qwen2: malformed example %d", i)
		}
		for _, ids := range [][]int32{e.Inputs, e.Targets} {
			for _, id := range ids {
				if id < 0 || int(id) >= c.VocabSize {
					return fmt.Errorf("qwen2: invalid token in example %d", i)
				}
			}
		}
	}
	return nil
}

func (a *Adapters) exampleLoss(w *Weights, e Example, params []mlx.Array) (mlx.Array, error) {
	s := &scope{}
	defer s.close()
	logits := s.add(forward(w, a.config, e.Inputs, nil, a.hook(params), true))
	positions := make([]int32, len(e.Targets)-e.LossStart)
	for i := range positions {
		positions[i] = int32(e.LossStart + i)
	}
	index := s.add(mlx.NewInt32(positions, []int{len(positions)}))
	logits = s.add(mlx.TakeAxis(logits, index, 1))
	logits = s.add(mlx.AsType(logits, mlx.Float32))
	targets := s.add(mlx.NewInt32(e.Targets[e.LossStart:], []int{1, len(positions)}))
	loss := s.add(mlx.CrossEntropyAxis(logits, targets, -1))
	return s.take(loss)
}

func (a *Adapters) Loss(w *Weights, data []Example) (loss float32, err error) {
	if err := a.validate(); err != nil {
		return 0, err
	}
	if err := validateExamples(a.config, data); err != nil {
		return 0, err
	}
	err = mlx.Batch(func() error {
		var total float64
		count := 0
		for _, e := range data {
			l, err := a.exampleLoss(w, e, a.Params)
			if err != nil {
				return err
			}
			d, err := l.Float32Data()
			_ = l.Close()
			if err != nil {
				return err
			}
			n := len(e.Targets) - e.LossStart
			total += float64(d[0]) * float64(n)
			count += n
		}
		loss = float32(total / float64(count))
		return finiteLoss(loss)
	})
	return
}

type TrainOptions struct {
	Steps, BatchSize          int
	LearningRate, WeightDecay float32
	Seed                      int64
	// Report runs outside Batch and may safely call MLX.
	Report func(step int, loss float32)
}

// Train updates only LoRA parameters using AdamW. Each batch accumulates
// token-weighted gradients one example at a time to bound activation memory.
// A new call starts fresh optimizer moments; saved adapters are not optimizer checkpoints.
func (a *Adapters) Train(w *Weights, data []Example, options TrainOptions) error {
	if err := a.validate(); err != nil {
		return err
	}
	if err := validateExamples(a.config, data); err != nil {
		return err
	}
	if options.Steps <= 0 || options.BatchSize <= 0 {
		return fmt.Errorf("qwen2: positive steps and batch size required")
	}
	optimizer, err := mlx.NewAdamW(options.LearningRate, options.WeightDecay)
	if err != nil {
		return err
	}
	defer optimizer.Close()
	var current Example
	argnums := make([]int, len(a.Params))
	for i := range argnums {
		argnums[i] = i
	}
	vg, err := mlx.NewValueAndGrad(func(params []mlx.Array) ([]mlx.Array, error) {
		loss, err := a.exampleLoss(w, current, params)
		if err != nil {
			return nil, err
		}
		return []mlx.Array{loss}, nil
	}, argnums...)
	if err != nil {
		return err
	}
	defer vg.Close()
	rng := rand.New(rand.NewSource(options.Seed))
	order := rng.Perm(len(data))
	cursor := 0
	for step := 0; step < options.Steps; step++ {
		var batchLoss float32
		err := mlx.Batch(func() error {
			var accumulated []mlx.Array
			defer func() { _ = mlx.CloseArrays(accumulated) }()
			tokens := 0
			lossSum := float64(0)
			for range options.BatchSize {
				if cursor == len(order) {
					order = rng.Perm(len(data))
					cursor = 0
				}
				current = data[order[cursor]]
				cursor++
				values, grads, err := vg.Apply(a.Params...)
				if err != nil {
					return err
				}
				n := len(current.Targets) - current.LossStart
				err = func() error {
					defer mlx.CloseArrays(values)
					defer mlx.CloseArrays(grads)
					if err := mlx.Eval(append(append([]mlx.Array{}, grads...), values...)...); err != nil {
						return err
					}
					d, err := values[0].Float32Data()
					if err != nil {
						return err
					}
					if err := finiteLoss(d[0]); err != nil {
						return err
					}
					lossSum += float64(d[0]) * float64(n)
					if len(accumulated) == 0 {
						for _, g := range grads {
							z, err := mlx.ZerosLike(g)
							if err != nil {
								return err
							}
							accumulated = append(accumulated, z)
						}
					}
					weight, err := mlx.NewScalarFloat32(float32(n))
					if err != nil {
						return err
					}
					defer weight.Close()
					for i, g := range grads {
						weighted, err := mlx.Multiply(g, weight)
						if err != nil {
							return err
						}
						next, err := mlx.Add(accumulated[i], weighted)
						_ = weighted.Close()
						if err != nil {
							return err
						}
						_ = accumulated[i].Close()
						accumulated[i] = next
					}
					return mlx.Eval(accumulated...)
				}()
				if err != nil {
					return err
				}
				tokens += n
			}
			normalizer, err := mlx.NewScalarFloat32(float32(tokens))
			if err != nil {
				return err
			}
			defer normalizer.Close()
			for i, g := range accumulated {
				avg, err := mlx.Divide(g, normalizer)
				if err != nil {
					return err
				}
				_ = accumulated[i].Close()
				accumulated[i] = avg
			}
			next, err := optimizer.Update(a.Params, accumulated)
			if err != nil {
				return err
			}
			_ = mlx.CloseArrays(a.Params)
			a.Params = next
			batchLoss = float32(lossSum / float64(tokens))
			return nil
		})
		if err != nil {
			return fmt.Errorf("qwen2: training step %d: %w", step+1, err)
		}
		if options.Report != nil {
			options.Report(step+1, batchLoss)
		}
	}
	return nil
}

func finiteLoss(loss float32) error {
	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		return fmt.Errorf("qwen2: nonfinite loss %g", loss)
	}
	return nil
}
