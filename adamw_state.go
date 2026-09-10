package mlx

import (
	"fmt"
	"math"
	"slices"
)

// AdamWState is an owned snapshot. Close First and Second after saving or
// restoring it. Mutating a snapshot does not mutate the optimizer's state.
type AdamWState struct {
	Step                      int
	LearningRate, WeightDecay float32
	First, Second             []Array
}

// State returns independently owned handles to the immutable moment tensors.
func (o *AdamW) State() (state AdamWState, err error) {
	if o == nil || o.closed {
		return state, fmt.Errorf("mlxgo: AdamW is closed")
	}
	state.Step, state.LearningRate, state.WeightDecay = o.step, o.learningRate, o.weightDecay
	err = Batch(func() error {
		for _, pair := range []struct {
			in  []Array
			out *[]Array
		}{{o.first, &state.First}, {o.second, &state.Second}} {
			for _, a := range pair.in {
				copy, err := Reshape(a, a.Shape())
				if err != nil {
					return err
				}
				*pair.out = append(*pair.out, copy)
			}
		}
		return nil
	})
	if err != nil {
		_ = CloseArrays(state.First)
		_ = CloseArrays(state.Second)
		return AdamWState{}, err
	}
	return state, nil
}

// NewAdamWFromState validates and borrows state and params, returning an optimizer
// with its own moment handles. Parameters are used only to validate tensor shapes.
func NewAdamWFromState(state AdamWState, params []Array) (*AdamW, error) {
	if state.Step < 0 || state.Step == math.MaxInt || len(params) == 0 ||
		state.Step == 0 && (len(state.First) != 0 || len(state.Second) != 0) ||
		state.Step > 0 && (len(state.First) != len(params) || len(state.Second) != len(params)) {
		return nil, fmt.Errorf("mlxgo: invalid AdamW state dimensions or step")
	}
	o, err := NewAdamW(state.LearningRate, state.WeightDecay)
	if err != nil {
		return nil, err
	}
	err = Batch(func() error {
		for i, p := range params {
			dt, err := p.DType()
			if err != nil {
				return err
			}
			if dt != Float32 {
				return fmt.Errorf("mlxgo: AdamW parameter %d must be float32", i)
			}
			if state.Step == 0 {
				continue
			}
			for j, a := range []Array{state.First[i], state.Second[i]} {
				dt, err := a.DType()
				if err != nil {
					return err
				}
				if dt != Float32 || !slices.Equal(a.Shape(), p.Shape()) {
					return fmt.Errorf("mlxgo: invalid AdamW moment shape or dtype at %d", i)
				}
				values, err := a.Float32Data()
				if err != nil {
					return err
				}
				for _, v := range values {
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || j == 1 && v < 0 {
						return fmt.Errorf("mlxgo: invalid AdamW moment value at %d", i)
					}
				}
			}
		}
		copy := &AdamW{learningRate: state.LearningRate, weightDecay: state.WeightDecay, step: state.Step, first: state.First, second: state.Second}
		snapshot, err := copy.State()
		if err != nil {
			return err
		}
		o.step, o.first, o.second = snapshot.Step, snapshot.First, snapshot.Second
		return nil
	})
	if err != nil {
		_ = o.Close()
		return nil, err
	}
	return o, nil
}
