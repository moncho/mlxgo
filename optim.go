package mlx

import "fmt"

// CloseArrays closes every array in arrays and returns the first close error.
func CloseArrays(arrays []Array) error {
	var first error
	if err := Batch(func() error {
		for i := range arrays {
			if err := arrays[i].Close(); err != nil && first == nil {
				first = err
			}
		}
		return nil
	}); err != nil && first == nil {
		first = err
	}
	return first
}

// SGD updates params with params - learningRate*grads.
func SGD(params, grads []Array, learningRate float32) ([]Array, error) {
	return batchValue(func() ([]Array, error) {
		rate, err := NewScalarFloat32(learningRate)
		if err != nil {
			return nil, err
		}
		defer rate.Close()

		return SGDWithLearningRate(params, grads, rate)
	})
}

// SGDWithLearningRate updates params with params - learningRate*grads, using an
// MLX scalar or broadcastable learning-rate array.
func SGDWithLearningRate(params, grads []Array, learningRate Array) ([]Array, error) {
	if len(params) == 0 {
		return nil, fmt.Errorf("mlxgo: params must not be empty")
	}
	if len(params) != len(grads) {
		return nil, fmt.Errorf("mlxgo: params length %d does not match grads length %d", len(params), len(grads))
	}

	return batchValue(func() ([]Array, error) {
		next := make([]Array, 0, len(params))
		for i := range params {
			update, err := Multiply(grads[i], learningRate)
			if err != nil {
				_ = CloseArrays(next)
				return nil, fmt.Errorf("mlxgo: parameter %d update: %w", i, err)
			}
			updated, err := Subtract(params[i], update)
			_ = update.Close()
			if err != nil {
				_ = CloseArrays(next)
				return nil, fmt.Errorf("mlxgo: parameter %d apply update: %w", i, err)
			}
			next = append(next, updated)
		}
		return next, nil
	})
}
