package mlx

// Linear computes x @ weight + bias.
func Linear(x, weight, bias Array) (Array, error) {
	return batchValue(func() (Array, error) {
		product, err := Matmul(x, weight)
		if err != nil {
			return Array{}, err
		}
		out, err := Add(product, bias)
		_ = product.Close()
		if err != nil {
			return Array{}, err
		}
		return out, nil
	})
}

// LinearNoBias computes x @ weight.
func LinearNoBias(x, weight Array) (Array, error) {
	return Matmul(x, weight)
}

// MSELoss computes the mean squared error between predictions and targets.
func MSELoss(predictions, targets Array) (Array, error) {
	return batchValue(func() (Array, error) {
		residuals, err := Subtract(predictions, targets)
		if err != nil {
			return Array{}, err
		}
		squared, err := Square(residuals)
		_ = residuals.Close()
		if err != nil {
			return Array{}, err
		}
		loss, err := Mean(squared, false)
		_ = squared.Close()
		if err != nil {
			return Array{}, err
		}
		return loss, nil
	})
}

// LogSoftmaxAxis computes log softmax over axis using logsumexp.
func LogSoftmaxAxis(logits Array, axis int) (Array, error) {
	return batchValue(func() (Array, error) {
		normalizer, err := LogSumExpAxis(logits, axis, true)
		if err != nil {
			return Array{}, err
		}
		out, err := Subtract(logits, normalizer)
		_ = normalizer.Close()
		if err != nil {
			return Array{}, err
		}
		return out, nil
	})
}

// SoftmaxCrossEntropyAxis computes mean cross entropy from logits and one-hot
// targets along axis.
func SoftmaxCrossEntropyAxis(logits, oneHotTargets Array, axis int) (Array, error) {
	return batchValue(func() (Array, error) {
		logProbs, err := LogSoftmaxAxis(logits, axis)
		if err != nil {
			return Array{}, err
		}
		weighted, err := Multiply(oneHotTargets, logProbs)
		_ = logProbs.Close()
		if err != nil {
			return Array{}, err
		}
		perExample, err := SumAxis(weighted, axis, false)
		_ = weighted.Close()
		if err != nil {
			return Array{}, err
		}
		negative, err := Negative(perExample)
		_ = perExample.Close()
		if err != nil {
			return Array{}, err
		}
		loss, err := Mean(negative, false)
		_ = negative.Close()
		if err != nil {
			return Array{}, err
		}
		return loss, nil
	})
}

// CrossEntropyAxis computes mean cross entropy from logits and integer class
// targets along axis.
func CrossEntropyAxis(logits, targets Array, axis int) (Array, error) {
	return batchValue(func() (Array, error) {
		logProbs, err := LogSoftmaxAxis(logits, axis)
		if err != nil {
			return Array{}, err
		}
		expandedTargets, err := ExpandDims(targets, axis)
		if err != nil {
			_ = logProbs.Close()
			return Array{}, err
		}
		picked, err := TakeAlongAxis(logProbs, expandedTargets, axis)
		_ = logProbs.Close()
		_ = expandedTargets.Close()
		if err != nil {
			return Array{}, err
		}
		squeezed, err := SqueezeAxis(picked, axis)
		_ = picked.Close()
		if err != nil {
			return Array{}, err
		}
		negative, err := Negative(squeezed)
		_ = squeezed.Close()
		if err != nil {
			return Array{}, err
		}
		loss, err := Mean(negative, false)
		_ = negative.Close()
		if err != nil {
			return Array{}, err
		}
		return loss, nil
	})
}
