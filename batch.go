package mlx

func batchValue[T any](fn func() (T, error)) (T, error) {
	var value T
	err := Batch(func() error {
		var err error
		value, err = fn()
		return err
	})
	return value, err
}
