//go:build !mlx

package mlx

// Func is a Go function that can be wrapped as an MLX closure in the native
// build.
type Func func([]Array) ([]Array, error)

// Closure is an MLX function closure in the native build.
type Closure struct{}

// ValueAndGrad is an MLX value-and-gradient transform in the native build.
type ValueAndGrad struct{}

// NewClosure creates an MLX closure in the native build. Native callbacks run
// on the MLX worker thread; do not block them on goroutines or work that needs
// to call MLX.
func NewClosure(_ Func) (*Closure, error) {
	return nil, errBuiltWithoutMLX
}

func (*Closure) Apply(_ ...Array) ([]Array, error) {
	return nil, errBuiltWithoutMLX
}

func (*Closure) Close() error {
	return nil
}

// NewValueAndGrad creates a value-and-gradient transform in the native build.
// Native callbacks run on the MLX worker thread; do not block them on
// goroutines or work that needs to call MLX.
func NewValueAndGrad(_ Func, _ ...int) (*ValueAndGrad, error) {
	return nil, errBuiltWithoutMLX
}

func (*ValueAndGrad) Apply(_ ...Array) ([]Array, []Array, error) {
	return nil, nil, errBuiltWithoutMLX
}

func (*ValueAndGrad) Close() error {
	return nil
}

func Eval(_ ...Array) error {
	return errBuiltWithoutMLX
}

func AsyncEval(_ ...Array) error {
	return errBuiltWithoutMLX
}
