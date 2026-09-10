package mlx

import (
	"fmt"
	"math"
	"slices"
)

// AdamW owns float32 first/second moments. It uses beta1=.9, beta2=.999 and
// epsilon=1e-8. Like model parameters, one optimizer must not be used concurrently.
type AdamW struct {
	learningRate, weightDecay float32
	first, second             []Array
	step                      int
	closed                    bool
}

func NewAdamW(learningRate, weightDecay float32) (*AdamW, error) {
	if learningRate <= 0 || weightDecay < 0 || math.IsNaN(float64(learningRate)) || math.IsInf(float64(learningRate), 0) || math.IsNaN(float64(weightDecay)) || math.IsInf(float64(weightDecay), 0) {
		return nil, fmt.Errorf("mlxgo: invalid AdamW learning rate or weight decay")
	}
	return &AdamW{learningRate: learningRate, weightDecay: weightDecay}, nil
}

// Update returns new float32 parameters and advances optimizer state only after
// successful evaluation. It borrows params/grads; callers close old parameters.
func (o *AdamW) Update(params, grads []Array) ([]Array, error) {
	if o == nil || o.closed {
		return nil, fmt.Errorf("mlxgo: AdamW is closed")
	}
	if o.step == math.MaxInt {
		return nil, fmt.Errorf("mlxgo: AdamW step overflow")
	}
	if len(params) == 0 || len(params) != len(grads) || o.step > 0 && len(params) != len(o.first) {
		return nil, fmt.Errorf("mlxgo: incompatible AdamW parameter count")
	}
	return batchValue(func() ([]Array, error) {
		for i := range params {
			dt, err := params[i].DType()
			if err != nil {
				return nil, err
			}
			gdt, err := grads[i].DType()
			if err != nil {
				return nil, err
			}
			if dt != Float32 || gdt != Float32 || !slices.Equal(params[i].Shape(), grads[i].Shape()) || o.step > 0 && !slices.Equal(params[i].Shape(), o.first[i].Shape()) {
				return nil, fmt.Errorf("mlxgo: AdamW parameter %d requires matching float32 tensors", i)
			}
		}
		var next, first, second []Array
		ok := false
		defer func() {
			if !ok {
				_ = CloseArrays(next)
				_ = CloseArrays(first)
				_ = CloseArrays(second)
			}
		}()
		for i := range params {
			var m, v Array
			if o.step > 0 {
				m, v = o.first[i], o.second[i]
			} else {
				var err error
				m, err = ZerosLike(params[i])
				if err != nil {
					return nil, err
				}
				defer m.Close()
				v, err = ZerosLike(params[i])
				if err != nil {
					return nil, err
				}
				defer v.Close()
			}
			p, nm, nv, err := adamUpdate(params[i], grads[i], m, v, o.step+1, o.learningRate, o.weightDecay)
			if err != nil {
				return nil, err
			}
			next = append(next, p)
			first = append(first, nm)
			second = append(second, nv)
		}
		all := append(append(append([]Array{}, next...), first...), second...)
		if err := Eval(all...); err != nil {
			return nil, err
		}
		_ = CloseArrays(o.first)
		_ = CloseArrays(o.second)
		o.first, o.second = first, second
		o.step++
		ok = true
		return next, nil
	})
}

func (o *AdamW) Close() error {
	if o == nil || o.closed {
		return nil
	}
	o.closed = true
	return CloseArrays(append(append([]Array{}, o.first...), o.second...))
}

func adamUpdate(p, g, m, v Array, step int, rate, decay float32) (Array, Array, Array, error) {
	var temps []Array
	var firstErr error
	add := func(a Array, err error) Array {
		if err != nil && firstErr == nil {
			firstErr = err
		}
		if err == nil {
			temps = append(temps, a)
		}
		return a
	}
	defer func() { _ = CloseArrays(temps) }()
	scale := func(a Array, f float32) Array { scalar := add(NewScalarFloat32(f)); return add(Multiply(a, scalar)) }
	m1 := scale(m, .9)
	m2 := scale(g, .1)
	nm := add(Add(m1, m2))
	v1 := scale(v, .999)
	squared := add(Square(g))
	v2 := scale(squared, .001)
	nv := add(Add(v1, v2))
	mhat := scale(nm, float32(1/(1-math.Pow(.9, float64(step)))))
	vhat := scale(nv, float32(1/(1-math.Pow(.999, float64(step)))))
	denom := add(Sqrt(vhat))
	eps := add(NewScalarFloat32(1e-8))
	denom = add(Add(denom, eps))
	update := add(Divide(mhat, denom))
	update = scale(update, rate)
	decayed := scale(p, 1-rate*decay)
	next := add(Subtract(decayed, update))
	if firstErr != nil {
		return Array{}, Array{}, Array{}, firstErr
	}
	// Duplicate output handles so deferred cleanup releases intermediates only.
	nextOut, err := Reshape(next, next.Shape())
	if err != nil {
		return Array{}, Array{}, Array{}, err
	}
	mOut, err := Reshape(nm, nm.Shape())
	if err != nil {
		_ = nextOut.Close()
		return Array{}, Array{}, Array{}, err
	}
	vOut, err := Reshape(nv, nv.Shape())
	if err != nil {
		_ = CloseArrays([]Array{nextOut, mOut})
		return Array{}, Array{}, Array{}, err
	}
	return nextOut, mOut, vOut, nil
}
