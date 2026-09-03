//go:build mlx

package mlx

/*
#cgo darwin,arm64 CFLAGS: -I/opt/homebrew/include
#include <stdlib.h>
#include <stdint.h>
#include <mlx/c/mlx.h>

extern int mlxgoClosureApply(mlx_vector_array*, const mlx_vector_array, void*);
extern void mlxgoClosureFree(void*);

static inline mlx_closure mlxgo_closure_new_func_payload(void* payload) {
	return mlx_closure_new_func_payload(mlxgoClosureApply, payload, mlxgoClosureFree);
}
*/
import "C"

import (
	"errors"
	"fmt"
	"sync"
	"unsafe"
)

// Func is a Go function that can be wrapped as an MLX closure.
//
// Callback input arrays are temporary handles managed by the wrapper. Use them
// only during the callback. Returned arrays are transferred to MLX, so return
// freshly created arrays rather than arrays you intend to keep using after the
// callback returns.
type Func func([]Array) ([]Array, error)

// Closure owns an MLX function closure backed by a Go callback.
type Closure struct {
	handle C.mlx_closure
	state  *closureState
	closed bool
}

// ValueAndGrad owns an MLX value-and-gradient transform.
type ValueAndGrad struct {
	handle C.mlx_closure_value_and_grad
	fun    *Closure
	closed bool
}

type valueAndGradResult struct {
	values []Array
	grads  []Array
}

type closureState struct {
	fn Func

	mu      sync.Mutex
	lastErr error
}

func (s *closureState) setErr(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastErr = err
}

func (s *closureState) popErr() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	err := s.lastErr
	s.lastErr = nil
	return err
}

var closureRegistry = struct {
	sync.Mutex
	next uint64
	m    map[uint64]*closureState
}{
	next: 1,
	m:    make(map[uint64]*closureState),
}

// NewClosure creates an MLX closure backed by fn.
func NewClosure(fn Func) (*Closure, error) {
	if fn == nil {
		return nil, errors.New("mlxgo: closure function must not be nil")
	}

	payload, state, err := registerClosureFunc(fn)
	if err != nil {
		return nil, err
	}

	handle, err := runMLXValue(func() (C.mlx_closure, error) {
		clearMLXError()
		handle := C.mlxgo_closure_new_func_payload(payload)
		if handle.ctx == nil {
			return C.mlx_closure{}, mlxEmptyHandleError("mlx_closure_new_func_payload", "closure")
		}
		return handle, nil
	})
	if err != nil {
		freeClosurePayload(payload)
		return nil, err
	}
	return &Closure{handle: handle, state: state}, nil
}

// Apply calls the closure with inputs.
func (c *Closure) Apply(inputs ...Array) ([]Array, error) {
	if c == nil || c.closed || c.handle.ctx == nil {
		return nil, errors.New("mlxgo: closure is closed")
	}
	return runMLXValue(func() ([]Array, error) {
		inputVec, err := cArrayVector(inputs)
		if err != nil {
			return nil, err
		}
		defer C.mlx_vector_array_free(inputVec)

		clearMLXError()
		outVec := C.mlx_vector_array_new()
		if outVec.ctx == nil {
			return nil, mlxEmptyHandleError("mlx_vector_array_new", "array vector")
		}
		defer C.mlx_vector_array_free(outVec)

		clearMLXError()
		if code := C.mlx_closure_apply(&outVec, c.handle, inputVec); code != 0 {
			if c.state != nil {
				if err := c.state.popErr(); err != nil {
					return nil, err
				}
			}
			return nil, mlxError("mlx_closure_apply", int(code))
		}
		return vectorToArrays(outVec, true)
	})
}

// Close releases the closure and unregisters its Go callback.
func (c *Closure) Close() error {
	if c == nil || c.closed {
		return nil
	}
	if c.handle.ctx == nil {
		c.closed = true
		return nil
	}
	err := runMLX(func() error {
		clearMLXError()
		if code := C.mlx_closure_free(c.handle); code != 0 {
			return mlxError("mlx_closure_free", int(code))
		}
		c.handle.ctx = nil
		c.closed = true
		return nil
	})
	return err
}

// NewValueAndGrad creates a value-and-gradient transform for fn. If argnums is
// empty, gradients are computed with respect to argument 0.
func NewValueAndGrad(fn Func, argnums ...int) (*ValueAndGrad, error) {
	if len(argnums) == 0 {
		argnums = []int{0}
	}
	for _, argnum := range argnums {
		if argnum < 0 {
			return nil, fmt.Errorf("mlxgo: argnums must be non-negative, got %d", argnum)
		}
	}

	closure, err := NewClosure(fn)
	if err != nil {
		return nil, err
	}

	cargs := cInts(argnums)
	handle, err := runMLXValue(func() (C.mlx_closure_value_and_grad, error) {
		clearMLXError()
		handle := C.mlx_closure_value_and_grad_new()
		handleErr := mlxEmptyHandleError("mlx_closure_value_and_grad_new", "value-and-grad closure")
		clearMLXError()
		if code := C.mlx_value_and_grad(&handle, closure.handle, (*C.int)(unsafe.Pointer(&cargs[0])), C.size_t(len(cargs))); code != 0 {
			_ = C.mlx_closure_value_and_grad_free(handle)
			_ = closure.Close()
			return C.mlx_closure_value_and_grad{}, mlxError("mlx_value_and_grad", int(code))
		}
		if handle.ctx == nil {
			_ = closure.Close()
			return C.mlx_closure_value_and_grad{}, handleErr
		}
		return handle, nil
	})
	if err != nil {
		return nil, err
	}

	return &ValueAndGrad{handle: handle, fun: closure}, nil
}

// Apply returns the function values and gradients for inputs.
func (v *ValueAndGrad) Apply(inputs ...Array) ([]Array, []Array, error) {
	if v == nil || v.closed || v.handle.ctx == nil {
		return nil, nil, errors.New("mlxgo: value-and-grad closure is closed")
	}

	result, err := runMLXValue(func() (valueAndGradResult, error) {
		inputVec, err := cArrayVector(inputs)
		if err != nil {
			return valueAndGradResult{}, err
		}
		defer C.mlx_vector_array_free(inputVec)

		clearMLXError()
		valuesVec := C.mlx_vector_array_new()
		if valuesVec.ctx == nil {
			return valueAndGradResult{}, mlxEmptyHandleError("mlx_vector_array_new", "value array vector")
		}
		defer C.mlx_vector_array_free(valuesVec)

		clearMLXError()
		gradsVec := C.mlx_vector_array_new()
		if gradsVec.ctx == nil {
			return valueAndGradResult{}, mlxEmptyHandleError("mlx_vector_array_new", "gradient array vector")
		}
		defer C.mlx_vector_array_free(gradsVec)

		clearMLXError()
		if code := C.mlx_closure_value_and_grad_apply(&valuesVec, &gradsVec, v.handle, inputVec); code != 0 {
			if v.fun != nil && v.fun.state != nil {
				if err := v.fun.state.popErr(); err != nil {
					return valueAndGradResult{}, err
				}
			}
			return valueAndGradResult{}, mlxError("mlx_closure_value_and_grad_apply", int(code))
		}

		values, err := vectorToArrays(valuesVec, true)
		if err != nil {
			return valueAndGradResult{}, err
		}
		grads, err := vectorToArrays(gradsVec, true)
		if err != nil {
			closeArrays(values)
			return valueAndGradResult{}, err
		}
		return valueAndGradResult{values: values, grads: grads}, nil
	})
	if err != nil {
		return nil, nil, err
	}
	return result.values, result.grads, nil
}

// Close releases the value-and-gradient transform and its backing closure.
func (v *ValueAndGrad) Close() error {
	if v == nil || v.closed {
		return nil
	}
	first := runMLX(func() error {
		var first error
		if v.handle.ctx != nil {
			clearMLXError()
			if code := C.mlx_closure_value_and_grad_free(v.handle); code != 0 {
				first = mlxError("mlx_closure_value_and_grad_free", int(code))
			}
			v.handle.ctx = nil
		}
		if v.fun != nil {
			if err := v.fun.Close(); err != nil && first == nil {
				first = err
			}
		}
		return first
	})
	v.closed = true
	return first
}

// Eval evaluates arrays together.
func Eval(arrays ...Array) error {
	return withCurrentStream(func(_ C.mlx_stream) error {
		vec, err := cArrayVector(arrays)
		if err != nil {
			return err
		}
		defer C.mlx_vector_array_free(vec)

		clearMLXError()
		if code := C.mlx_eval(vec); code != 0 {
			return mlxError("mlx_eval", int(code))
		}
		return nil
	})
}

// AsyncEval schedules asynchronous evaluation of arrays.
func AsyncEval(arrays ...Array) error {
	return withCurrentStream(func(_ C.mlx_stream) error {
		vec, err := cArrayVector(arrays)
		if err != nil {
			return err
		}
		defer C.mlx_vector_array_free(vec)

		clearMLXError()
		if code := C.mlx_async_eval(vec); code != 0 {
			return mlxError("mlx_async_eval", int(code))
		}
		return nil
	})
}

func registerClosureFunc(fn Func) (unsafe.Pointer, *closureState, error) {
	payload := C.malloc(C.size_t(unsafe.Sizeof(C.uint64_t(0))))
	if payload == nil {
		return nil, nil, errors.New("mlxgo: failed to allocate closure payload")
	}
	state := &closureState{fn: fn}

	closureRegistry.Lock()
	id := closureRegistry.next
	closureRegistry.next++
	closureRegistry.m[id] = state
	closureRegistry.Unlock()

	*(*C.uint64_t)(payload) = C.uint64_t(id)
	return payload, state, nil
}

func lookupClosurePayload(payload unsafe.Pointer) *closureState {
	if payload == nil {
		return nil
	}
	id := uint64(*(*C.uint64_t)(payload))

	closureRegistry.Lock()
	defer closureRegistry.Unlock()
	return closureRegistry.m[id]
}

func freeClosurePayload(payload unsafe.Pointer) {
	if payload == nil {
		return
	}
	id := uint64(*(*C.uint64_t)(payload))

	closureRegistry.Lock()
	delete(closureRegistry.m, id)
	closureRegistry.Unlock()

	C.free(payload)
}

func vectorToArrays(vec C.mlx_vector_array, owned bool) ([]Array, error) {
	if vec.ctx == nil {
		return nil, errors.New("mlxgo: vector is empty")
	}
	size := int(C.mlx_vector_array_size(vec))
	arrays := make([]Array, size)
	for i := 0; i < size; i++ {
		var handle C.mlx_array
		clearMLXError()
		if code := C.mlx_vector_array_get(&handle, vec, C.size_t(i)); code != 0 {
			closeArrays(arrays[:i])
			return nil, fmt.Errorf("mlxgo: mlx_vector_array_get(%d): %w", i, mlxError("mlx_vector_array_get", int(code)))
		}
		if handle.ctx == nil {
			closeArrays(arrays[:i])
			return nil, fmt.Errorf("mlxgo: mlx_vector_array_get(%d) returned an empty array", i)
		}
		if owned {
			arrays[i] = newArray(handle)
		} else {
			arrays[i] = borrowedArray(handle)
		}
	}
	return arrays, nil
}

func closeArrays(arrays []Array) {
	for i := range arrays {
		_ = arrays[i].Close()
	}
}
