//go:build mlx

package mlx

/*
#cgo darwin,arm64 CFLAGS: -I/opt/homebrew/include
#cgo darwin,arm64 LDFLAGS: -L/opt/homebrew/lib -lmlxc
#include <stdlib.h>
#include <mlx/c/mlx.h>
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

// Array owns an MLX C array handle. Copies share ownership state, so closing
// any copy marks every copy closed.
type Array struct {
	state *arrayState
}

type arrayState struct {
	handle C.mlx_array
	owned  bool
	closed bool
}

// SafeTensors owns maps loaded from a safetensors file.
type SafeTensors struct {
	arrays   C.mlx_map_string_to_array
	metadata C.mlx_map_string_to_string
	closed   bool
}

// NewFloat32 copies data into a new MLX float32 array with the provided shape.
func NewFloat32(data []float32, shape []int) (Array, error) {
	if len(data) == 0 {
		return Array{}, errors.New("mlxgo: data must not be empty")
	}
	return newData(unsafe.Pointer(&data[0]), len(data), shape, Float32)
}

// NewFloat64 copies data into a new MLX float64 array with the provided shape.
func NewFloat64(data []float64, shape []int) (Array, error) {
	if len(data) == 0 {
		return Array{}, errors.New("mlxgo: data must not be empty")
	}
	return newData(unsafe.Pointer(&data[0]), len(data), shape, Float64)
}

// NewInt32 copies data into a new MLX int32 array with the provided shape.
func NewInt32(data []int32, shape []int) (Array, error) {
	if len(data) == 0 {
		return Array{}, errors.New("mlxgo: data must not be empty")
	}
	return newData(unsafe.Pointer(&data[0]), len(data), shape, Int32)
}

// NewInt64 copies data into a new MLX int64 array with the provided shape.
func NewInt64(data []int64, shape []int) (Array, error) {
	if len(data) == 0 {
		return Array{}, errors.New("mlxgo: data must not be empty")
	}
	return newData(unsafe.Pointer(&data[0]), len(data), shape, Int64)
}

// NewScalarFloat32 creates a scalar MLX float32 array.
func NewScalarFloat32(value float32) (Array, error) {
	return checkedArray(C.mlx_array_new_float32(C.float(value)), "mlx_array_new_float32")
}

// NewScalarFloat64 creates a scalar MLX float64 array.
func NewScalarFloat64(value float64) (Array, error) {
	return checkedArray(C.mlx_array_new_float64(C.double(value)), "mlx_array_new_float64")
}

// NewScalarInt creates a scalar MLX int array.
func NewScalarInt(value int) (Array, error) {
	return checkedArray(C.mlx_array_new_int(C.int(value)), "mlx_array_new_int")
}

// Zeros creates an array of zeros with shape and dtype.
func Zeros(shape []int, dtype DType) (Array, error) {
	return shapeOp(shape, dtype, "mlx_zeros", func(out *C.mlx_array, cshape *C.int, ndim C.size_t, cdtype C.mlx_dtype, stream C.mlx_stream) C.int {
		return C.mlx_zeros(out, cshape, ndim, cdtype, stream)
	})
}

// Ones creates an array of ones with shape and dtype.
func Ones(shape []int, dtype DType) (Array, error) {
	return shapeOp(shape, dtype, "mlx_ones", func(out *C.mlx_array, cshape *C.int, ndim C.size_t, cdtype C.mlx_dtype, stream C.mlx_stream) C.int {
		return C.mlx_ones(out, cshape, ndim, cdtype, stream)
	})
}

// ZerosLike creates an array of zeros with the same shape and dtype as a.
func ZerosLike(a Array) (Array, error) {
	return unaryOp(a, "mlx_zeros_like", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_zeros_like(out, input, stream)
	})
}

// OnesLike creates an array of ones with the same shape and dtype as a.
func OnesLike(a Array) (Array, error) {
	return unaryOp(a, "mlx_ones_like", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_ones_like(out, input, stream)
	})
}

// Full creates an array filled with value, cast to dtype by MLX.
func Full(shape []int, value float64, dtype DType) (Array, error) {
	cshape, err := cDataShape(shape)
	if err != nil {
		return Array{}, err
	}
	cdtype, err := cDType(dtype)
	if err != nil {
		return Array{}, err
	}
	fill, err := NewScalarFloat64(value)
	if err != nil {
		return Array{}, err
	}
	defer fill.Close()
	fillHandle, err := fill.handleValue()
	if err != nil {
		return Array{}, err
	}

	stream := C.mlx_default_cpu_stream_new()
	defer C.mlx_stream_free(stream)

	out := newArray(C.mlx_array_new())
	if code := C.mlx_full(out.outHandle(), (*C.int)(unsafe.Pointer(&cshape[0])), C.size_t(len(cshape)), fillHandle, cdtype, stream); code != 0 {
		_ = out.Close()
		return Array{}, fmt.Errorf("mlxgo: mlx_full failed with code %d", int(code))
	}
	return out, nil
}

// Add returns a new array containing a + b.
func Add(a, b Array) (Array, error) {
	return binaryOp(a, b, "mlx_add", func(out *C.mlx_array, left C.mlx_array, right C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_add(out, left, right, stream)
	})
}

// AddMM returns beta*c + alpha*(a @ b).
func AddMM(c, a, b Array, alpha, beta float32) (Array, error) {
	chandle, err := c.handleValue()
	if err != nil {
		return Array{}, err
	}
	ahandle, err := a.handleValue()
	if err != nil {
		return Array{}, err
	}
	bhandle, err := b.handleValue()
	if err != nil {
		return Array{}, err
	}

	stream := C.mlx_default_cpu_stream_new()
	defer C.mlx_stream_free(stream)

	out := newArray(C.mlx_array_new())
	if code := C.mlx_addmm(out.outHandle(), chandle, ahandle, bhandle, C.float(alpha), C.float(beta), stream); code != 0 {
		_ = out.Close()
		return Array{}, fmt.Errorf("mlxgo: mlx_addmm failed with code %d", int(code))
	}
	return out, nil
}

// Subtract returns a new array containing a - b.
func Subtract(a, b Array) (Array, error) {
	return binaryOp(a, b, "mlx_subtract", func(out *C.mlx_array, left C.mlx_array, right C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_subtract(out, left, right, stream)
	})
}

// Multiply returns a new array containing a * b.
func Multiply(a, b Array) (Array, error) {
	return binaryOp(a, b, "mlx_multiply", func(out *C.mlx_array, left C.mlx_array, right C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_multiply(out, left, right, stream)
	})
}

// Divide returns a new array containing a / b.
func Divide(a, b Array) (Array, error) {
	return binaryOp(a, b, "mlx_divide", func(out *C.mlx_array, left C.mlx_array, right C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_divide(out, left, right, stream)
	})
}

// Matmul returns the matrix product of a and b.
func Matmul(a, b Array) (Array, error) {
	return binaryOp(a, b, "mlx_matmul", func(out *C.mlx_array, left C.mlx_array, right C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_matmul(out, left, right, stream)
	})
}

// Maximum returns the elementwise maximum of a and b.
func Maximum(a, b Array) (Array, error) {
	return binaryOp(a, b, "mlx_maximum", func(out *C.mlx_array, left C.mlx_array, right C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_maximum(out, left, right, stream)
	})
}

// Minimum returns the elementwise minimum of a and b.
func Minimum(a, b Array) (Array, error) {
	return binaryOp(a, b, "mlx_minimum", func(out *C.mlx_array, left C.mlx_array, right C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_minimum(out, left, right, stream)
	})
}

// Power returns a ** b elementwise.
func Power(a, b Array) (Array, error) {
	return binaryOp(a, b, "mlx_power", func(out *C.mlx_array, left C.mlx_array, right C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_power(out, left, right, stream)
	})
}

// Clip clamps a between min and max.
func Clip(a, min, max Array) (Array, error) {
	ahandle, err := a.handleValue()
	if err != nil {
		return Array{}, err
	}
	minHandle, err := min.handleValue()
	if err != nil {
		return Array{}, err
	}
	maxHandle, err := max.handleValue()
	if err != nil {
		return Array{}, err
	}

	stream := C.mlx_default_cpu_stream_new()
	defer C.mlx_stream_free(stream)

	out := newArray(C.mlx_array_new())
	if code := C.mlx_clip(out.outHandle(), ahandle, minHandle, maxHandle, stream); code != 0 {
		_ = out.Close()
		return Array{}, fmt.Errorf("mlxgo: mlx_clip failed with code %d", int(code))
	}
	return out, nil
}

// Arange creates an MLX float32 range using the default CPU stream.
func Arange(start, stop, step float64) (Array, error) {
	return ArangeDType(start, stop, step, Float32)
}

// ArangeDType creates an MLX range with dtype using the default CPU stream.
func ArangeDType(start, stop, step float64, dtype DType) (Array, error) {
	if step == 0 {
		return Array{}, errors.New("mlxgo: arange step must not be zero")
	}
	cdtype, err := cDType(dtype)
	if err != nil {
		return Array{}, err
	}

	stream := C.mlx_default_cpu_stream_new()
	defer C.mlx_stream_free(stream)

	out := newArray(C.mlx_array_new())
	if code := C.mlx_arange(out.outHandle(), C.double(start), C.double(stop), C.double(step), cdtype, stream); code != 0 {
		_ = out.Close()
		return Array{}, fmt.Errorf("mlxgo: mlx_arange failed with code %d", int(code))
	}
	return out, nil
}

// AsType casts an array to dtype.
func AsType(a Array, dtype DType) (Array, error) {
	cdtype, err := cDType(dtype)
	if err != nil {
		return Array{}, err
	}
	return unaryOp(a, "mlx_astype", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_astype(out, input, cdtype, stream)
	})
}

// Reshape returns an array view with shape. A single -1 dimension is allowed.
func Reshape(a Array, shape []int) (Array, error) {
	cshape, err := cReshapeShape(shape)
	if err != nil {
		return Array{}, err
	}
	return unaryOp(a, "mlx_reshape", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_reshape(out, input, (*C.int)(unsafe.Pointer(&cshape[0])), C.size_t(len(cshape)), stream)
	})
}

// Transpose reverses the array axes.
func Transpose(a Array) (Array, error) {
	return unaryOp(a, "mlx_transpose", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_transpose(out, input, stream)
	})
}

// TransposeAxes permutes array axes using the provided axis order.
func TransposeAxes(a Array, axes []int) (Array, error) {
	if len(axes) == 0 {
		return Transpose(a)
	}
	caxes := cInts(axes)
	return unaryOp(a, "mlx_transpose_axes", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_transpose_axes(out, input, (*C.int)(unsafe.Pointer(&caxes[0])), C.size_t(len(caxes)), stream)
	})
}

// BroadcastTo broadcasts a to shape.
func BroadcastTo(a Array, shape []int) (Array, error) {
	cshape, err := cDataShape(shape)
	if err != nil {
		return Array{}, err
	}
	return unaryOp(a, "mlx_broadcast_to", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_broadcast_to(out, input, (*C.int)(unsafe.Pointer(&cshape[0])), C.size_t(len(cshape)), stream)
	})
}

// ExpandDims inserts a length-one dimension at axis.
func ExpandDims(a Array, axis int) (Array, error) {
	return unaryOp(a, "mlx_expand_dims", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_expand_dims(out, input, C.int(axis), stream)
	})
}

// ExpandDimsAxes inserts length-one dimensions at axes.
func ExpandDimsAxes(a Array, axes []int) (Array, error) {
	if len(axes) == 0 {
		return Array{}, errors.New("mlxgo: expand axes must not be empty")
	}
	caxes := cInts(axes)
	return unaryOp(a, "mlx_expand_dims_axes", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_expand_dims_axes(out, input, (*C.int)(unsafe.Pointer(&caxes[0])), C.size_t(len(caxes)), stream)
	})
}

// Squeeze removes all length-one dimensions.
func Squeeze(a Array) (Array, error) {
	return unaryOp(a, "mlx_squeeze", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_squeeze(out, input, stream)
	})
}

// SqueezeAxis removes one length-one dimension.
func SqueezeAxis(a Array, axis int) (Array, error) {
	return unaryOp(a, "mlx_squeeze_axis", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_squeeze_axis(out, input, C.int(axis), stream)
	})
}

// SqueezeAxes removes the provided length-one dimensions. Empty axes squeeze all.
func SqueezeAxes(a Array, axes []int) (Array, error) {
	if len(axes) == 0 {
		return Squeeze(a)
	}
	caxes := cInts(axes)
	return unaryOp(a, "mlx_squeeze_axes", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_squeeze_axes(out, input, (*C.int)(unsafe.Pointer(&caxes[0])), C.size_t(len(caxes)), stream)
	})
}

// Flatten flattens dimensions from startAxis through endAxis.
func Flatten(a Array, startAxis, endAxis int) (Array, error) {
	return unaryOp(a, "mlx_flatten", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_flatten(out, input, C.int(startAxis), C.int(endAxis), stream)
	})
}

// Abs returns the absolute value elementwise.
func Abs(a Array) (Array, error) {
	return unaryOp(a, "mlx_abs", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_abs(out, input, stream)
	})
}

// Exp returns e raised elementwise to a.
func Exp(a Array) (Array, error) {
	return unaryOp(a, "mlx_exp", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_exp(out, input, stream)
	})
}

// Log returns the natural logarithm elementwise.
func Log(a Array) (Array, error) {
	return unaryOp(a, "mlx_log", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_log(out, input, stream)
	})
}

// Negative returns -a.
func Negative(a Array) (Array, error) {
	return unaryOp(a, "mlx_negative", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_negative(out, input, stream)
	})
}

// Square returns a * a elementwise.
func Square(a Array) (Array, error) {
	return unaryOp(a, "mlx_square", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_square(out, input, stream)
	})
}

// Sqrt returns the square root elementwise.
func Sqrt(a Array) (Array, error) {
	return unaryOp(a, "mlx_sqrt", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_sqrt(out, input, stream)
	})
}

// Sigmoid returns 1 / (1 + exp(-a)) elementwise.
func Sigmoid(a Array) (Array, error) {
	return unaryOp(a, "mlx_sigmoid", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_sigmoid(out, input, stream)
	})
}

// Tanh returns hyperbolic tangent elementwise.
func Tanh(a Array) (Array, error) {
	return unaryOp(a, "mlx_tanh", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_tanh(out, input, stream)
	})
}

// Sin returns sine elementwise.
func Sin(a Array) (Array, error) {
	return unaryOp(a, "mlx_sin", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_sin(out, input, stream)
	})
}

// Cos returns cosine elementwise.
func Cos(a Array) (Array, error) {
	return unaryOp(a, "mlx_cos", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_cos(out, input, stream)
	})
}

// ReLU returns max(a, 0) elementwise.
func ReLU(a Array) (Array, error) {
	zeros, err := ZerosLike(a)
	if err != nil {
		return Array{}, err
	}
	defer zeros.Close()
	return Maximum(a, zeros)
}

// StopGradient returns a detached array for gradient transforms.
func StopGradient(a Array) (Array, error) {
	return unaryOp(a, "mlx_stop_gradient", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_stop_gradient(out, input, stream)
	})
}

// Sum reduces all elements of a.
func Sum(a Array, keepdims bool) (Array, error) {
	return unaryOp(a, "mlx_sum", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_sum(out, input, C.bool(keepdims), stream)
	})
}

// SumAxis reduces a along one axis.
func SumAxis(a Array, axis int, keepdims bool) (Array, error) {
	return unaryOp(a, "mlx_sum_axis", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_sum_axis(out, input, C.int(axis), C.bool(keepdims), stream)
	})
}

// SumAxes reduces a along the provided axes. Empty axes reduce all elements.
func SumAxes(a Array, axes []int, keepdims bool) (Array, error) {
	if len(axes) == 0 {
		return Sum(a, keepdims)
	}
	caxes := cInts(axes)
	return unaryOp(a, "mlx_sum_axes", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_sum_axes(out, input, (*C.int)(unsafe.Pointer(&caxes[0])), C.size_t(len(caxes)), C.bool(keepdims), stream)
	})
}

// Mean reduces all elements of a.
func Mean(a Array, keepdims bool) (Array, error) {
	return unaryOp(a, "mlx_mean", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_mean(out, input, C.bool(keepdims), stream)
	})
}

// MeanAxis reduces a along one axis.
func MeanAxis(a Array, axis int, keepdims bool) (Array, error) {
	return unaryOp(a, "mlx_mean_axis", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_mean_axis(out, input, C.int(axis), C.bool(keepdims), stream)
	})
}

// MeanAxes reduces a along the provided axes. Empty axes reduce all elements.
func MeanAxes(a Array, axes []int, keepdims bool) (Array, error) {
	if len(axes) == 0 {
		return Mean(a, keepdims)
	}
	caxes := cInts(axes)
	return unaryOp(a, "mlx_mean_axes", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_mean_axes(out, input, (*C.int)(unsafe.Pointer(&caxes[0])), C.size_t(len(caxes)), C.bool(keepdims), stream)
	})
}

// LogSumExp reduces all elements with log(sum(exp(a))).
func LogSumExp(a Array, keepdims bool) (Array, error) {
	return unaryOp(a, "mlx_logsumexp", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_logsumexp(out, input, C.bool(keepdims), stream)
	})
}

// LogSumExpAxis reduces a along one axis with log(sum(exp(a))).
func LogSumExpAxis(a Array, axis int, keepdims bool) (Array, error) {
	return unaryOp(a, "mlx_logsumexp_axis", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_logsumexp_axis(out, input, C.int(axis), C.bool(keepdims), stream)
	})
}

// LogSumExpAxes reduces a along the provided axes with log(sum(exp(a))).
func LogSumExpAxes(a Array, axes []int, keepdims bool) (Array, error) {
	if len(axes) == 0 {
		return LogSumExp(a, keepdims)
	}
	caxes := cInts(axes)
	return unaryOp(a, "mlx_logsumexp_axes", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_logsumexp_axes(out, input, (*C.int)(unsafe.Pointer(&caxes[0])), C.size_t(len(caxes)), C.bool(keepdims), stream)
	})
}

// Softmax computes softmax over all elements.
func Softmax(a Array, precise bool) (Array, error) {
	return unaryOp(a, "mlx_softmax", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_softmax(out, input, C.bool(precise), stream)
	})
}

// SoftmaxAxis computes softmax along one axis.
func SoftmaxAxis(a Array, axis int, precise bool) (Array, error) {
	return unaryOp(a, "mlx_softmax_axis", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_softmax_axis(out, input, C.int(axis), C.bool(precise), stream)
	})
}

// SoftmaxAxes computes softmax along the provided axes. Empty axes use all elements.
func SoftmaxAxes(a Array, axes []int, precise bool) (Array, error) {
	if len(axes) == 0 {
		return Softmax(a, precise)
	}
	caxes := cInts(axes)
	return unaryOp(a, "mlx_softmax_axes", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_softmax_axes(out, input, (*C.int)(unsafe.Pointer(&caxes[0])), C.size_t(len(caxes)), C.bool(precise), stream)
	})
}

// Argmax returns indices of maximum values over all elements.
func Argmax(a Array, keepdims bool) (Array, error) {
	return unaryOp(a, "mlx_argmax", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_argmax(out, input, C.bool(keepdims), stream)
	})
}

// ArgmaxAxis returns indices of maximum values along one axis.
func ArgmaxAxis(a Array, axis int, keepdims bool) (Array, error) {
	return unaryOp(a, "mlx_argmax_axis", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_argmax_axis(out, input, C.int(axis), C.bool(keepdims), stream)
	})
}

// Argmin returns indices of minimum values over all elements.
func Argmin(a Array, keepdims bool) (Array, error) {
	return unaryOp(a, "mlx_argmin", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_argmin(out, input, C.bool(keepdims), stream)
	})
}

// ArgminAxis returns indices of minimum values along one axis.
func ArgminAxis(a Array, axis int, keepdims bool) (Array, error) {
	return unaryOp(a, "mlx_argmin_axis", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_argmin_axis(out, input, C.int(axis), C.bool(keepdims), stream)
	})
}

// Equal returns a == b elementwise.
func Equal(a, b Array) (Array, error) {
	return binaryOp(a, b, "mlx_equal", func(out *C.mlx_array, left C.mlx_array, right C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_equal(out, left, right, stream)
	})
}

// Greater returns a > b elementwise.
func Greater(a, b Array) (Array, error) {
	return binaryOp(a, b, "mlx_greater", func(out *C.mlx_array, left C.mlx_array, right C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_greater(out, left, right, stream)
	})
}

// GreaterEqual returns a >= b elementwise.
func GreaterEqual(a, b Array) (Array, error) {
	return binaryOp(a, b, "mlx_greater_equal", func(out *C.mlx_array, left C.mlx_array, right C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_greater_equal(out, left, right, stream)
	})
}

// Less returns a < b elementwise.
func Less(a, b Array) (Array, error) {
	return binaryOp(a, b, "mlx_less", func(out *C.mlx_array, left C.mlx_array, right C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_less(out, left, right, stream)
	})
}

// LessEqual returns a <= b elementwise.
func LessEqual(a, b Array) (Array, error) {
	return binaryOp(a, b, "mlx_less_equal", func(out *C.mlx_array, left C.mlx_array, right C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_less_equal(out, left, right, stream)
	})
}

// Where returns x where condition is true and y otherwise.
func Where(condition, x, y Array) (Array, error) {
	chandle, err := condition.handleValue()
	if err != nil {
		return Array{}, err
	}
	xhandle, err := x.handleValue()
	if err != nil {
		return Array{}, err
	}
	yhandle, err := y.handleValue()
	if err != nil {
		return Array{}, err
	}

	stream := C.mlx_default_cpu_stream_new()
	defer C.mlx_stream_free(stream)

	out := newArray(C.mlx_array_new())
	if code := C.mlx_where(out.outHandle(), chandle, xhandle, yhandle, stream); code != 0 {
		_ = out.Close()
		return Array{}, fmt.Errorf("mlxgo: mlx_where failed with code %d", int(code))
	}
	return out, nil
}

// Take gathers values from a by flat indices.
func Take(a, indices Array) (Array, error) {
	return binaryOp(a, indices, "mlx_take", func(out *C.mlx_array, input C.mlx_array, idx C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_take(out, input, idx, stream)
	})
}

// TakeAxis gathers values from a by indices along axis.
func TakeAxis(a, indices Array, axis int) (Array, error) {
	return binaryOp(a, indices, "mlx_take_axis", func(out *C.mlx_array, input C.mlx_array, idx C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_take_axis(out, input, idx, C.int(axis), stream)
	})
}

// TakeAlongAxis gathers values using indices aligned with a along axis.
func TakeAlongAxis(a, indices Array, axis int) (Array, error) {
	return binaryOp(a, indices, "mlx_take_along_axis", func(out *C.mlx_array, input C.mlx_array, idx C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_take_along_axis(out, input, idx, C.int(axis), stream)
	})
}

// Gather is a convenience alias for TakeAxis.
func Gather(a, indices Array, axis int) (Array, error) {
	return TakeAxis(a, indices, axis)
}

// GatherSlices gathers slices from a along axis.
func GatherSlices(a, indices Array, axis int, sliceSizes []int) (Array, error) {
	if len(sliceSizes) == 0 {
		return Array{}, errors.New("mlxgo: sliceSizes must not be empty")
	}
	csizes := cInts(sliceSizes)
	return binaryOp(a, indices, "mlx_gather_single", func(out *C.mlx_array, input C.mlx_array, idx C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_gather_single(out, input, idx, C.int(axis), (*C.int)(unsafe.Pointer(&csizes[0])), C.size_t(len(csizes)), stream)
	})
}

// Concatenate joins arrays along the first axis.
func Concatenate(arrays []Array) (Array, error) {
	return vectorOp(arrays, "mlx_concatenate", func(out *C.mlx_array, vec C.mlx_vector_array, stream C.mlx_stream) C.int {
		return C.mlx_concatenate(out, vec, stream)
	})
}

// ConcatenateAxis joins arrays along axis.
func ConcatenateAxis(arrays []Array, axis int) (Array, error) {
	return vectorOp(arrays, "mlx_concatenate_axis", func(out *C.mlx_array, vec C.mlx_vector_array, stream C.mlx_stream) C.int {
		return C.mlx_concatenate_axis(out, vec, C.int(axis), stream)
	})
}

// Stack joins arrays along a new first axis.
func Stack(arrays []Array) (Array, error) {
	return vectorOp(arrays, "mlx_stack", func(out *C.mlx_array, vec C.mlx_vector_array, stream C.mlx_stream) C.int {
		return C.mlx_stack(out, vec, stream)
	})
}

// StackAxis joins arrays along a new axis.
func StackAxis(arrays []Array, axis int) (Array, error) {
	return vectorOp(arrays, "mlx_stack_axis", func(out *C.mlx_array, vec C.mlx_vector_array, stream C.mlx_stream) C.int {
		return C.mlx_stack_axis(out, vec, C.int(axis), stream)
	})
}

// Load reads a single array file supported by MLX, such as .npy.
func Load(file string) (Array, error) {
	if file == "" {
		return Array{}, errors.New("mlxgo: load file must not be empty")
	}
	cfile := C.CString(file)
	defer C.free(unsafe.Pointer(cfile))

	stream := C.mlx_default_cpu_stream_new()
	defer C.mlx_stream_free(stream)

	out := newArray(C.mlx_array_new())
	if code := C.mlx_load(out.outHandle(), cfile, stream); code != 0 {
		_ = out.Close()
		return Array{}, fmt.Errorf("mlxgo: mlx_load failed with code %d", int(code))
	}
	return out, nil
}

// Save writes a single array file supported by MLX, such as .npy.
func Save(file string, a Array) error {
	if file == "" {
		return errors.New("mlxgo: save file must not be empty")
	}
	handle, err := a.handleValue()
	if err != nil {
		return err
	}
	cfile := C.CString(file)
	defer C.free(unsafe.Pointer(cfile))

	if code := C.mlx_save(cfile, handle); code != 0 {
		return fmt.Errorf("mlxgo: mlx_save failed with code %d", int(code))
	}
	return nil
}

// LoadSafetensors opens a safetensors file and returns a handle for looking up
// arrays by name.
func LoadSafetensors(file string) (*SafeTensors, error) {
	if file == "" {
		return nil, errors.New("mlxgo: safetensors file must not be empty")
	}
	cfile := C.CString(file)
	defer C.free(unsafe.Pointer(cfile))

	stream := C.mlx_default_cpu_stream_new()
	defer C.mlx_stream_free(stream)

	arrays := C.mlx_map_string_to_array_new()
	metadata := C.mlx_map_string_to_string_new()
	if arrays.ctx == nil || metadata.ctx == nil {
		if arrays.ctx != nil {
			C.mlx_map_string_to_array_free(arrays)
		}
		if metadata.ctx != nil {
			C.mlx_map_string_to_string_free(metadata)
		}
		return nil, errors.New("mlxgo: failed to allocate safetensors maps")
	}

	if code := C.mlx_load_safetensors(&arrays, &metadata, cfile, stream); code != 0 {
		C.mlx_map_string_to_array_free(arrays)
		C.mlx_map_string_to_string_free(metadata)
		return nil, fmt.Errorf("mlxgo: mlx_load_safetensors failed with code %d", int(code))
	}
	return &SafeTensors{arrays: arrays, metadata: metadata}, nil
}

// SaveSafetensors writes named arrays and optional metadata to a safetensors file.
func SaveSafetensors(file string, arrays map[string]Array, metadata map[string]string) error {
	if file == "" {
		return errors.New("mlxgo: safetensors file must not be empty")
	}
	if len(arrays) == 0 {
		return errors.New("mlxgo: safetensors arrays must not be empty")
	}
	cfile := C.CString(file)
	defer C.free(unsafe.Pointer(cfile))

	carrays := C.mlx_map_string_to_array_new()
	cmetadata := C.mlx_map_string_to_string_new()
	if carrays.ctx == nil || cmetadata.ctx == nil {
		if carrays.ctx != nil {
			C.mlx_map_string_to_array_free(carrays)
		}
		if cmetadata.ctx != nil {
			C.mlx_map_string_to_string_free(cmetadata)
		}
		return errors.New("mlxgo: failed to allocate safetensors maps")
	}
	defer C.mlx_map_string_to_array_free(carrays)
	defer C.mlx_map_string_to_string_free(cmetadata)

	for key, arr := range arrays {
		handle, err := arr.handleValue()
		if err != nil {
			return fmt.Errorf("mlxgo: safetensors array %q: %w", key, err)
		}
		ckey := C.CString(key)
		code := C.mlx_map_string_to_array_insert(carrays, ckey, handle)
		C.free(unsafe.Pointer(ckey))
		if code != 0 {
			return fmt.Errorf("mlxgo: safetensors insert array %q failed with code %d", key, int(code))
		}
	}

	for key, value := range metadata {
		ckey := C.CString(key)
		cvalue := C.CString(value)
		code := C.mlx_map_string_to_string_insert(cmetadata, ckey, cvalue)
		C.free(unsafe.Pointer(ckey))
		C.free(unsafe.Pointer(cvalue))
		if code != 0 {
			return fmt.Errorf("mlxgo: safetensors insert metadata %q failed with code %d", key, int(code))
		}
	}

	if code := C.mlx_save_safetensors(cfile, carrays, cmetadata); code != 0 {
		return fmt.Errorf("mlxgo: mlx_save_safetensors failed with code %d", int(code))
	}
	return nil
}

// RandomSeed sets MLX's global random seed.
func RandomSeed(seed uint64) error {
	if code := C.mlx_random_seed(C.uint64_t(seed)); code != 0 {
		return fmt.Errorf("mlxgo: mlx_random_seed failed with code %d", int(code))
	}
	return nil
}

// RandomKey creates a random key array from seed.
func RandomKey(seed uint64) (Array, error) {
	out := newArray(C.mlx_array_new())
	if code := C.mlx_random_key(out.outHandle(), C.uint64_t(seed)); code != 0 {
		_ = out.Close()
		return Array{}, fmt.Errorf("mlxgo: mlx_random_key failed with code %d", int(code))
	}
	return out, nil
}

// RandomNormal samples a normal distribution with mean loc and standard deviation scale.
func RandomNormal(shape []int, dtype DType, loc, scale float32) (Array, error) {
	cshape, cdtype, err := randomShapeAndDType(shape, dtype)
	if err != nil {
		return Array{}, err
	}
	stream := C.mlx_default_cpu_stream_new()
	defer C.mlx_stream_free(stream)

	out := newArray(C.mlx_array_new())
	if code := C.mlx_random_normal(out.outHandle(), (*C.int)(unsafe.Pointer(&cshape[0])), C.size_t(len(cshape)), cdtype, C.float(loc), C.float(scale), C.mlx_array{}, stream); code != 0 {
		_ = out.Close()
		return Array{}, fmt.Errorf("mlxgo: mlx_random_normal failed with code %d", int(code))
	}
	return out, nil
}

// RandomUniform samples a uniform distribution over [low, high).
func RandomUniform(shape []int, dtype DType, low, high float32) (Array, error) {
	cshape, cdtype, err := randomShapeAndDType(shape, dtype)
	if err != nil {
		return Array{}, err
	}
	lowArr, err := NewScalarFloat32(low)
	if err != nil {
		return Array{}, err
	}
	defer lowArr.Close()
	highArr, err := NewScalarFloat32(high)
	if err != nil {
		return Array{}, err
	}
	defer highArr.Close()
	lowHandle, err := lowArr.handleValue()
	if err != nil {
		return Array{}, err
	}
	highHandle, err := highArr.handleValue()
	if err != nil {
		return Array{}, err
	}

	stream := C.mlx_default_cpu_stream_new()
	defer C.mlx_stream_free(stream)

	out := newArray(C.mlx_array_new())
	if code := C.mlx_random_uniform(out.outHandle(), lowHandle, highHandle, (*C.int)(unsafe.Pointer(&cshape[0])), C.size_t(len(cshape)), cdtype, C.mlx_array{}, stream); code != 0 {
		_ = out.Close()
		return Array{}, fmt.Errorf("mlxgo: mlx_random_uniform failed with code %d", int(code))
	}
	return out, nil
}

// RandomRandint samples integer values over [low, high).
func RandomRandint(shape []int, dtype DType, low, high int) (Array, error) {
	cshape, cdtype, err := randomShapeAndDType(shape, dtype)
	if err != nil {
		return Array{}, err
	}
	lowArr, err := NewScalarInt(low)
	if err != nil {
		return Array{}, err
	}
	defer lowArr.Close()
	highArr, err := NewScalarInt(high)
	if err != nil {
		return Array{}, err
	}
	defer highArr.Close()
	lowHandle, err := lowArr.handleValue()
	if err != nil {
		return Array{}, err
	}
	highHandle, err := highArr.handleValue()
	if err != nil {
		return Array{}, err
	}

	stream := C.mlx_default_cpu_stream_new()
	defer C.mlx_stream_free(stream)

	out := newArray(C.mlx_array_new())
	if code := C.mlx_random_randint(out.outHandle(), lowHandle, highHandle, (*C.int)(unsafe.Pointer(&cshape[0])), C.size_t(len(cshape)), cdtype, C.mlx_array{}, stream); code != 0 {
		_ = out.Close()
		return Array{}, fmt.Errorf("mlxgo: mlx_random_randint failed with code %d", int(code))
	}
	return out, nil
}

// RandomBernoulli samples bool values with probability p.
func RandomBernoulli(shape []int, p float32) (Array, error) {
	cshape, err := cDataShape(shape)
	if err != nil {
		return Array{}, err
	}
	pArr, err := NewScalarFloat32(p)
	if err != nil {
		return Array{}, err
	}
	defer pArr.Close()
	pHandle, err := pArr.handleValue()
	if err != nil {
		return Array{}, err
	}

	stream := C.mlx_default_cpu_stream_new()
	defer C.mlx_stream_free(stream)

	out := newArray(C.mlx_array_new())
	if code := C.mlx_random_bernoulli(out.outHandle(), pHandle, (*C.int)(unsafe.Pointer(&cshape[0])), C.size_t(len(cshape)), C.mlx_array{}, stream); code != 0 {
		_ = out.Close()
		return Array{}, fmt.Errorf("mlxgo: mlx_random_bernoulli failed with code %d", int(code))
	}
	return out, nil
}

// RandomCategorical samples class indices from logits along axis.
func RandomCategorical(logits Array, axis int) (Array, error) {
	return unaryOp(logits, "mlx_random_categorical", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_random_categorical(out, input, C.int(axis), C.mlx_array{}, stream)
	})
}

// Close releases safetensors maps.
func (s *SafeTensors) Close() error {
	if s == nil || s.closed {
		return nil
	}
	var first error
	if s.arrays.ctx != nil {
		if code := C.mlx_map_string_to_array_free(s.arrays); code != 0 {
			first = fmt.Errorf("mlxgo: mlx_map_string_to_array_free failed with code %d", int(code))
		}
		s.arrays.ctx = nil
	}
	if s.metadata.ctx != nil {
		if code := C.mlx_map_string_to_string_free(s.metadata); code != 0 && first == nil {
			first = fmt.Errorf("mlxgo: mlx_map_string_to_string_free failed with code %d", int(code))
		}
		s.metadata.ctx = nil
	}
	s.closed = true
	return first
}

// Get returns a named array from a safetensors file.
func (s *SafeTensors) Get(name string) (Array, error) {
	if s == nil || s.closed || s.arrays.ctx == nil {
		return Array{}, errors.New("mlxgo: safetensors handle is closed")
	}
	if name == "" {
		return Array{}, errors.New("mlxgo: safetensors key must not be empty")
	}
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))

	var value C.mlx_array
	if code := C.mlx_map_string_to_array_get(&value, s.arrays, cname); code != 0 {
		return Array{}, fmt.Errorf("mlxgo: safetensors array %q not found", name)
	}
	return checkedArray(value, "mlx_map_string_to_array_get")
}

// Metadata returns a named metadata value from a safetensors file.
func (s *SafeTensors) Metadata(name string) (string, bool, error) {
	if s == nil || s.closed || s.metadata.ctx == nil {
		return "", false, errors.New("mlxgo: safetensors handle is closed")
	}
	if name == "" {
		return "", false, errors.New("mlxgo: safetensors key must not be empty")
	}
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))

	var value *C.char
	if code := C.mlx_map_string_to_string_get((**C.char)(unsafe.Pointer(&value)), s.metadata, cname); code != 0 || value == nil {
		return "", false, nil
	}
	return C.GoString(value), true, nil
}

// SetDefaultCPU switches MLX's default device to CPU index 0.
func SetDefaultCPU() error {
	dev := C.mlx_device_new_type(C.MLX_CPU, 0)
	if dev.ctx == nil {
		return errors.New("mlxgo: failed to create CPU device")
	}
	defer C.mlx_device_free(dev)

	if code := C.mlx_set_default_device(dev); code != 0 {
		return fmt.Errorf("mlxgo: mlx_set_default_device failed with code %d", int(code))
	}
	return nil
}

// Close releases the underlying MLX array.
func (a *Array) Close() error {
	if a == nil {
		return nil
	}
	if a.state == nil || a.state.closed || a.state.handle.ctx == nil {
		return nil
	}
	if !a.state.owned {
		a.state.handle.ctx = nil
		a.state.closed = true
		return nil
	}
	if code := C.mlx_array_free(a.state.handle); code != 0 {
		return fmt.Errorf("mlxgo: mlx_array_free failed with code %d", int(code))
	}
	a.state.handle.ctx = nil
	a.state.closed = true
	return nil
}

// Eval materializes the array.
func (a Array) Eval() error {
	handle, err := a.handleValue()
	if err != nil {
		return err
	}
	if code := C.mlx_array_eval(handle); code != 0 {
		return fmt.Errorf("mlxgo: mlx_array_eval failed with code %d", int(code))
	}
	return nil
}

// Shape returns a copy of the array shape.
func (a Array) Shape() []int {
	handle, err := a.handleValue()
	if err != nil {
		return nil
	}
	ndim := int(C.mlx_array_ndim(handle))
	ptr := C.mlx_array_shape(handle)
	dims := unsafe.Slice((*C.int)(unsafe.Pointer(ptr)), ndim)

	shape := make([]int, ndim)
	for i, dim := range dims {
		shape[i] = int(dim)
	}
	return shape
}

// DType returns the array element type.
func (a Array) DType() (DType, error) {
	handle, err := a.handleValue()
	if err != nil {
		return Float32, err
	}
	return goDType(C.mlx_array_dtype(handle)), nil
}

// Size returns the number of elements in the array.
func (a Array) Size() int {
	handle, err := a.handleValue()
	if err != nil {
		return 0
	}
	return int(C.mlx_array_size(handle))
}

// Float32Data evaluates the array and copies its float32 contents into Go.
func (a Array) Float32Data() ([]float32, error) {
	if err := a.Eval(); err != nil {
		return nil, err
	}
	if err := a.expectDType(Float32); err != nil {
		return nil, err
	}

	n := a.Size()
	handle, err := a.handleValue()
	if err != nil {
		return nil, err
	}
	ptr := C.mlx_array_data_float32(handle)
	if ptr == nil {
		return nil, errors.New("mlxgo: MLX did not return materialized float32 data")
	}

	cdata := unsafe.Slice((*C.float)(unsafe.Pointer(ptr)), n)
	data := make([]float32, n)
	for i, v := range cdata {
		data[i] = float32(v)
	}
	return data, nil
}

// Float64Data evaluates the array and copies its float64 contents into Go.
func (a Array) Float64Data() ([]float64, error) {
	if err := a.Eval(); err != nil {
		return nil, err
	}
	if err := a.expectDType(Float64); err != nil {
		return nil, err
	}

	n := a.Size()
	handle, err := a.handleValue()
	if err != nil {
		return nil, err
	}
	ptr := C.mlx_array_data_float64(handle)
	if ptr == nil {
		return nil, errors.New("mlxgo: MLX did not return materialized float64 data")
	}

	cdata := unsafe.Slice((*C.double)(unsafe.Pointer(ptr)), n)
	data := make([]float64, n)
	for i, v := range cdata {
		data[i] = float64(v)
	}
	return data, nil
}

// Int32Data evaluates the array and copies its int32 contents into Go.
func (a Array) Int32Data() ([]int32, error) {
	if err := a.Eval(); err != nil {
		return nil, err
	}
	if err := a.expectDType(Int32); err != nil {
		return nil, err
	}

	n := a.Size()
	handle, err := a.handleValue()
	if err != nil {
		return nil, err
	}
	ptr := C.mlx_array_data_int32(handle)
	if ptr == nil {
		return nil, errors.New("mlxgo: MLX did not return materialized int32 data")
	}

	cdata := unsafe.Slice((*C.int32_t)(unsafe.Pointer(ptr)), n)
	data := make([]int32, n)
	for i, v := range cdata {
		data[i] = int32(v)
	}
	return data, nil
}

// Int64Data evaluates the array and copies its int64 contents into Go.
func (a Array) Int64Data() ([]int64, error) {
	if err := a.Eval(); err != nil {
		return nil, err
	}
	if err := a.expectDType(Int64); err != nil {
		return nil, err
	}

	n := a.Size()
	handle, err := a.handleValue()
	if err != nil {
		return nil, err
	}
	ptr := C.mlx_array_data_int64(handle)
	if ptr == nil {
		return nil, errors.New("mlxgo: MLX did not return materialized int64 data")
	}

	cdata := unsafe.Slice((*C.int64_t)(unsafe.Pointer(ptr)), n)
	data := make([]int64, n)
	for i, v := range cdata {
		data[i] = int64(v)
	}
	return data, nil
}

// UInt32Data evaluates the array and copies its uint32 contents into Go.
func (a Array) UInt32Data() ([]uint32, error) {
	if err := a.Eval(); err != nil {
		return nil, err
	}
	if err := a.expectDType(UInt32); err != nil {
		return nil, err
	}

	n := a.Size()
	handle, err := a.handleValue()
	if err != nil {
		return nil, err
	}
	ptr := C.mlx_array_data_uint32(handle)
	if ptr == nil {
		return nil, errors.New("mlxgo: MLX did not return materialized uint32 data")
	}

	cdata := unsafe.Slice((*C.uint32_t)(unsafe.Pointer(ptr)), n)
	data := make([]uint32, n)
	for i, v := range cdata {
		data[i] = uint32(v)
	}
	return data, nil
}

// UInt64Data evaluates the array and copies its uint64 contents into Go.
func (a Array) UInt64Data() ([]uint64, error) {
	if err := a.Eval(); err != nil {
		return nil, err
	}
	if err := a.expectDType(UInt64); err != nil {
		return nil, err
	}

	n := a.Size()
	handle, err := a.handleValue()
	if err != nil {
		return nil, err
	}
	ptr := C.mlx_array_data_uint64(handle)
	if ptr == nil {
		return nil, errors.New("mlxgo: MLX did not return materialized uint64 data")
	}

	cdata := unsafe.Slice((*C.uint64_t)(unsafe.Pointer(ptr)), n)
	data := make([]uint64, n)
	for i, v := range cdata {
		data[i] = uint64(v)
	}
	return data, nil
}

// BoolData evaluates the array and copies its bool contents into Go.
func (a Array) BoolData() ([]bool, error) {
	if err := a.Eval(); err != nil {
		return nil, err
	}
	if err := a.expectDType(Bool); err != nil {
		return nil, err
	}

	n := a.Size()
	handle, err := a.handleValue()
	if err != nil {
		return nil, err
	}
	ptr := C.mlx_array_data_bool(handle)
	if ptr == nil {
		return nil, errors.New("mlxgo: MLX did not return materialized bool data")
	}

	cdata := unsafe.Slice((*C.bool)(unsafe.Pointer(ptr)), n)
	data := make([]bool, n)
	for i, v := range cdata {
		data[i] = bool(v)
	}
	return data, nil
}

// String returns MLX's textual representation of the array.
func (a Array) String() string {
	handle, err := a.handleValue()
	if err != nil {
		return "<closed mlx array>"
	}

	str := C.mlx_string_new()
	defer C.mlx_string_free(str)

	if code := C.mlx_array_tostring(&str, handle); code != 0 {
		return fmt.Sprintf("<mlx_array_tostring failed: %d>", int(code))
	}

	return C.GoString(C.mlx_string_data(str))
}

func shapeOp(shape []int, dtype DType, name string, op func(*C.mlx_array, *C.int, C.size_t, C.mlx_dtype, C.mlx_stream) C.int) (Array, error) {
	cshape, err := cDataShape(shape)
	if err != nil {
		return Array{}, err
	}
	cdtype, err := cDType(dtype)
	if err != nil {
		return Array{}, err
	}

	stream := C.mlx_default_cpu_stream_new()
	defer C.mlx_stream_free(stream)

	out := newArray(C.mlx_array_new())
	if code := op(out.outHandle(), (*C.int)(unsafe.Pointer(&cshape[0])), C.size_t(len(cshape)), cdtype, stream); code != 0 {
		_ = out.Close()
		return Array{}, fmt.Errorf("mlxgo: %s failed with code %d", name, int(code))
	}
	return out, nil
}

func randomShapeAndDType(shape []int, dtype DType) ([]C.int, C.mlx_dtype, error) {
	cshape, err := cDataShape(shape)
	if err != nil {
		return nil, C.MLX_FLOAT32, err
	}
	cdtype, err := cDType(dtype)
	if err != nil {
		return nil, C.MLX_FLOAT32, err
	}
	return cshape, cdtype, nil
}

func newData(data unsafe.Pointer, dataLen int, shape []int, dtype DType) (Array, error) {
	cshape, err := cDataShape(shape)
	if err != nil {
		return Array{}, err
	}
	if elementCount(shape) != dataLen {
		return Array{}, fmt.Errorf("mlxgo: shape has %d elements, data has %d", elementCount(shape), dataLen)
	}

	cdtype, err := cDType(dtype)
	if err != nil {
		return Array{}, err
	}

	return checkedArray(
		C.mlx_array_new_data(data, (*C.int)(unsafe.Pointer(&cshape[0])), C.int(len(cshape)), cdtype),
		"mlx_array_new_data",
	)
}

func checkedArray(handle C.mlx_array, name string) (Array, error) {
	if handle.ctx == nil {
		return Array{}, fmt.Errorf("mlxgo: %s returned an empty array", name)
	}
	return newArray(handle), nil
}

func unaryOp(a Array, name string, op func(*C.mlx_array, C.mlx_array, C.mlx_stream) C.int) (Array, error) {
	stream := C.mlx_default_cpu_stream_new()
	defer C.mlx_stream_free(stream)

	out := newArray(C.mlx_array_new())
	input, err := a.handleValue()
	if err != nil {
		return Array{}, err
	}
	if code := op(out.outHandle(), input, stream); code != 0 {
		_ = out.Close()
		return Array{}, fmt.Errorf("mlxgo: %s failed with code %d", name, int(code))
	}
	return out, nil
}

func binaryOp(a, b Array, name string, op func(*C.mlx_array, C.mlx_array, C.mlx_array, C.mlx_stream) C.int) (Array, error) {
	left, err := a.handleValue()
	if err != nil {
		return Array{}, err
	}
	right, err := b.handleValue()
	if err != nil {
		return Array{}, err
	}

	stream := C.mlx_default_cpu_stream_new()
	defer C.mlx_stream_free(stream)

	out := newArray(C.mlx_array_new())
	if code := op(out.outHandle(), left, right, stream); code != 0 {
		_ = out.Close()
		return Array{}, fmt.Errorf("mlxgo: %s failed with code %d", name, int(code))
	}
	return out, nil
}

func vectorOp(arrays []Array, name string, op func(*C.mlx_array, C.mlx_vector_array, C.mlx_stream) C.int) (Array, error) {
	vec, err := cArrayVector(arrays)
	if err != nil {
		return Array{}, err
	}
	defer C.mlx_vector_array_free(vec)

	stream := C.mlx_default_cpu_stream_new()
	defer C.mlx_stream_free(stream)

	out := newArray(C.mlx_array_new())
	if code := op(out.outHandle(), vec, stream); code != 0 {
		_ = out.Close()
		return Array{}, fmt.Errorf("mlxgo: %s failed with code %d", name, int(code))
	}
	return out, nil
}

func cArrayVector(arrays []Array) (C.mlx_vector_array, error) {
	handles, err := cArrayHandles(arrays)
	if err != nil {
		return C.mlx_vector_array{}, err
	}

	vec := C.mlx_vector_array_new_data((*C.mlx_array)(unsafe.Pointer(&handles[0])), C.size_t(len(handles)))
	if vec.ctx == nil {
		return C.mlx_vector_array{}, errors.New("mlxgo: failed to allocate MLX array vector")
	}
	return vec, nil
}

func setArrayVector(vec *C.mlx_vector_array, arrays []Array) error {
	if vec == nil || (*vec).ctx == nil {
		return errors.New("mlxgo: vector is empty")
	}
	handles, err := cArrayHandles(arrays)
	if err != nil {
		return err
	}
	if code := C.mlx_vector_array_set_data(vec, (*C.mlx_array)(unsafe.Pointer(&handles[0])), C.size_t(len(handles))); code != 0 {
		return fmt.Errorf("mlxgo: mlx_vector_array_set_data failed with code %d", int(code))
	}
	return nil
}

func cArrayHandles(arrays []Array) ([]C.mlx_array, error) {
	if len(arrays) == 0 {
		return nil, errors.New("mlxgo: array list must not be empty")
	}

	handles := make([]C.mlx_array, len(arrays))
	for i, arr := range arrays {
		handle, err := arr.handleValue()
		if err != nil {
			return nil, fmt.Errorf("mlxgo: array %d: %w", i, err)
		}
		handles[i] = handle
	}
	return handles, nil
}

func (a Array) handleValue() (C.mlx_array, error) {
	if a.state == nil || a.state.closed || a.state.handle.ctx == nil {
		return C.mlx_array{}, errors.New("mlxgo: array is closed")
	}
	return a.state.handle, nil
}

func (a Array) expectDType(expected DType) error {
	actual, err := a.DType()
	if err != nil {
		return err
	}
	if actual != expected {
		return fmt.Errorf("mlxgo: array dtype is %s, expected %s", actual, expected)
	}
	return nil
}

func newArray(handle C.mlx_array) Array {
	return Array{state: &arrayState{handle: handle, owned: true}}
}

func borrowedArray(handle C.mlx_array) Array {
	return Array{state: &arrayState{handle: handle}}
}

func (a Array) outHandle() *C.mlx_array {
	return &a.state.handle
}

func cDataShape(shape []int) ([]C.int, error) {
	if len(shape) == 0 {
		return nil, errors.New("mlxgo: shape must not be empty")
	}

	cshape := make([]C.int, len(shape))
	for i, dim := range shape {
		if dim <= 0 {
			return nil, fmt.Errorf("mlxgo: shape dimension %d must be positive", i)
		}
		cshape[i] = C.int(dim)
	}
	return cshape, nil
}

func cReshapeShape(shape []int) ([]C.int, error) {
	if len(shape) == 0 {
		return nil, errors.New("mlxgo: reshape shape must not be empty")
	}

	inferred := 0
	cshape := make([]C.int, len(shape))
	for i, dim := range shape {
		switch {
		case dim == -1:
			inferred++
			if inferred > 1 {
				return nil, errors.New("mlxgo: reshape shape can contain at most one inferred dimension")
			}
		case dim <= 0:
			return nil, fmt.Errorf("mlxgo: reshape dimension %d must be positive or -1", i)
		}
		cshape[i] = C.int(dim)
	}
	return cshape, nil
}

func cInts(values []int) []C.int {
	out := make([]C.int, len(values))
	for i, value := range values {
		out[i] = C.int(value)
	}
	return out
}

func elementCount(shape []int) int {
	total := 1
	for _, dim := range shape {
		total *= dim
	}
	return total
}

func cDType(dtype DType) (C.mlx_dtype, error) {
	switch dtype {
	case Bool:
		return C.MLX_BOOL, nil
	case UInt8:
		return C.MLX_UINT8, nil
	case UInt16:
		return C.MLX_UINT16, nil
	case UInt32:
		return C.MLX_UINT32, nil
	case UInt64:
		return C.MLX_UINT64, nil
	case Int8:
		return C.MLX_INT8, nil
	case Int16:
		return C.MLX_INT16, nil
	case Int32:
		return C.MLX_INT32, nil
	case Int64:
		return C.MLX_INT64, nil
	case Float16:
		return C.MLX_FLOAT16, nil
	case Float32:
		return C.MLX_FLOAT32, nil
	case Float64:
		return C.MLX_FLOAT64, nil
	case BFloat16:
		return C.MLX_BFLOAT16, nil
	case Complex64:
		return C.MLX_COMPLEX64, nil
	default:
		return C.MLX_FLOAT32, fmt.Errorf("mlxgo: unsupported dtype %d", dtype)
	}
}

func goDType(dtype C.mlx_dtype) DType {
	switch dtype {
	case C.MLX_BOOL:
		return Bool
	case C.MLX_UINT8:
		return UInt8
	case C.MLX_UINT16:
		return UInt16
	case C.MLX_UINT32:
		return UInt32
	case C.MLX_UINT64:
		return UInt64
	case C.MLX_INT8:
		return Int8
	case C.MLX_INT16:
		return Int16
	case C.MLX_INT32:
		return Int32
	case C.MLX_INT64:
		return Int64
	case C.MLX_FLOAT16:
		return Float16
	case C.MLX_FLOAT32:
		return Float32
	case C.MLX_FLOAT64:
		return Float64
	case C.MLX_BFLOAT16:
		return BFloat16
	case C.MLX_COMPLEX64:
		return Complex64
	default:
		return DType(dtype)
	}
}
