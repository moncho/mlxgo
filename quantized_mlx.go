//go:build mlx

package mlx

/*
#cgo darwin,arm64 CFLAGS: -I/opt/homebrew/include
#include <stdlib.h>
#include <mlx/c/mlx.h>
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// MXFP8Matmul computes x @ w.T using MLX's packed-weight kernel, without
// quantizing x. x is Float32, Float16 or BFloat16 with shape [M,K]. w is
// UInt32 [N,K/4], with four E4M3FN bytes per word (first in the low byte).
// scales is UInt8 [N,K/32], containing E8M0 scale codes, not integer exponents.
// N and K must be positive multiples of 32; M must be positive.
// The output has shape [M,N] and x's dtype. Floating-point accumulation and
// exceptional-value handling follow MLX, not the CPU reference decoder.
func MXFP8Matmul(x, w, scales Array) (Array, error) {
	return withCurrentStreamValue(func(s C.mlx_stream) (Array, error) {
		xh, err := x.handleValue()
		if err != nil {
			return Array{}, err
		}
		wh, err := w.handleValue()
		if err != nil {
			return Array{}, err
		}
		sh, err := scales.handleValue()
		if err != nil {
			return Array{}, err
		}
		xs, ws, ss := x.Shape(), w.Shape(), scales.Shape()
		if len(xs) != 2 || len(ws) != 2 || len(ss) != 2 || xs[0] <= 0 || xs[1] <= 0 || xs[1]%32 != 0 || ws[0] <= 0 || ws[0]%32 != 0 || ws[1] != xs[1]/4 || ss[0] != ws[0] || ss[1] != xs[1]/32 {
			return Array{}, fmt.Errorf("mlxgo: invalid MXFP8 shapes x=%v w=%v scales=%v", xs, ws, ss)
		}
		xt, err := x.DType()
		if err != nil {
			return Array{}, err
		}
		wt, err := w.DType()
		if err != nil {
			return Array{}, err
		}
		st, err := scales.DType()
		if err != nil {
			return Array{}, err
		}
		if (xt != Float32 && xt != Float16 && xt != BFloat16) || wt != UInt32 || st != UInt8 {
			return Array{}, fmt.Errorf("mlxgo: MXFP8 requires floating x, UInt32 w and UInt8 scales")
		}
		mode := C.CString("mxfp8")
		defer C.free(unsafe.Pointer(mode))
		out := newArray(C.mlx_array_new())
		clearMLXError()
		if code := C.mlx_quantized_matmul(out.outHandle(), xh, wh, sh, C.mlx_array{}, true, C.mlx_optional_int{value: 32, has_value: true}, C.mlx_optional_int{value: 8, has_value: true}, mode, s); code != 0 {
			return closeArrayAfterError(out, mlxError("mlx_quantized_matmul", int(code)))
		}
		return out, nil
	})
}
