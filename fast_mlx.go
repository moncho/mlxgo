//go:build mlx

package mlx

/*
#cgo darwin,arm64 CFLAGS: -I/opt/homebrew/include
#cgo darwin,arm64 CXXFLAGS: -I/opt/homebrew/include -std=c++17
#include <stdlib.h>
#include "fast_compat.h"
*/
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

// RMSNorm normalizes the last axis and multiplies by weight.
func RMSNorm(x, weight Array, eps float32) (Array, error) {
	if !positiveFinite(eps) {
		return Array{}, fmt.Errorf("mlxgo: RMSNorm epsilon must be positive and finite")
	}
	return binaryOp(x, weight, "mlx_fast_rms_norm", func(out *C.mlx_array, x, w C.mlx_array, s C.mlx_stream) C.int {
		return C.mlx_fast_rms_norm(out, x, w, C.float(eps), s)
	})
}

// RoPE rotates dims features; offset is the starting position on axis -2.
func RoPE(x Array, dims int, traditional bool, base, scale float32, offset int) (Array, error) {
	if dims <= 0 || dims%2 != 0 || dims > math.MaxInt32 || offset < 0 || offset > math.MaxInt32 || !positiveFinite(base) || !positiveFinite(scale) {
		return Array{}, fmt.Errorf("mlxgo: invalid RoPE dimensions, offset, base or scale")
	}
	return unaryOp(x, "mlx_fast_rope", func(out *C.mlx_array, x C.mlx_array, s C.mlx_stream) C.int {
		return C.mlx_fast_rope(out, x, C.int(dims), C.bool(traditional), C.mlx_optional_float{value: C.float(base), has_value: true}, C.float(scale), C.int(offset), C.mlx_array{}, s)
	})
}

// ScaledDotProductAttention applies attention with optional causal masking.
// maskMode must be "" or "causal". Grouped query attention is supported.
func ScaledDotProductAttention(q, k, v Array, scale float32, maskMode string) (Array, error) {
	if maskMode != "" && maskMode != "causal" || !positiveFinite(scale) {
		return Array{}, fmt.Errorf("mlxgo: invalid attention mask mode or scale")
	}
	return withCurrentStreamValue(func(s C.mlx_stream) (Array, error) {
		qh, err := q.handleValue()
		if err != nil {
			return Array{}, err
		}
		kh, err := k.handleValue()
		if err != nil {
			return Array{}, err
		}
		vh, err := v.handleValue()
		if err != nil {
			return Array{}, err
		}
		mask := C.CString(maskMode)
		defer C.free(unsafe.Pointer(mask))
		out := newArray(C.mlx_array_new())
		clearMLXError()
		if code := C.mlxgo_sdpa(out.outHandle(), qh, kh, vh, C.float(scale), mask, s); code != 0 {
			return closeArrayAfterError(out, mlxError("mlx_fast_scaled_dot_product_attention", int(code)))
		}
		return out, nil
	})
}

func positiveFinite(v float32) bool {
	return v > 0 && !math.IsInf(float64(v), 0) && !math.IsNaN(float64(v))
}
