//go:build mlx

package mlx

/*
#include <stdlib.h>
#include <mlx/c/mlx.h>
*/
import "C"

import (
	"errors"
	"log"
	"slices"
	"unsafe"
)

// GatherMM multiplies selected matrices from a and b. Both inputs have at
// least two dimensions; indices address their flattened batch dimensions.
// An empty Array omits that side's indices. Provided indices must be integers
// in range and broadcast together. sortedIndices promises that provided
// indices are sorted, allowing MLX to choose a faster implementation.
// The caller owns the lazy output; routing indices remain on the device.
func GatherMM(a, b, lhsIndices, rhsIndices Array, sortedIndices bool) (Array, error) {
	return withCurrentStreamValue(func(s C.mlx_stream) (Array, error) {
		ah, err := a.handleValue()
		if err != nil {
			return Array{}, err
		}
		bh, err := b.handleValue()
		if err != nil {
			return Array{}, err
		}
		as, bs := a.Shape(), b.Shape()
		if len(as) < 2 || len(bs) < 2 || as[len(as)-1] != bs[len(bs)-2] {
			return Array{}, errors.New("mlxgo: invalid GatherMM matrix shapes")
		}
		lh, err := gatherIndexHandle(lhsIndices)
		if err != nil {
			return Array{}, err
		}
		rh, err := gatherIndexHandle(rhsIndices)
		if err != nil {
			return Array{}, err
		}
		out := newArray(C.mlx_array_new())
		clearMLXError()
		if code := C.mlx_gather_mm(out.outHandle(), ah, bh, lh, rh, C.bool(sortedIndices), s); code != 0 {
			log.Printf("mlxgo: %v", mlxError("mlx_gather_mm", int(code)))
			return closeArrayAfterError(out, errors.New("mlxgo: GatherMM failed"))
		}
		return out, nil
	})
}

// GatherMXFP4 computes selected x @ weights.T products without quantizing x.
// x has shape [...,M,K] and dtype Float32, Float16 or BFloat16. weights is
// UInt32 [...,N,K/8], packing eight E2M1 nibbles per word, low nibble first.
// scales is UInt8 [...,N,K/32] with E8M0 codes and the same batch dimensions
// as weights. M is positive; N and K are positive multiples of 32.
// Indices follow GatherMM's contract. The caller owns the lazy output, with
// x's dtype; accumulation and underflow follow MLX.
func GatherMXFP4(x, weights, scales, lhsIndices, rhsIndices Array, sortedIndices bool) (Array, error) {
	return withCurrentStreamValue(func(s C.mlx_stream) (Array, error) {
		xh, err := x.handleValue()
		if err != nil {
			return Array{}, err
		}
		wh, err := weights.handleValue()
		if err != nil {
			return Array{}, err
		}
		sh, err := scales.handleValue()
		if err != nil {
			return Array{}, err
		}
		xs, ws, ss := x.Shape(), weights.Shape(), scales.Shape()
		if len(xs) < 2 || len(ws) < 2 || len(ss) != len(ws) {
			return Array{}, errors.New("mlxgo: invalid GatherMXFP4 matrix shapes")
		}
		m, k, n := xs[len(xs)-2], xs[len(xs)-1], ws[len(ws)-2]
		if m <= 0 || k <= 0 || k%32 != 0 || n <= 0 || n%32 != 0 ||
			ws[len(ws)-1] != k/8 || ss[len(ss)-1] != k/32 ||
			!slices.Equal(ws[:len(ws)-1], ss[:len(ss)-1]) {
			return Array{}, errors.New("mlxgo: invalid GatherMXFP4 matrix shapes")
		}
		xt, err := x.DType()
		if err != nil {
			return Array{}, err
		}
		wt, err := weights.DType()
		if err != nil {
			return Array{}, err
		}
		st, err := scales.DType()
		if err != nil {
			return Array{}, err
		}
		if (xt != Float32 && xt != Float16 && xt != BFloat16) || wt != UInt32 || st != UInt8 {
			return Array{}, errors.New("mlxgo: GatherMXFP4 requires floating x, UInt32 weights and UInt8 scales")
		}
		lh, err := gatherIndexHandle(lhsIndices)
		if err != nil {
			return Array{}, err
		}
		rh, err := gatherIndexHandle(rhsIndices)
		if err != nil {
			return Array{}, err
		}
		mode := C.CString("mxfp4")
		defer C.free(unsafe.Pointer(mode))
		out := newArray(C.mlx_array_new())
		clearMLXError()
		if code := C.mlx_gather_qmm(out.outHandle(), xh, wh, sh, C.mlx_array{}, lh, rh, true,
			C.mlx_optional_int{value: 32, has_value: true}, C.mlx_optional_int{value: 4, has_value: true}, mode, C.bool(sortedIndices), s); code != 0 {
			log.Printf("mlxgo: %v", mlxError("mlx_gather_qmm", int(code)))
			return closeArrayAfterError(out, errors.New("mlxgo: GatherMXFP4 failed"))
		}
		return out, nil
	})
}

func gatherIndexHandle(indices Array) (C.mlx_array, error) {
	if indices.state == nil {
		return C.mlx_array{}, nil
	}
	h, err := indices.handleValue()
	if err != nil {
		return C.mlx_array{}, err
	}
	dtype, err := indices.DType()
	if err != nil {
		return C.mlx_array{}, err
	}
	switch dtype {
	case UInt8, UInt16, UInt32, UInt64, Int8, Int16, Int32, Int64:
		return h, nil
	default:
		return C.mlx_array{}, errors.New("mlxgo: gather indices must have integer dtype")
	}
}
