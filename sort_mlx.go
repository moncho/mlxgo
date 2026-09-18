//go:build mlx

package mlx

/*
#cgo darwin,arm64 CFLAGS: -I/opt/homebrew/include
#include <mlx/c/mlx.h>
*/
import "C"

// SortAxis sorts values in ascending order along axis.
func SortAxis(a Array, axis int) (Array, error) {
	return unaryOp(a, "mlx_sort_axis", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_sort_axis(out, input, C.int(axis), stream)
	})
}

// ArgSortAxis returns indices that sort values in ascending order along axis.
func ArgSortAxis(a Array, axis int) (Array, error) {
	return unaryOp(a, "mlx_argsort_axis", func(out *C.mlx_array, input C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_argsort_axis(out, input, C.int(axis), stream)
	})
}
