//go:build mlx

package mlx

/*
#cgo darwin,arm64 CFLAGS: -I/opt/homebrew/include
#include <mlx/c/mlx.h>
*/
import "C"

// GetMemoryUsage samples MLX's process-wide allocator counters on the worker.
// Other active models are included; this is not per-session accounting.
func GetMemoryUsage() (MemoryUsage, error) {
	return runMLXValue(func() (MemoryUsage, error) {
		var active, cache, peak C.size_t
		clearMLXError()
		if code := C.mlx_get_active_memory(&active); code != 0 {
			return MemoryUsage{}, mlxError("mlx_get_active_memory", int(code))
		}
		if code := C.mlx_get_cache_memory(&cache); code != 0 {
			return MemoryUsage{}, mlxError("mlx_get_cache_memory", int(code))
		}
		if code := C.mlx_get_peak_memory(&peak); code != 0 {
			return MemoryUsage{}, mlxError("mlx_get_peak_memory", int(code))
		}
		return MemoryUsage{ActiveBytes: uint64(active), CacheBytes: uint64(cache), PeakBytes: uint64(peak)}, nil
	})
}

// ResetPeakMemory resets MLX's process-wide peak allocation counter. It does
// not free memory. Use only in coordinated profiling, not inside inference calls.
func ResetPeakMemory() error {
	return runMLX(func() error {
		clearMLXError()
		if code := C.mlx_reset_peak_memory(); code != 0 {
			return mlxError("mlx_reset_peak_memory", int(code))
		}
		return nil
	})
}
