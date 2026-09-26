//go:build !mlx

package mlx

// GetMemoryUsage requires native MLX.
func GetMemoryUsage() (MemoryUsage, error) { return MemoryUsage{}, errBuiltWithoutMLX }

// ResetPeakMemory requires native MLX.
func ResetPeakMemory() error { return errBuiltWithoutMLX }
