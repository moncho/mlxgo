//go:build !mlx

package mlx

// MXFP8Matmul computes x @ w.T using packed E4M3FN weights and E8M0 scales.
// Native execution requires the mlx build tag.
func MXFP8Matmul(x, w, scales Array) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}
