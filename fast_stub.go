//go:build !mlx

package mlx

func RMSNorm(x, weight Array, eps float32) (Array, error) { return Array{}, errBuiltWithoutMLX }
func RoPE(x Array, dims int, traditional bool, base, scale float32, offset int) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}
func ScaledDotProductAttention(q, k, v Array, scale float32, maskMode string) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}
