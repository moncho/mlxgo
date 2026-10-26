//go:build !mlx

package mlx

// GatherMM multiplies selected matrices from a and b. Native execution
// requires the mlx build tag.
func GatherMM(a, b, lhsIndices, rhsIndices Array, sortedIndices bool) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}

// GatherMXFP4 multiplies selected matrices with packed E2M1 weights and E8M0
// scales. Native execution requires the mlx build tag.
func GatherMXFP4(x, weights, scales, lhsIndices, rhsIndices Array, sortedIndices bool) (Array, error) {
	return Array{}, errBuiltWithoutMLX
}
