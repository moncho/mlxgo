//go:build !mlx

package mlx

func SortAxis(_ Array, _ int) (Array, error)    { return Array{}, errBuiltWithoutMLX }
func ArgSortAxis(_ Array, _ int) (Array, error) { return Array{}, errBuiltWithoutMLX }
