package qwen2

import (
	"fmt"
	"github.com/moncho/mlxgo"
)

// KVCache belongs to one generation; do not share it between goroutines.
// Forward advances Offset. After a forward failure, discard the cache.
type KVCache struct {
	keys, values []mlx.Array
	Offset       int
	invalid      bool
}

func NewKVCache(layers int) *KVCache {
	if layers < 0 {
		layers = 0
	}
	return &KVCache{keys: make([]mlx.Array, layers), values: make([]mlx.Array, layers)}
}

// update consumes k and v even on error. Returned arrays remain cache-owned.
func (c *KVCache) update(layer int, k, v mlx.Array) (mlx.Array, mlx.Array, error) {
	if c == nil || c.invalid || layer < 0 || layer >= len(c.keys) {
		_ = mlx.CloseArrays([]mlx.Array{k, v})
		return mlx.Array{}, mlx.Array{}, fmt.Errorf("qwen2: invalid cache")
	}
	if c.Offset == 0 {
		c.keys[layer], c.values[layer] = k, v
		return k, v, nil
	}
	defer mlx.CloseArrays([]mlx.Array{k, v})
	kAll, err := mlx.ConcatenateAxis([]mlx.Array{c.keys[layer], k}, 2)
	if err != nil {
		c.invalid = true
		return mlx.Array{}, mlx.Array{}, err
	}
	vAll, err := mlx.ConcatenateAxis([]mlx.Array{c.values[layer], v}, 2)
	if err != nil {
		_ = kAll.Close()
		c.invalid = true
		return mlx.Array{}, mlx.Array{}, err
	}
	_ = mlx.CloseArrays([]mlx.Array{c.keys[layer], c.values[layer]})
	c.keys[layer], c.values[layer] = kAll, vAll
	return kAll, vAll, nil
}

func (c *KVCache) Arrays() []mlx.Array { return append(append([]mlx.Array{}, c.keys...), c.values...) }
func (c *KVCache) Close() error {
	if c == nil {
		return nil
	}
	c.invalid = true
	return mlx.CloseArrays(c.Arrays())
}
