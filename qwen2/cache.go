package qwen2

import (
	"fmt"
	"slices"

	"github.com/moncho/mlxgo"
)

const cacheBlockSize = 256

// KVCache belongs to one generation; discard it after a forward failure.
type KVCache struct {
	keys, values                             []mlx.Array
	Offset                                   int
	capacity, kvHeads, headDim, maxPositions int
	dtype                                    mlx.DType
	invalid                                  bool
}

// NewKVCache reserves capacity lazily in 256-position blocks using the weights' dtype.
func NewKVCache(c Config, dtype mlx.DType) *KVCache {
	cache := &KVCache{capacity: cacheBlockSize, kvHeads: c.NumKVHeads, headDim: c.HeadDim(), maxPositions: c.MaxPositions, dtype: dtype}
	if c.Validate() != nil || (dtype != mlx.Float32 && dtype != mlx.BFloat16 && dtype != mlx.Float16) {
		cache.invalid = true
		return cache
	}
	cache.keys, cache.values = make([]mlx.Array, c.NumLayers), make([]mlx.Array, c.NumLayers)
	return cache
}

// update consumes k/v even on error; returned attention views are caller-owned.
func (c *KVCache) update(layer int, k, v mlx.Array) (keys, values mlx.Array, err error) {
	defer mlx.CloseArrays([]mlx.Array{k, v})
	if c == nil || c.invalid || layer < 0 || layer >= len(c.keys) {
		return keys, values, fmt.Errorf("qwen2: invalid cache")
	}
	defer func() {
		if err != nil {
			c.invalid = true
		}
	}()
	shape := k.Shape()
	if len(shape) != 4 || shape[0] != 1 || shape[1] != c.kvHeads || shape[3] != c.headDim || shape[2] <= 0 || !slices.Equal(shape, v.Shape()) {
		return keys, values, fmt.Errorf("qwen2: incompatible cache update shape")
	}
	if c.Offset < 0 || c.Offset > c.maxPositions || shape[2] > c.maxPositions-c.Offset {
		return keys, values, fmt.Errorf("qwen2: cache exceeds context")
	}
	for _, a := range []mlx.Array{k, v} {
		dtype, e := a.DType()
		if e != nil {
			return keys, values, e
		}
		if dtype != c.dtype {
			return keys, values, fmt.Errorf("qwen2: cache dtype %v, update dtype %v", c.dtype, dtype)
		}
	}
	end := c.Offset + shape[2]
	if end > c.capacity {
		c.capacity = ((end-1)/cacheBlockSize + 1) * cacheBlockSize
	}
	kNext, err := c.write(c.keys[layer], k, end)
	if err != nil {
		return keys, values, err
	}
	vNext, err := c.write(c.values[layer], v, end)
	if err != nil {
		kNext.Close()
		return keys, values, err
	}
	_ = mlx.CloseArrays([]mlx.Array{c.keys[layer], c.values[layer]})
	c.keys[layer], c.values[layer] = kNext, vNext
	start, stop, stride := []int{0, 0, 0, 0}, []int{1, c.kvHeads, end, c.headDim}, []int{1, 1, 1, 1}
	keys, err = mlx.Slice(kNext, start, stop, stride)
	if err != nil {
		return mlx.Array{}, mlx.Array{}, err
	}
	values, err = mlx.Slice(vNext, start, stop, stride)
	if err != nil {
		keys.Close()
		return mlx.Array{}, mlx.Array{}, err
	}
	return keys, values, nil
}

func (c *KVCache) write(old, update mlx.Array, end int) (mlx.Array, error) {
	s := &scope{}
	defer s.close()
	buffer := old
	shape := old.Shape()
	if len(shape) == 0 && c.Offset != 0 {
		return mlx.Array{}, fmt.Errorf("qwen2: missing cache buffer")
	}
	if len(shape) == 0 || shape[2] < c.capacity {
		buffer = s.add(mlx.Zeros([]int{1, c.kvHeads, c.capacity, c.headDim}, c.dtype))
		if c.Offset > 0 {
			stop := []int{1, c.kvHeads, c.Offset, c.headDim}
			previous := s.add(mlx.Slice(old, []int{0, 0, 0, 0}, stop, []int{1, 1, 1, 1}))
			buffer = s.add(mlx.SliceUpdate(buffer, previous, []int{0, 0, 0, 0}, stop, []int{1, 1, 1, 1}))
		}
	}
	if s.err != nil {
		return mlx.Array{}, s.err
	}
	out := s.add(mlx.SliceUpdate(buffer, update, []int{0, 0, c.Offset, 0}, []int{1, c.kvHeads, end, c.headDim}, []int{1, 1, 1, 1}))
	return s.take(out)
}

func (c *KVCache) Arrays() []mlx.Array { return append(append([]mlx.Array{}, c.keys...), c.values...) }
func (c *KVCache) Close() error {
	if c == nil {
		return nil
	}
	c.invalid = true
	return mlx.CloseArrays(c.Arrays())
}
