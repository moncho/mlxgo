//go:build mlx && mlxruntime

package qwen2

import (
	"slices"
	"testing"

	mlx "github.com/moncho/mlxgo"
)

func TestCacheBlockGrowth(t *testing.T) {
	for _, device := range []mlx.DeviceType{mlx.DeviceCPU, mlx.DeviceGPU} {
		if err := mlx.SetDefaultDevice(device, 0); err != nil {
			t.Fatal(err)
		}
		for _, dtype := range []mlx.DType{mlx.Float32, mlx.BFloat16} {
			c := tinyConfig()
			c.NumKVHeads = 2
			cache := NewKVCache(c, dtype)
			for _, n := range []int{255, 2, 3} {
				end := cache.Offset + n
				makeData := func(length, offset int, delta float32) []float32 {
					data := make([]float32, c.NumKVHeads*length*c.HeadDim())
					for h := 0; h < c.NumKVHeads; h++ {
						for p := 0; p < length; p++ {
							for d := 0; d < c.HeadDim(); d++ {
								data[(h*length+p)*c.HeadDim()+d] = float32(h*16+(offset+p)%16+d) + delta
							}
						}
					}
					return data
				}
				makeArray := func(delta float32) mlx.Array {
					a, err := mlx.NewFloat32(makeData(n, cache.Offset, delta), []int{1, c.NumKVHeads, n, c.HeadDim()})
					if err != nil {
						t.Fatal(err)
					}
					defer a.Close()
					b, err := mlx.AsType(a, dtype)
					if err != nil {
						t.Fatal(err)
					}
					return b
				}
				for layer := 0; layer < c.NumLayers; layer++ {
					k, v := makeArray(0), makeArray(64)
					keys, values, err := cache.update(layer, k, v)
					if err != nil {
						t.Fatal(err)
					}
					if _, err := k.DType(); err == nil {
						t.Fatal("update did not consume input")
					}
					if !slices.Equal(keys.Shape(), []int{1, c.NumKVHeads, end, c.HeadDim()}) {
						t.Fatalf("view %v", keys.Shape())
					}
					if !slices.Equal(floatData(t, keys), makeData(end, 0, 0)) || !slices.Equal(floatData(t, values), makeData(end, 0, 64)) {
						t.Fatal("cache contents changed across growth")
					}
					keys.Close()
					values.Close()
				}
				cache.Offset = end
				if err := mlx.Eval(cache.Arrays()...); err != nil {
					t.Fatal(err)
				}
				want := ((end-1)/cacheBlockSize + 1) * cacheBlockSize
				if cache.capacity != want || cache.keys[0].Shape()[2] != want {
					t.Fatal("incorrect block capacity")
				}
			}
			cache.Close()
		}
	}
}

func TestCacheUpdateFailure(t *testing.T) {
	c := tinyConfig()
	cache := NewKVCache(c, mlx.BFloat16)
	defer cache.Close()
	k, err := mlx.Zeros([]int{1, c.NumKVHeads, 1, c.HeadDim()}, mlx.Float32)
	if err != nil {
		t.Fatal(err)
	}
	v, err := mlx.Zeros(k.Shape(), mlx.Float32)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err = cache.update(0, k, v); err == nil || !cache.invalid {
		t.Fatal("invalid dtype did not poison cache")
	}
	for _, a := range []mlx.Array{k, v} {
		if _, err := a.DType(); err == nil {
			t.Fatal("failed update did not consume input")
		}
	}
}
