//go:build mlx && mlxruntime

package deepseek

import (
	"slices"
	"testing"

	mlx "github.com/moncho/mlxgo"
)

func TestQuantizedModelReference(t *testing.T) {
	testReducedModelReference(t, readQuantizedModelFixture(t))
}

func TestQuantizedSessionIsolation(t *testing.T) {
	testReducedSessionIsolation(t, readQuantizedModelFixture(t))
}

func TestQuantizedSessionLifecycle(t *testing.T) {
	testReducedSessionLifecycle(t, readQuantizedModelFixture(t))
}

func checkCacheStorage(t *testing.T, config Config, layer int, cache layerCache, packed bool) {
	t.Helper()
	for _, part := range []struct {
		data, scales       mlx.Array
		count, cols, group int
		fp4                bool
	}{
		{cache.window, cache.windowScale, cache.windowLen, config.HeadDim, 32, false},
		{cache.compressed, cache.compressedScale, cache.compressedLen, config.HeadDim, 16, true},
		{cache.keys, cache.keyScale, cache.compressedLen, config.IndexDim, 32, true},
	} {
		if part.count == 0 {
			if len(part.data.Shape()) != 0 || len(part.scales.Shape()) != 0 {
				t.Fatalf("layer %d empty cache owns storage", layer)
			}
			continue
		}
		dtype, cols := mlx.Float32, part.cols
		if packed {
			dtype = mlx.UInt8
			if part.fp4 {
				cols /= 2
			}
			d, err := part.scales.DType()
			if err != nil || d != mlx.UInt8 || !slices.Equal(part.scales.Shape(), []int{1, part.count, part.cols / part.group}) {
				t.Fatalf("layer %d invalid packed scales: %v", layer, err)
			}
		} else if len(part.scales.Shape()) != 0 {
			t.Fatal("default cache allocated scales")
		}
		d, err := part.data.DType()
		if err != nil || d != dtype || !slices.Equal(part.data.Shape(), []int{1, part.count, cols}) {
			t.Fatalf("layer %d invalid cache storage %v %v: %v", layer, d, part.data.Shape(), err)
		}
	}
	if cache.pendingLen > 0 {
		d, err := cache.pending.DType()
		if err != nil || d != mlx.Float32 || !slices.Equal(cache.pending.Shape(), []int{1, cache.pendingLen, config.Dim}) {
			t.Fatal("pending compressor input was quantized or reshaped")
		}
	}
}

func TestQuantizedSessionOptions(t *testing.T) {
	if err := mlx.SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}
	f := readModelFixture(t)
	m := modelFromFixture(t, f.Config, f.Parameters)
	if _, err := m.NewSessionWithOptions(SessionOptions{QuantizedCaches: true}); err == nil {
		t.Fatal("accepted unaligned cache width")
	}
	f = readQuantizedModelFixture(t)
	m = modelFromFixture(t, f.Config, f.Parameters)
	options := SessionOptions{QuantizedCaches: true}
	s, err := m.NewSessionWithOptions(options)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	options.QuantizedCaches = false
	if !s.options.QuantizedCaches {
		t.Fatal("options were not copied")
	}
	if _, err := closeLogits(s.Step(f.Tokens[:3])); err != nil {
		t.Fatal(err)
	}
	before := make([][][]int32, len(s.layers))
	for i, c := range s.layers {
		for _, a := range []mlx.Array{c.window, c.windowScale, c.compressed, c.compressedScale, c.keys, c.keyScale} {
			before[i] = append(before[i], readCacheBytes(t, a))
		}
	}
	if _, err := closeLogits(s.Step(f.Tokens[3:4])); err != nil {
		t.Fatal(err)
	}
	for i, c := range s.layers {
		for j, a := range []mlx.Array{c.window, c.windowScale, c.compressed, c.compressedScale, c.keys, c.keyScale} {
			old, now := before[i][j], readCacheBytes(t, a)
			if j < 2 {
				// The three-row window rolls over by exactly one encoded row.
				width := len(old) / 3
				if !slices.Equal(old[width:], now[:2*width]) {
					t.Fatal("window entries were requantized or reordered")
				}
			} else if len(old) > 0 && !slices.Equal(old, now[:len(old)]) {
				t.Fatal("existing compressed entries changed")
			}
		}
	}
	var owned []mlx.Array
	for _, cache := range s.layers {
		owned = append(owned, cache.arrays()...)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	for _, a := range owned {
		if _, err := a.DType(); err == nil {
			t.Fatal("cache handle survived session close")
		}
	}
	m.Close()
	if _, err := m.NewSessionWithOptions(SessionOptions{QuantizedCaches: true}); err == nil {
		t.Fatal("accepted closed model")
	}
	var nilModel *Model
	if _, err := nilModel.NewSessionWithOptions(SessionOptions{QuantizedCaches: true}); err == nil {
		t.Fatal("accepted nil model")
	}
}

func readCacheBytes(t *testing.T, a mlx.Array) []int32 {
	t.Helper()
	if len(a.Shape()) == 0 {
		return nil
	}
	b, err := mlx.AsType(a, mlx.Int32)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	data, err := b.Int32Data()
	if err != nil {
		t.Fatal(err)
	}
	return data
}
