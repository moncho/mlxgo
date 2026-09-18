//go:build mlx && mlxruntime

package deepseek

import (
	"fmt"
	"math"
	"slices"
	"sync"
	"testing"

	mlx "github.com/moncho/mlxgo"
)

func owned(t *testing.T, a mlx.Array, err error) mlx.Array {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = a.Close() })
	return a
}
func tensor(t *testing.T, f fixture, name string) mlx.Array {
	t.Helper()
	v, ok := f.Tensors[name]
	if !ok {
		t.Fatalf("missing tensor %s", name)
	}
	a, err := mlx.NewFloat32(v.Data, v.Shape)
	return owned(t, a, err)
}
func matches(t *testing.T, f fixture, a mlx.Array, name string) {
	t.Helper()
	v := f.Tensors[name]
	if !slices.Equal(a.Shape(), v.Shape) {
		t.Fatalf("%s: got shape %v, want %v", name, a.Shape(), v.Shape)
	}
	cast, err := mlx.AsType(a, mlx.Float32)
	if err != nil {
		t.Fatal(err)
	}
	defer cast.Close()
	data, err := cast.Float32Data()
	if err != nil {
		t.Fatal(err)
	}
	maxError := 0.0
	for i, want := range v.Data {
		diff := math.Abs(float64(data[i] - want))
		maxError = math.Max(maxError, diff)
		if math.IsNaN(float64(data[i])) || diff > 3e-5*(1+math.Abs(float64(want))) {
			t.Fatalf("%s[%d]: got %g, want %g", name, i, data[i], want)
		}
	}
	t.Logf("%s max absolute error %.3g", name, maxError)
}
func routingConfig() RouterConfig {
	return RouterConfig{TopK: 2, Score: "sqrtsoftplus", Temperature: .7, Scale: 1.5, Normalize: true}
}
func hyperConfig() HyperConfig {
	return HyperConfig{Streams: 3, Iterations: 20, NormEpsilon: 1e-20, Epsilon: 1e-6}
}
func expertFixture(t *testing.T, f fixture, prefix string) ExpertWeights {
	return ExpertWeights{Gate: tensor(t, f, prefix+"_gate"), Up: tensor(t, f, prefix+"_up"), Down: tensor(t, f, prefix+"_down")}
}
func sparseIndices(f fixture) []int32 {
	data := f.Tensors["sparse_indices"].Data
	ids := make([]int32, len(data))
	for i, value := range data {
		ids[i] = int32(value)
	}
	return ids
}

func gradient(t *testing.T, f fixture, input mlx.Array, fn func(mlx.Array) (mlx.Array, error), expected string) {
	t.Helper()
	vg, err := mlx.NewValueAndGrad(func(in []mlx.Array) ([]mlx.Array, error) {
		y, err := fn(in[0])
		if err != nil {
			return nil, err
		}
		defer y.Close()
		sq, err := mlx.Square(y)
		if err != nil {
			return nil, err
		}
		defer sq.Close()
		loss, err := mlx.Mean(sq, false)
		if err != nil {
			return nil, err
		}
		return []mlx.Array{loss}, nil
	}, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer vg.Close()
	values, grads, err := vg.Apply(input)
	if err != nil {
		t.Fatal(err)
	}
	defer mlx.CloseArrays(values)
	defer mlx.CloseArrays(grads)
	if len(grads) != 1 {
		t.Fatal("unexpected gradient count")
	}
	matches(t, f, grads[0], expected)
}

func TestReferencePrimitives(t *testing.T) {
	f := readFixture(t)
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			t.Run("routing", func(t *testing.T) {
				x, w, b := tensor(t, f, "x"), tensor(t, f, "router_weight"), tensor(t, f, "router_bias")
				for _, score := range []string{"sqrtsoftplus", "sigmoid", "softmax"} {
					c := routingConfig()
					c.Score = score
					weights, ids, err := Route(x, w, b, c)
					if err != nil {
						t.Fatal(err)
					}
					defer weights.Close()
					defer ids.Close()
					matches(t, f, weights, "route_"+score)
					matches(t, f, ids, "indices_"+score)
				}
				c := routingConfig()
				c.TopK = 1
				weights, ids, err := Route(x, w, b, c)
				if err != nil {
					t.Fatal(err)
				}
				defer weights.Close()
				defer ids.Close()
				matches(t, f, weights, "route_top1")
				matches(t, f, ids, "indices_top1")
				c = routingConfig()
				c.Normalize = false
				weights2, ids2, err := Route(x, w, b, c)
				if err != nil {
					t.Fatal(err)
				}
				defer weights2.Close()
				defer ids2.Close()
				matches(t, f, weights2, "route_unnormalized")
			})
			t.Run("moe", func(t *testing.T) {
				x, w, b := tensor(t, f, "x"), tensor(t, f, "router_weight"), tensor(t, f, "router_bias")
				experts := make([]ExpertWeights, 4)
				for i := range experts {
					experts[i] = expertFixture(t, f, fmt.Sprintf("expert%d", i))
				}
				shared := expertFixture(t, f, "shared")
				fn := func(x mlx.Array) (mlx.Array, error) { return MoE(x, w, b, experts, shared, routingConfig(), .7) }
				y, err := fn(x)
				owned(t, y, err)
				matches(t, f, y, "moe")
				gradient(t, f, x, fn, "moe_gradient")
			})
			t.Run("hyper", func(t *testing.T) {
				x, w, scale, base := tensor(t, f, "hx"), tensor(t, f, "hyper_projection"), tensor(t, f, "hyper_scale"), tensor(t, f, "hyper_base")
				pre, post, comb, err := HyperMix(x, w, scale, base, hyperConfig())
				if err != nil {
					t.Fatal(err)
				}
				defer pre.Close()
				defer post.Close()
				defer comb.Close()
				matches(t, f, pre, "pre")
				matches(t, f, post, "post")
				matches(t, f, comb, "comb")
				collapsed, err := HyperPre(x, pre)
				owned(t, collapsed, err)
				matches(t, f, collapsed, "collapsed")
				expanded, err := HyperPost(collapsed, x, post, comb)
				owned(t, expanded, err)
				matches(t, f, expanded, "expanded")
				gradient(t, f, x, func(x mlx.Array) (mlx.Array, error) {
					pre, post, comb, err := HyperMix(x, w, scale, base, hyperConfig())
					if err != nil {
						return mlx.Array{}, err
					}
					defer pre.Close()
					defer post.Close()
					defer comb.Close()
					h, err := HyperPre(x, pre)
					if err != nil {
						return mlx.Array{}, err
					}
					defer h.Close()
					return HyperPost(h, x, post, comb)
				}, "hyper_gradient")
			})
			t.Run("sparse", func(t *testing.T) {
				q, kv, sink := tensor(t, f, "q"), tensor(t, f, "kv"), tensor(t, f, "sink")
				fn := func(q mlx.Array) (mlx.Array, error) { return SparseAttention(q, kv, sink, sparseIndices(f), 3, .5) }
				y, err := fn(q)
				owned(t, y, err)
				matches(t, f, y, "attention")
				gradient(t, f, q, fn, "attention_gradient")
			})
			t.Run("compression", func(t *testing.T) {
				x, kv, gate, norm := tensor(t, f, "cx"), tensor(t, f, "compress_kv"), tensor(t, f, "compress_gate"), tensor(t, f, "compress_norm")
				for _, ratio := range []int{1, 2, 3} {
					y, err := CompressComplete(x, kv, gate, norm, ratio, 1e-20)
					owned(t, y, err)
					matches(t, f, y, fmt.Sprintf("compress%d", ratio))
				}
			})
		})
	}
}

func TestPrimitiveValidation(t *testing.T) {
	if err := mlx.SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}
	f := readFixture(t)
	x, w, b := tensor(t, f, "x"), tensor(t, f, "router_weight"), tensor(t, f, "router_bias")
	c := routingConfig()
	c.TopK = 5
	if a, i, err := Route(x, w, b, c); err == nil {
		a.Close()
		i.Close()
		t.Fatal("accepted excessive top-k")
	}
	if a, i, err := Route(mlx.Array{}, w, b, routingConfig()); err == nil {
		a.Close()
		i.Close()
		t.Fatal("accepted closed input")
	}
	q, kv, sink := tensor(t, f, "q"), tensor(t, f, "kv"), tensor(t, f, "sink")
	for _, bad := range []int32{-2, 5} {
		ids := sparseIndices(f)
		ids[0] = bad
		if a, err := SparseAttention(q, kv, sink, ids, 3, .5); err == nil {
			a.Close()
			t.Fatal("accepted invalid gather index")
		}
	}
	if a, err := SparseAttention(q, kv, sink, sparseIndices(f)[:2], 3, .5); err == nil {
		a.Close()
		t.Fatal("accepted invalid indices length")
	}
	if a, err := CompressComplete(tensor(t, f, "cx"), tensor(t, f, "compress_kv"), tensor(t, f, "compress_gate"), tensor(t, f, "compress_norm"), 4, 1e-20); err == nil {
		a.Close()
		t.Fatal("accepted partial group")
	}
	// Closing outputs must not close any borrowed input.
	weights, ids, err := Route(x, w, b, routingConfig())
	if err != nil {
		t.Fatal(err)
	}
	weights.Close()
	ids.Close()
	matches(t, f, x, "x")
}

func TestConcurrentPrimitives(t *testing.T) {
	f := readFixture(t)
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			q, kv, sink := tensor(t, f, "q"), tensor(t, f, "kv"), tensor(t, f, "sink")
			ids, want := sparseIndices(f), f.Tensors["attention"].Data
			errs := make(chan error, 8)
			var wg sync.WaitGroup
			for range 8 {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for range 50 {
						y, err := SparseAttention(q, kv, sink, ids, 3, .5)
						if err != nil {
							errs <- err
							return
						}
						data, err := y.Float32Data()
						y.Close()
						if err != nil {
							errs <- err
							return
						}
						for i, v := range data {
							if math.IsNaN(float64(v)) || math.Abs(float64(v-want[i])) > 3e-5 {
								errs <- fmt.Errorf("concurrent result mismatch")
								return
							}
						}
					}
				}()
			}
			wg.Wait()
			close(errs)
			for err := range errs {
				t.Error(err)
			}
		})
	}
}

func TestRouterExtremeScoresAndTies(t *testing.T) {
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			x, err := mlx.NewFloat32([]float32{1, -1}, []int{2, 1})
			owned(t, x, err)
			w, err := mlx.NewFloat32([]float32{-80, -82}, []int{2, 1})
			owned(t, w, err)
			b, err := mlx.Zeros([]int{2}, mlx.Float32)
			owned(t, b, err)
			c := RouterConfig{TopK: 2, Score: "sqrtsoftplus", Temperature: 1, Scale: 1, Normalize: true}
			weights, ids, err := Route(x, w, b, c)
			if err != nil {
				t.Fatal(err)
			}
			defer weights.Close()
			defer ids.Close()
			data, err := weights.Float32Data()
			if err != nil {
				t.Fatal(err)
			}
			for row, logits := range [][2]float64{{-80, -82}, {82, 80}} {
				a, b := math.Sqrt(math.Log1p(math.Exp(logits[0]))), math.Sqrt(math.Log1p(math.Exp(logits[1])))
				for j, value := range []float64{a, b} {
					want := value / (a + b + 1e-20)
					if math.IsNaN(float64(data[row*2+j])) || math.Abs(float64(data[row*2+j])-want) > 1e-5 {
						t.Fatalf("extreme score weight %g, want %g", data[row*2+j], want)
					}
				}
			}
			zeroW, err := mlx.Zeros([]int{2, 1}, mlx.Float32)
			owned(t, zeroW, err)
			tieWeights, tieIDs, err := Route(x, zeroW, b, c)
			if err != nil {
				t.Fatal(err)
			}
			defer tieWeights.Close()
			defer tieIDs.Close()
			cast, err := mlx.AsType(tieIDs, mlx.Int32)
			owned(t, cast, err)
			got, err := cast.Int32Data()
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got, []int32{0, 1, 0, 1}) {
				t.Fatalf("tie ordering: %v", got)
			}
		})
	}
}
