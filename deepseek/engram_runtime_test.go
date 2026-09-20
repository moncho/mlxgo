//go:build mlx && mlxruntime

package deepseek

import (
	"fmt"
	"slices"
	"testing"

	mlx "github.com/moncho/mlxgo"
)

func TestEngramReference(t *testing.T) {
	f := readEngramFixture(t)
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			x := tensor(t, f.fixture, "x")
			w := EngramWeights{tensor(t, f.fixture, "embed.weight"), tensor(t, f.fixture, "wkv.weight"), tensor(t, f.fixture, "q_weight"), tensor(t, f.fixture, "k_weight")}
			for _, masked := range []bool{false, true} {
				name := "unmasked"
				var mask []bool
				if masked {
					name = "masked"
					for _, row := range f.Cases[1].Mask {
						mask = append(mask, row...)
					}
				}
				y, err := Engram(x, f.Lookup, mask, w, f.Epsilon)
				if err != nil {
					t.Fatal(err)
				}
				matches(t, f.fixture, y, name)
				if masked {
					data, err := y.Float32Data()
					if err != nil {
						t.Fatal(err)
					}
					original := f.Tensors["x"].Data
					stride := x.Shape()[2] * x.Shape()[3]
					for i, keep := range mask {
						if !keep && !slices.Equal(data[i*stride:(i+1)*stride], original[i*stride:(i+1)*stride]) {
							t.Fatal("masked residual changed")
						}
					}
				}
				y.Close()
				vg, err := mlx.NewValueAndGrad(func(in []mlx.Array) ([]mlx.Array, error) {
					y, err := Engram(in[0], f.Lookup, mask, EngramWeights{in[1], in[2], in[3], in[4]}, f.Epsilon)
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
				}, 0, 1, 2, 3, 4)
				if err != nil {
					t.Fatal(err)
				}
				values, grads, err := vg.Apply(x, w.Embedding, w.Projection, w.QueryNorm, w.KeyNorm)
				if err != nil {
					vg.Close()
					t.Fatal(err)
				}
				for i, key := range []string{"x", "embed.weight", "wkv.weight", "q_weight", "k_weight"} {
					matches(t, f.fixture, grads[i], name+"_grad_"+key)
				}
				mlx.CloseArrays(values)
				mlx.CloseArrays(grads)
				vg.Close()
			}
			zero, err := mlx.Zeros(x.Shape(), mlx.Float32)
			if err != nil {
				t.Fatal(err)
			}
			defer zero.Close()
			y, err := Engram(zero, f.Lookup, nil, w, f.Epsilon)
			if err != nil {
				t.Fatal(err)
			}
			defer y.Close()
			matches(t, f.fixture, y, "zero_output")
			matches(t, f.fixture, x, "x")
			edgeX := tensor(t, f.fixture, "edge_x")
			edgeW := EngramWeights{tensor(t, f.fixture, "edge_embed.weight"), tensor(t, f.fixture, "edge_wkv.weight"), tensor(t, f.fixture, "edge_q_weight"), tensor(t, f.fixture, "edge_k_weight")}
			edgeIDs := make([]int32, 6)
			edge, err := Engram(edgeX, edgeIDs, nil, edgeW, f.Epsilon)
			if err != nil {
				t.Fatal(err)
			}
			defer edge.Close()
			matches(t, f.fixture, edge, "edge_output")
			gradient(t, f.fixture, edgeX, func(x mlx.Array) (mlx.Array, error) { return Engram(x, edgeIDs, nil, edgeW, f.Epsilon) }, "edge_grad_x")
		})
	}
}

func TestEngramValidation(t *testing.T) {
	f := readEngramFixture(t)
	x := tensor(t, f.fixture, "x")
	w := EngramWeights{tensor(t, f.fixture, "embed.weight"), tensor(t, f.fixture, "wkv.weight"), tensor(t, f.fixture, "q_weight"), tensor(t, f.fixture, "k_weight")}
	for _, ids := range [][]int32{nil, f.Lookup[:1], slices.Repeat([]int32{-1}, len(f.Lookup)), slices.Repeat([]int32{int32(w.Embedding.Shape()[0])}, len(f.Lookup))} {
		if a, err := Engram(x, ids, nil, w, f.Epsilon); err == nil {
			a.Close()
			t.Fatal("accepted bad indices")
		}
	}
	if a, err := Engram(x, f.Lookup, []bool{}, w, f.Epsilon); err == nil {
		a.Close()
		t.Fatal("accepted bad mask")
	}
	if a, err := Engram(x, f.Lookup, nil, w, 0); err == nil {
		a.Close()
		t.Fatal("accepted zero epsilon")
	}
	bad := w
	bad.QueryNorm = mlx.Array{}
	if a, err := Engram(x, f.Lookup, nil, bad, f.Epsilon); err == nil {
		a.Close()
		t.Fatal("accepted closed weights")
	}
	wrong, err := mlx.AsType(w.Embedding, mlx.BFloat16)
	if err != nil {
		t.Fatal(err)
	}
	defer wrong.Close()
	bad = w
	bad.Embedding = wrong
	if a, err := Engram(x, f.Lookup, nil, bad, f.Epsilon); err == nil {
		a.Close()
		t.Fatal("accepted BF16 table")
	}
}

func TestEngramModelReference(t *testing.T) {
	testReducedModelReference(t, readModelFixturePath(t, "testdata/model_engram.json"))
}
func TestEngramSessionIsolation(t *testing.T) {
	testReducedSessionIsolation(t, readModelFixturePath(t, "testdata/model_engram.json"))
}
func TestEngramGreedy(t *testing.T) {
	testReducedGreedy(t, readModelFixturePath(t, "testdata/model_engram.json"))
}

func TestEngramModelLifecycle(t *testing.T) {
	testReducedSessionLifecycle(t, readModelFixturePath(t, "testdata/model_engram.json"))
	f := readModelFixturePath(t, "testdata/model_engram.json")
	c := f.Config.clone()
	m := modelFromFixture(t, c, f.Parameters)
	c.Engram.TokenMap[0] = 0
	c.Engram.Multipliers[0][0] = 1
	c.Engram.Primes[0][0] = 2
	c.Engram.Layers[0] = 1
	s, err := m.NewSession()
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if a, err := s.Step([]int32{-1}); err == nil {
		a.Close()
		t.Fatal("invalid token accepted")
	}
	if s.engram.Position() != 0 {
		t.Fatal("validation changed history")
	}
	got, err := closeLogits(s.ForwardAll(f.Tokens[:5]))
	if err != nil {
		t.Fatal(err)
	}
	compareLogits(t, got, f.Cases[0].Logits.Data[:5*f.Config.VocabSize])
	if len(s.engram.history) > f.Config.Engram.MaxNGram-1 || s.engram.Position() != 5 {
		t.Fatal("wrong history state")
	}
	broken := m.weights[fmt.Sprintf("layers.%d.engram.wkv.weight", f.Config.Engram.Layers[1])]
	broken.Close()
	if a, err := s.Step(f.Tokens[5:6]); err == nil {
		a.Close()
		t.Fatal("accepted closed Engram parameter")
	}
	if !s.invalid || s.Position() != 5 {
		t.Fatal("failed native operation did not invalidate session")
	}
	if a, err := s.Step(f.Tokens[5:6]); err == nil {
		a.Close()
		t.Fatal("continued invalid session")
	}
}
