//go:build mlx && mlxruntime

package deepseek

import (
	"fmt"
	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/lm"
	"math"
	"slices"
	"sync"
	"testing"
)

func modelFromFixture(t *testing.T, c Config, parameters map[string]tensorFixture) *Model {
	t.Helper()
	arrays := make(map[string]mlx.Array, len(parameters))
	defer func() {
		for _, a := range arrays {
			a.Close()
		}
	}()
	for name, value := range parameters {
		a, err := mlx.NewFloat32(value.Data, value.Shape)
		if err != nil {
			t.Fatal(err)
		}
		arrays[name] = a
	}
	m, err := NewModel(c, arrays)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}
func closeLogits(a mlx.Array, err error) ([]float32, error) {
	if err != nil {
		return nil, err
	}
	defer a.Close()
	return a.Float32Data()
}
func compareLogits(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("logits length %d, want %d", len(got), len(want))
	}
	worst := 0.
	for i, v := range got {
		e := math.Abs(float64(v - want[i]))
		worst = math.Max(worst, e)
		if math.IsNaN(float64(v)) || e > 2e-5 {
			t.Fatalf("logit %d = %g, want %g (error %g)", i, v, want[i], e)
		}
	}
	t.Logf("max logit error %.3g", worst)
}

func TestReducedModelReference(t *testing.T) {
	testReducedModelReference(t, readModelFixture(t))
}
func testReducedModelReference(t *testing.T, f modelFixture) {
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			for _, test := range f.Cases {
				t.Run(fmt.Sprintf("prefill_%d_yarn_%v", test.Prefill, test.Yarn), func(t *testing.T) {
					c := f.Config.clone()
					if test.Yarn {
						c.OriginalSeq = 4
					}
					m := modelFromFixture(t, c, f.Parameters)
					s, err := m.NewSessionWithOptions(SessionOptions{QuantizedCaches: f.CacheMode != ""})
					if err != nil {
						t.Fatal(err)
					}
					defer s.Close()
					data, err := closeLogits(s.ForwardAll(f.Tokens[:test.Prefill]))
					if err != nil {
						t.Fatal(err)
					}
					for pos := test.Prefill; pos < len(f.Tokens); pos++ {
						next, err := closeLogits(s.ForwardAll(f.Tokens[pos : pos+1]))
						if err != nil {
							t.Fatal(err)
						}
						data = append(data, next...)
					}
					compareLogits(t, data, test.Logits.Data)
					if s.Position() != len(f.Tokens) {
						t.Fatal("session offset mismatch")
					}
					fullSession, err := m.NewSessionWithOptions(SessionOptions{QuantizedCaches: f.CacheMode != ""})
					if err != nil {
						t.Fatal(err)
					}
					full, err := closeLogits(fullSession.ForwardAll(f.Tokens))
					fullSession.Close()
					if err != nil {
						t.Fatal(err)
					}
					compareLogits(t, data, full)
					for i, cache := range s.layers {
						checkCacheStorage(t, c, i, cache, f.CacheMode != "")
						if cache.windowLen != c.Window {
							t.Fatalf("layer %d window not bounded", i)
						}
						if source(c, i) && (cache.pendingLen != len(f.Tokens)%c.Ratios[i] || cache.compressedLen != len(f.Tokens)/c.Ratios[i]) {
							t.Fatalf("layer %d compression counters", i)
						}
					}
				})
			}
		})
	}
}

func TestReducedSessionIsolation(t *testing.T) {
	testReducedSessionIsolation(t, readModelFixture(t))
}
func testReducedSessionIsolation(t *testing.T, f modelFixture) {
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			a := modelFromFixture(t, f.Config, f.Parameters)
			b := modelFromFixture(t, f.Config, f.Second)
			errs := make(chan error, 8)
			var wg sync.WaitGroup
			for i := range 8 {
				wg.Add(1)
				go func(i int) {
					defer wg.Done()
					m, want := a, f.Cases[0].Logits.Data
					if i%2 == 1 {
						m, want = b, f.SecondLogits.Data
					}
					s, err := m.NewSessionWithOptions(SessionOptions{QuantizedCaches: f.CacheMode != ""})
					if err != nil {
						errs <- err
						return
					}
					defer s.Close()
					var data []float32
					for pos := range f.Tokens {
						next, err := closeLogits(s.ForwardAll(f.Tokens[pos : pos+1]))
						if err != nil {
							errs <- err
							return
						}
						data = append(data, next...)
					}
					for j, v := range data {
						if math.IsNaN(float64(v)) || math.Abs(float64(v-want[j])) > 2e-5 {
							errs <- fmt.Errorf("session %d logit %d mismatch", i, j)
							return
						}
					}
				}(i)
			}
			wg.Wait()
			close(errs)
			for err := range errs {
				t.Error(err)
			}
		})
	}
}

func TestReducedSessionLifecycle(t *testing.T) {
	testReducedSessionLifecycle(t, readModelFixture(t))
}
func testReducedSessionLifecycle(t *testing.T, f modelFixture) {
	if err := mlx.SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}
	m := modelFromFixture(t, f.Config, f.Parameters)
	s, err := m.NewSessionWithOptions(SessionOptions{QuantizedCaches: f.CacheMode != ""})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	for _, bad := range [][]int32{nil, {-1}, {int32(f.Config.VocabSize)}} {
		if a, err := s.Step(bad); err == nil {
			a.Close()
			t.Fatal("accepted invalid token input")
		}
	}
	if s.Position() != 0 {
		t.Fatal("validation mutated cache")
	}
	first, err := s.Step(f.Tokens[:3])
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(first.Shape(), []int{1, 1, f.Config.VocabSize}) {
		t.Fatal(first.Shape())
	}
	first.Close()
	if a, err := s.Step(f.Tokens[:2]); err == nil {
		a.Close()
		t.Fatal("accepted multi-token decode")
	}
	if s.Position() != 3 {
		t.Fatal("invalid decode advanced cache")
	}
	if _, err = closeLogits(s.Step(f.Tokens[3:4])); err != nil {
		t.Fatal(err)
	}
	s.Close()
	s.Close()
	if a, err := s.Step([]int32{1}); err == nil {
		a.Close()
		t.Fatal("used closed session")
	}
	other, err := m.NewSessionWithOptions(SessionOptions{QuantizedCaches: f.CacheMode != ""})
	if err != nil {
		t.Fatal(err)
	}
	defer other.Close()
	prompt := make([]int32, f.Config.MaxSeq)
	if _, err = closeLogits(other.Step(prompt)); err != nil {
		t.Fatal(err)
	}
	if a, err := other.Step([]int32{1}); err == nil {
		a.Close()
		t.Fatal("exceeded context")
	}
	m.Close()
	m.Close()
	if a, err := m.Forward([]int32{1}); err == nil {
		a.Close()
		t.Fatal("used closed model")
	}
	if _, err = m.NewSession(); err == nil {
		t.Fatal("session from closed model")
	}
}

func TestReducedGreedy(t *testing.T) {
	testReducedGreedy(t, readModelFixture(t))
}
func testReducedGreedy(t *testing.T, f modelFixture) {
	m := modelFromFixture(t, f.Config, f.Parameters)
	s, err := m.NewSession()
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	got, err := lm.Greedy(s, f.Tokens[:5], 4)
	if err != nil {
		t.Fatal(err)
	}
	sequence := slices.Clone(f.Tokens[:5])
	var want []int32
	for range 4 {
		data, err := closeLogits(m.Forward(sequence))
		if err != nil {
			t.Fatal(err)
		}
		last := data[len(data)-f.Config.VocabSize:]
		best := 0
		for i := range last {
			if last[i] > last[best] {
				best = i
			}
		}
		want = append(want, int32(best))
		sequence = append(sequence, int32(best))
	}
	if !slices.Equal(got, want) {
		t.Fatalf("greedy tokens %v, want %v", got, want)
	}
}

func TestReducedParameterValidation(t *testing.T) {
	if err := mlx.SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}
	f := readModelFixture(t)
	params := make(map[string]mlx.Array, len(f.Parameters))
	defer func() {
		for _, a := range params {
			a.Close()
		}
	}()
	for name, v := range f.Parameters {
		a, err := mlx.NewFloat32(v.Data, v.Shape)
		if err != nil {
			t.Fatal(err)
		}
		params[name] = a
	}
	embed := params["embed.weight"]
	delete(params, "embed.weight")
	if m, err := NewModel(f.Config, params); err == nil {
		m.Close()
		t.Fatal("accepted missing parameter")
	}
	params["unknown.weight"] = embed
	if m, err := NewModel(f.Config, params); err == nil {
		m.Close()
		t.Fatal("accepted unknown parameter")
	}
	delete(params, "unknown.weight")
	wrong, err := mlx.Reshape(embed, []int{f.Config.Dim, f.Config.VocabSize})
	if err != nil {
		t.Fatal(err)
	}
	params["embed.weight"] = wrong
	if m, err := NewModel(f.Config, params); err == nil {
		m.Close()
		t.Fatal("accepted transposed parameter shape")
	}
	wrong.Close()
	wrong, err = mlx.AsType(embed, mlx.BFloat16)
	if err != nil {
		t.Fatal(err)
	}
	params["embed.weight"] = wrong
	if m, err := NewModel(f.Config, params); err == nil {
		m.Close()
		t.Fatal("accepted non-float32 parameter")
	}
	wrong.Close()
	params["embed.weight"] = embed
	if _, err := embed.Float32Data(); err != nil {
		t.Fatal("failed construction closed borrowed input", err)
	}
	c := f.Config.clone()
	m, err := NewModel(c, params)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	c.Ratios[1] = 0
	c.KVSources[0] = 7
	c.IndexSources[0] = 7
	data, err := closeLogits(m.Forward(f.Tokens))
	if err != nil {
		t.Fatal(err)
	}
	compareLogits(t, data, f.Cases[0].Logits.Data)
	s, err := m.NewSession()
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	// Simulate a native/handle failure after construction. The session must not
	// attempt to continue using partly updated caches after a failed forward.
	broken := m.weights["layers.3.ffn.experts.0.w1.weight"]
	broken.Close()
	if a, err := s.ForwardAll(f.Tokens[:5]); err == nil {
		a.Close()
		t.Fatal("used invalid parameter handle")
	}
	if !s.invalid || s.Position() != 0 {
		t.Fatal("failed native forward did not invalidate session")
	}
	if a, err := s.Step([]int32{1}); err == nil {
		a.Close()
		t.Fatal("continued invalid session")
	}
}
