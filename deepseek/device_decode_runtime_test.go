//go:build mlx && mlxruntime

package deepseek

import (
	"fmt"
	"maps"
	"math"
	"slices"
	"sync"
	"testing"

	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/lm"
)

func deviceDecodeModel(t *testing.T, storage string) *Model {
	t.Helper()
	c, parameters, packed, _ := fp4ModelInputsWithCompression(t, true)
	c.MaxSeq = 16
	inputs := maps.Clone(parameters)
	options := ModelOptions{FP4Experts: map[ExpertID]FP4ExpertWeights{}}
	for id, weight := range packed {
		selected := storage == "fp4" || (storage == "layerwise" && id.Layer == 0) || (storage == "mixed_routed" && id.Layer == 1 && id.Index == 0) || (storage == "mixed_shared" && id.Layer == 1 && id.Index == -1)
		if !selected {
			continue
		}
		options.FP4Experts[id] = weight
		for _, part := range []string{"w1.weight", "w2.weight", "w3.weight"} {
			delete(inputs, expertPrefix(id)+part)
		}
	}
	m, err := NewModelWithOptions(c, inputs, options)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	for _, a := range parameters {
		a.Close()
	}
	return m
}

func compareDeviceDecodeLogits(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("logits length %d, want %d", len(got), len(want))
	}
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Abs(float64(v-want[i])) > 2e-5*(1+math.Abs(float64(want[i]))) {
			t.Fatalf("logit %d = %g, want %g", i, v, want[i])
		}
	}
}

func deviceDecodeGreedy(m *Model, options SessionOptions, prompt []int32) ([]int32, error) {
	s, err := m.NewSessionWithOptions(options)
	if err != nil {
		return nil, err
	}
	defer s.Close()
	return lm.Greedy(s, prompt, 6)
}

func TestDeviceDecodeParity(t *testing.T) {
	defer mlx.SetDefaultCPU()
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			for _, storage := range []string{"float32", "fp4", "layerwise"} {
				t.Run(storage, func(t *testing.T) {
					m := deviceDecodeModel(t, storage)
					for _, packedCaches := range []bool{false, true} {
						t.Run(fmt.Sprintf("packed_cache_%t", packedCaches), func(t *testing.T) {
							tokens := []int32{1, 5, 2, 9, 4, 3}
							baseline := SessionOptions{SparseExperts: true, QuantizedCaches: packedCaches}
							want, err := sessionOutputOptions(m, tokens, baseline)
							if err != nil {
								t.Fatal(err)
							}
							wantTokens, err := deviceDecodeGreedy(m, baseline, tokens[:2])
							if err != nil {
								t.Fatal(err)
							}
							for _, sparseFlag := range []bool{false, true} {
								options := SessionOptions{DeviceDecode: true, SparseExperts: sparseFlag, QuantizedCaches: packedCaches}
								got, err := sessionOutputOptions(m, tokens, options)
								if err != nil {
									t.Fatal(err)
								}
								compareDeviceDecodeLogits(t, got, want)
								gotTokens, err := deviceDecodeGreedy(m, options, tokens[:2])
								if err != nil || !slices.Equal(gotTokens, wantTokens) {
									t.Fatalf("greedy tokens %v, want %v: %v", gotTokens, wantTokens, err)
								}
							}
						})
					}
				})
			}
		})
	}
}

func TestDeviceDecodePrefillDispatch(t *testing.T) {
	defer mlx.SetDefaultCPU()
	if err := mlx.SetDefaultGPU(); err != nil {
		t.Fatal(err)
	}
	for _, storage := range []string{"float32", "fp4"} {
		for _, prompt := range [][]int32{{1}, {1, 5, 2}} {
			for _, sparseFlag := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/tokens%d/sparse_%t", storage, len(prompt), sparseFlag), func(t *testing.T) {
					m := deviceDecodeModel(t, storage)
					baseline, err := m.NewSessionWithOptions(SessionOptions{SparseExperts: true})
					if err != nil {
						t.Fatal(err)
					}
					want, err := closeLogits(baseline.Step(prompt))
					baseline.Close()
					if err != nil {
						t.Fatal(err)
					}
					s, err := m.NewSessionWithOptions(SessionOptions{DeviceDecode: true, SparseExperts: sparseFlag})
					if err != nil {
						t.Fatal(err)
					}
					defer s.Close()
					// The original expert handles are independently owned bank views.
					// Closing only the gather bank must break decode, not sparse prefill.
					if err := m.deviceExperts[0].Close(); err != nil {
						t.Fatal(err)
					}
					got, err := closeLogits(s.Step(prompt))
					if err != nil {
						t.Fatal("prefill incorrectly used device dispatch", err)
					}
					compareDeviceDecodeLogits(t, got, want)
					if a, err := s.Step([]int32{3}); err == nil {
						a.Close()
						t.Fatal("decode did not use the device bank")
					}
				})
			}
		}
	}
}

func TestDeviceDecodeBankLifecycle(t *testing.T) {
	defer mlx.SetDefaultCPU()
	if err := mlx.SetDefaultGPU(); err != nil {
		t.Fatal(err)
	}
	for _, storage := range []string{"float32", "fp4"} {
		t.Run(storage, func(t *testing.T) {
			m := deviceDecodeModel(t, storage)
			tokens := []int32{1, 5, 2, 9, 4, 3}
			want, err := sessionOutputOptions(m, tokens, SessionOptions{SparseExperts: true})
			if err != nil {
				t.Fatal(err)
			}
			var old []*Session
			for _, sparse := range []bool{false, true} {
				s, err := m.NewSessionWithOptions(SessionOptions{SparseExperts: sparse})
				if err != nil {
					t.Fatal(err)
				}
				defer s.Close()
				old = append(old, s)
			}
			if len(m.deviceExperts) != 0 {
				t.Fatal("ordinary sessions prepared a device bank")
			}
			s, err := m.NewSessionWithOptions(SessionOptions{DeviceDecode: true})
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			banks := maps.Clone(m.deviceExperts)
			if len(banks) != m.config.Layers {
				t.Fatal("not every layer has a device bank")
			}
			s.Close()
			for _, before := range old {
				got, err := closeLogits(before.ForwardAll(tokens[:2]))
				if err != nil {
					t.Fatal(err)
				}
				for i := 2; i < len(tokens); i++ {
					v, err := closeLogits(before.ForwardAll(tokens[i : i+1]))
					if err != nil {
						t.Fatal(err)
					}
					got = append(got, v...)
				}
				compareDeviceDecodeLogits(t, got, want)
			}
			var wg sync.WaitGroup
			errs := make(chan error, 4)
			for range 4 {
				wg.Add(1)
				go func() {
					defer wg.Done()
					got, err := sessionOutputOptions(m, tokens, SessionOptions{DeviceDecode: true})
					if err == nil && len(got) != len(want) {
						err = fmt.Errorf("concurrent logits length %d, want %d", len(got), len(want))
					}
					if err == nil {
						for i, v := range got {
							if math.IsNaN(float64(v)) || math.Abs(float64(v-want[i])) > 2e-5*(1+math.Abs(float64(want[i]))) {
								err = fmt.Errorf("concurrent logits differ at %d", i)
								break
							}
						}
					}
					errs <- err
				}()
			}
			wg.Wait()
			close(errs)
			for err := range errs {
				if err != nil {
					t.Fatal(err)
				}
			}
			for layer, bank := range banks {
				if m.deviceExperts[layer] != bank || bank.closed {
					t.Fatal("session recreated or closed a model-owned bank")
				}
			}
			live, err := m.NewSessionWithOptions(SessionOptions{DeviceDecode: true})
			if err != nil {
				t.Fatal(err)
			}
			defer live.Close()
			if err := m.Close(); err != nil {
				t.Fatal(err)
			}
			for _, bank := range banks {
				if !bank.closed {
					t.Fatal("model did not close its bank")
				}
			}
			if a, err := live.Step([]int32{1}); err == nil {
				a.Close()
				t.Fatal("session used a closed model")
			}
			if next, err := m.NewSessionWithOptions(SessionOptions{DeviceDecode: true}); err == nil {
				next.Close()
				t.Fatal("closed model accepted device session")
			}
		})
	}
}

func TestDeviceDecodeRejectsMixedLayer(t *testing.T) {
	if err := mlx.SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}
	for _, storage := range []string{"mixed_routed", "mixed_shared"} {
		t.Run(storage, func(t *testing.T) {
			m := deviceDecodeModel(t, storage)
			tokens := []int32{1, 5, 2, 9, 4, 3}
			want, err := sessionOutputOptions(m, tokens, SessionOptions{SparseExperts: true})
			if err != nil {
				t.Fatal(err)
			}
			before := mlx.LiveArrays()
			for range 2 {
				if s, err := m.NewSessionWithOptions(SessionOptions{DeviceDecode: true}); err == nil {
					s.Close()
					t.Fatal("mixed expert layer accepted")
				}
				if mlx.LiveArrays() != before || len(m.deviceExperts) != 0 {
					t.Fatal("rejected options allocated or installed a partial bank")
				}
			}
			got, err := sessionOutputOptions(m, tokens, SessionOptions{SparseExperts: true})
			if err != nil {
				t.Fatal("rejected device options damaged sparse weights", err)
			}
			compareDeviceDecodeLogits(t, got, want)
		})
	}
}

func TestDeviceDecodeRetainedMemory(t *testing.T) {
	defer mlx.SetDefaultCPU()
	if err := mlx.SetDefaultGPU(); err != nil {
		t.Fatal(err)
	}
	for _, storage := range []string{"float32", "fp4"} {
		t.Run(storage, func(t *testing.T) {
			m := deviceDecodeModel(t, storage)
			// Dense warmup evaluates every source expert before the comparison.
			if _, err := closeLogits(m.Forward([]int32{1, 5})); err != nil {
				t.Fatal(err)
			}
			if err := mlx.Synchronize(); err != nil {
				t.Fatal(err)
			}
			before, err := mlx.GetMemoryUsage()
			if err != nil {
				t.Fatal(err)
			}
			s, err := m.NewSessionWithOptions(SessionOptions{DeviceDecode: true})
			if err != nil {
				t.Fatal(err)
			}
			s.Close()
			// Materialize every fallback view: an unevaluated TakeAxis graph
			// looks like a view until its first use allocates a copied expert.
			if _, err := closeLogits(m.Forward([]int32{1, 5})); err != nil {
				t.Fatal(err)
			}
			if _, err := deviceDecodeGreedy(m, SessionOptions{DeviceDecode: true}, []int32{1, 5}); err != nil {
				t.Fatal(err)
			}
			if err := mlx.Synchronize(); err != nil {
				t.Fatal(err)
			}
			after, err := mlx.GetMemoryUsage()
			if err != nil {
				t.Fatal(err)
			}
			if after.ActiveBytes > before.ActiveBytes+1024 {
				t.Fatalf("device preparation retained duplicate weights: active bytes %d -> %d", before.ActiveBytes, after.ActiveBytes)
			}
		})
	}
}

func TestDeviceDecodePreparationRollback(t *testing.T) {
	if err := mlx.SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}
	for _, storage := range []string{"float32", "fp4"} {
		t.Run(storage, func(t *testing.T) {
			m := deviceDecodeModel(t, storage)
			tokens := []int32{1, 5, 2, 9, 4, 3}
			want, err := sessionOutputOptions(m, tokens, SessionOptions{SparseExperts: true})
			if err != nil {
				t.Fatal(err)
			}
			// Fail after preparing layer zero, without closing any model handle.
			prefix := expertPrefix(ExpertID{Layer: 1, Index: 0})
			var restore func()
			if storage == "float32" {
				name := prefix + "w1.weight"
				original := m.weights[name]
				m.weights[name] = mlx.Array{}
				restore = func() { m.weights[name] = original }
			} else {
				expert := m.fp4Experts[prefix]
				original := expert.gate
				expert.gate = nil
				restore = func() { expert.gate = original }
			}
			before := mlx.LiveArrays()
			s, err := m.NewSessionWithOptions(SessionOptions{DeviceDecode: true})
			restore()
			if err == nil {
				s.Close()
				t.Fatal("invalid projection accepted during preparation")
			}
			if mlx.LiveArrays() != before || len(m.deviceExperts) != 0 {
				t.Fatal("failed preparation leaked arrays or published partial banks")
			}
			got, err := sessionOutputOptions(m, tokens, SessionOptions{SparseExperts: true})
			if err != nil {
				t.Fatal("failed preparation damaged original model", err)
			}
			compareDeviceDecodeLogits(t, got, want)
			got, err = sessionOutputOptions(m, tokens, SessionOptions{DeviceDecode: true})
			if err != nil {
				t.Fatal("could not retry preparation after restored input", err)
			}
			compareDeviceDecodeLogits(t, got, want)
		})
	}
}
