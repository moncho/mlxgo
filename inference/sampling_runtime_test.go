//go:build mlx && mlxruntime

package inference

import (
	"slices"
	"testing"

	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/lm"
)

func TestBundleSampling(t *testing.T) {
	if err := mlx.SetDefaultGPU(); err != nil {
		t.Fatal(err)
	}
	m, err := Open(exportFixture(t, nil), Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	prompt := []int32{1, 5, 2, 9, 4}
	a, err := m.GenerateTokens(prompt, 4)
	if err != nil {
		t.Fatal(err)
	}
	b, err := m.GenerateTokensWith(prompt, 4, lm.SamplingOptions{})
	if err != nil || !slices.Equal(a.Tokens, b.Tokens) {
		t.Fatalf("greedy mismatch: %+v %v", b, err)
	}
	o := lm.SamplingOptions{Temperature: .8, TopP: .9, Seed: 42}
	a, err = m.GenerateTokensWith(prompt, 4, o)
	if err != nil {
		t.Fatal(err)
	}
	b, err = m.GenerateTokensWith(prompt, 4, o)
	if err != nil || !slices.Equal(a.Tokens, b.Tokens) {
		t.Fatalf("sample mismatch: %+v %v", b, err)
	}
}
