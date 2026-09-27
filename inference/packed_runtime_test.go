//go:build mlx && mlxruntime

package inference

import (
	"fmt"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/deepseek"
	"github.com/moncho/mlxgo/deepseek/quant"
)

func TestPackedDeepSeekBundle(t *testing.T) {
	defer mlx.SetDefaultCPU()
	if err := mlx.SetDefaultGPU(); err != nil {
		t.Fatal(err)
	}
	c := readFixture(t).Config
	c.Dim = 32
	c.InterDim = 32
	c.QRank = 32
	c.HeadDim = 32
	c.ORank = 32
	c.IndexDim = 32
	shapes, err := c.ParameterShapes()
	if err != nil {
		t.Fatal(err)
	}
	params := map[string]mlx.Array{}
	defer func() {
		for _, a := range params {
			a.Close()
		}
	}()
	for name, shape := range shapes {
		n := 1
		for _, d := range shape {
			n *= d
		}
		v := make([]float32, n)
		for i := range v {
			v[i] = float32((i*37+len(name)*11)%127-63) / 512
			if strings.Contains(name, "norm.weight") {
				v[i] = 1
			}
		}
		a, err := mlx.NewFloat32(v, shape)
		if err != nil {
			t.Fatal(err)
		}
		params[name] = a
	}
	packed := map[string][]byte{}
	for _, part := range []string{"w1", "w2", "w3"} {
		name := fmt.Sprintf("layers.0.ffn.experts.0.%s.weight", part)
		shape := shapes[name]
		data := make([]byte, shape[0]*shape[1]/2)
		scales := make([]byte, shape[0]*shape[1]/32)
		for i := range data {
			data[i] = byte(i*37 + len(part))
		}
		for i := range scales {
			scales[i] = 120
		}
		decoded := make([]float32, shape[0]*shape[1])
		if err := quant.Decode(decoded, data, scales, shape[0], shape[1], quant.FP4Row32, quant.Float32); err != nil {
			t.Fatal(err)
		}
		a, err := mlx.NewFloat32(decoded, shape)
		if err != nil {
			t.Fatal(err)
		}
		old := params[name]
		old.Close()
		params[name] = a
		packed[name] = data
		packed[strings.TrimSuffix(name, ".weight")+".scale"] = scales
	}
	dir := t.TempDir()
	writeConfig(t, dir, c)
	if err := mlx.SaveSafetensors(filepath.Join(dir, "model.safetensors"), params, nil); err != nil {
		t.Fatal(err)
	}
	base, err := Open(dir, Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer base.Close()
	want, err := base.GenerateTokens([]int32{1, 5, 2, 9, 4}, 4)
	if err != nil {
		t.Fatal(err)
	}
	for name, data := range packed {
		shape := []int{32, 16}
		if strings.HasSuffix(name, ".scale") {
			shape = []int{32, 1}
		}
		a, err := mlx.NewUInt8(data, shape)
		if err != nil {
			t.Fatal(err)
		}
		old := params[name]
		old.Close()
		params[name] = a
	}
	packedDir := t.TempDir()
	writeConfig(t, packedDir, c)
	if err := mlx.SaveSafetensors(filepath.Join(packedDir, "model.safetensors"), params, nil); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{packedDir, shardBundle(t, packedDir)} {
		o := &DeepSeekOptions{Weights: deepseek.LoadOptions{FP4Experts: []deepseek.ExpertID{{Layer: 0, Index: 0}}}, Session: deepseek.SessionOptions{SparseExperts: true, QuantizedCaches: true}}
		m, err := Open(path, Options{DeepSeek: o})
		if err != nil {
			t.Fatal(err)
		}
		o.Weights.FP4Experts[0].Layer = 999
		o.Session.SparseExperts = false
		got, err := m.GenerateTokens([]int32{1, 5, 2, 9, 4}, 4)
		m.Close()
		if err != nil || !slices.Equal(got.Tokens, want.Tokens) {
			t.Fatal(got, want, err)
		}
		if m, err := Open(path, Options{}); err == nil {
			m.Close()
			t.Fatal("packed checkpoint silently accepted without selections")
		}
	}
}

func TestPackedDeepSeekRejectsSessionOptions(t *testing.T) {
	dir := exportFixture(t, nil)
	if m, err := Open(dir, Options{DeepSeek: &DeepSeekOptions{Session: deepseek.SessionOptions{QuantizedCaches: true}}}); err == nil {
		m.Close()
		t.Fatal("invalid cache dimensions accepted at Open")
	}
}
