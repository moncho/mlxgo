package qwen2

import (
	"fmt"
	"path/filepath"
	"slices"

	"github.com/moncho/mlxgo"
)

type Layer struct {
	InputNorm, PostAttnNorm    mlx.Array
	Wq, Bq, Wk, Bk, Wv, Bv, Wo mlx.Array
	Wgate, Wup, Wdown          mlx.Array
}

type Weights struct {
	Embed, Norm mlx.Array
	Layers      []Layer
	closed      bool
	swiglu      *mlx.Closure
}

func (l Layer) arrays() []mlx.Array {
	return []mlx.Array{l.InputNorm, l.PostAttnNorm, l.Wq, l.Bq, l.Wk, l.Bk, l.Wv, l.Bv, l.Wo, l.Wgate, l.Wup, l.Wdown}
}

func (w *Weights) Arrays() []mlx.Array {
	a := []mlx.Array{w.Embed, w.Norm}
	for _, l := range w.Layers {
		a = append(a, l.arrays()...)
	}
	return a
}

func (w *Weights) Close() error {
	if w == nil || w.closed {
		return nil
	}
	w.closed = true
	if w.swiglu != nil {
		_ = w.swiglu.Close()
	}
	return mlx.CloseArrays(w.Arrays())
}

// Load validates and loads the single-file Hugging Face checkpoint. Projection
// weights are transposed to the [in, out] layout expected by mlx.Linear.
func Load(dir string, c Config) (*Weights, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	f, err := mlx.LoadSafetensors(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	w := &Weights{Layers: make([]Layer, c.NumLayers)}
	err = mlx.Batch(func() error {
		get := func(dst *mlx.Array, name string, shape []int, transpose bool) error {
			a, err := f.Get(name)
			if err != nil {
				return fmt.Errorf("qwen2: %s: %w", name, err)
			}
			dt, err := a.DType()
			if err != nil || !slices.Equal(a.Shape(), shape) || dt != mlx.BFloat16 {
				_ = a.Close()
				return fmt.Errorf("qwen2: %s must be bfloat16 %v", name, shape)
			}
			if transpose {
				b, e := mlx.Transpose(a)
				_ = a.Close()
				if e != nil {
					return e
				}
				a = b
			}
			*dst = a
			return a.Eval()
		}
		if err := get(&w.Embed, "model.embed_tokens.weight", []int{c.VocabSize, c.HiddenSize}, false); err != nil {
			return err
		}
		if err := get(&w.Norm, "model.norm.weight", []int{c.HiddenSize}, false); err != nil {
			return err
		}
		kv := c.NumKVHeads * c.HeadDim()
		for i := range w.Layers {
			l := &w.Layers[i]
			for _, item := range []struct {
				dst       *mlx.Array
				name      string
				shape     []int
				transpose bool
			}{
				{&l.InputNorm, "input_layernorm.weight", []int{c.HiddenSize}, false},
				{&l.PostAttnNorm, "post_attention_layernorm.weight", []int{c.HiddenSize}, false},
				{&l.Wq, "self_attn.q_proj.weight", []int{c.HiddenSize, c.HiddenSize}, true},
				{&l.Bq, "self_attn.q_proj.bias", []int{c.HiddenSize}, false},
				{&l.Wk, "self_attn.k_proj.weight", []int{kv, c.HiddenSize}, true},
				{&l.Bk, "self_attn.k_proj.bias", []int{kv}, false},
				{&l.Wv, "self_attn.v_proj.weight", []int{kv, c.HiddenSize}, true},
				{&l.Bv, "self_attn.v_proj.bias", []int{kv}, false},
				{&l.Wo, "self_attn.o_proj.weight", []int{c.HiddenSize, c.HiddenSize}, true},
				{&l.Wgate, "mlp.gate_proj.weight", []int{c.IntermediateSize, c.HiddenSize}, true},
				{&l.Wup, "mlp.up_proj.weight", []int{c.IntermediateSize, c.HiddenSize}, true},
				{&l.Wdown, "mlp.down_proj.weight", []int{c.HiddenSize, c.IntermediateSize}, true},
			} {
				if err := get(item.dst, fmt.Sprintf("model.layers.%d.%s", i, item.name), item.shape, item.transpose); err != nil {
					return err
				}
			}
		}
		return nil
	})
	if err != nil {
		_ = w.Close()
		return nil, err
	}
	return w, nil
}
