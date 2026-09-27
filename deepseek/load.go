package deepseek

import (
	"fmt"
	"slices"
	"strings"

	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/checkpoint"
)

// LoadOptions selects packed tensors in a prepared local bundle. Unselected
// parameters must be F32. Selected data and scales must be U8, not native F8
// safetensors dtypes. This does not load the released DeepSeek checkpoint.
type LoadOptions struct {
	FP4Experts                                     []ExpertID
	FP8AttentionKV, FP8AttentionQB, FP8AttentionOA []int
	// MaxBytes caps total on-disk tensor payload before native allocation.
	// Zero uses 1 GiB. This is not an RSS cap: shard maps, Go buffers and native
	// copies may coexist during loading. Negative limits are rejected.
	MaxBytes int64
}

type loadTensor struct {
	shape  []int
	packed bool
}

func loadLayout(c Config, o LoadOptions) (map[string]loadTensor, map[string]int, error) {
	shapes, err := c.ParameterShapes()
	if err != nil {
		return nil, nil, err
	}
	if o.MaxBytes < 0 {
		return nil, nil, fmt.Errorf("deepseek: negative checkpoint byte limit")
	}
	kinds := make(map[string]int)
	selectWeight := func(name string, bits int) error {
		shape, ok := shapes[name]
		if !ok || kinds[name] != 0 {
			return fmt.Errorf("deepseek: invalid or duplicate packed selection %s", name)
		}
		if shape[0]%32 != 0 || shape[1]%32 != 0 {
			return fmt.Errorf("deepseek: packed dimensions must be multiples of 32: %s", name)
		}
		kinds[name] = bits
		return nil
	}
	for _, id := range o.FP4Experts {
		if id.Layer < 0 || id.Layer >= c.Layers || id.Index < -1 || id.Index >= c.Experts {
			return nil, nil, fmt.Errorf("deepseek: invalid packed expert %+v", id)
		}
		for _, part := range []string{"w1.weight", "w2.weight", "w3.weight"} {
			if err := selectWeight(expertPrefix(id)+part, 4); err != nil {
				return nil, nil, err
			}
		}
	}
	for _, p := range []struct {
		name   string
		layers []int
	}{{"wkv", o.FP8AttentionKV}, {"wq_b", o.FP8AttentionQB}, {"wo_a", o.FP8AttentionOA}} {
		for _, layer := range p.layers {
			if err := selectWeight(fmt.Sprintf("layers.%d.attn.%s.weight", layer, p.name), 8); err != nil {
				return nil, nil, err
			}
		}
	}
	layout := make(map[string]loadTensor, len(shapes)+len(kinds))
	for name, shape := range shapes {
		bits := kinds[name]
		if bits == 0 {
			layout[name] = loadTensor{shape: shape}
			continue
		}
		r, c := shape[0], shape[1]
		sr := r
		if bits == 8 {
			sr = r / 32
		}
		layout[name] = loadTensor{shape: []int{r, c * bits / 8}, packed: true}
		layout[strings.TrimSuffix(name, ".weight")+".scale"] = loadTensor{shape: []int{sr, c / 32}, packed: true}
	}
	return layout, kinds, nil
}

// Load reads a prepared single-file or indexed safetensors bundle. Packed
// selections use the original parameter name for U8 data and replace .weight
// with .scale for U8 E8M0 scales. FP4 has [rows,cols/2] data and [rows,cols/32]
// scales; FP8 has [rows,cols] data and [rows/32,cols/32] block scales. The grouped
// output projection retains its BF16-exactness guard. All other tensors are F32.
// Exact inventory, shapes, dtypes and payload budget are checked before loading.
// Returned models own all weights independently of files and option slices.
func Load(dir string, c Config, options LoadOptions) (*Model, error) {
	layout, _, err := loadLayout(c, options)
	if err != nil {
		return nil, err
	}
	meta, err := checkpoint.Inspect(dir)
	if err != nil {
		return nil, err
	}
	if len(meta.Names()) != len(layout) {
		return nil, fmt.Errorf("deepseek: checkpoint tensor inventory mismatch")
	}
	budget := options.MaxBytes
	if budget == 0 {
		budget = 1 << 30
	}
	for name, want := range layout {
		got, ok := meta.Tensor(name)
		dtype := "F32"
		if want.packed {
			dtype = "U8"
		}
		if !ok || got.DType != dtype || !slices.Equal(got.Shape, want.shape) {
			return nil, fmt.Errorf("deepseek: checkpoint tensor %s must be %s %v", name, dtype, want.shape)
		}
		if got.Bytes > budget {
			return nil, fmt.Errorf("deepseek: checkpoint exceeds byte limit")
		}
		budget -= got.Bytes
	}
	r, err := checkpoint.Open(dir)
	if err != nil {
		return nil, err
	}
	defer r.Close()
	params := make(map[string]mlx.Array)
	packed := make(map[string][]byte)
	defer func() {
		for _, a := range params {
			a.Close()
		}
	}()
	// Stable shard order avoids reopening each shard for every parameter.
	names := meta.Names()
	slices.SortFunc(names, func(a, b string) int {
		x, _ := meta.Tensor(a)
		y, _ := meta.Tensor(b)
		if v := strings.Compare(x.File, y.File); v != 0 {
			return v
		}
		return strings.Compare(a, b)
	})
	for _, name := range names {
		a, err := r.Get(name)
		if err != nil {
			return nil, fmt.Errorf("deepseek: %s: %w", name, err)
		}
		want := layout[name]
		dtype, err := a.DType()
		expected := mlx.Float32
		if want.packed {
			expected = mlx.UInt8
		}
		if err != nil || dtype != expected || !slices.Equal(a.Shape(), want.shape) {
			a.Close()
			return nil, fmt.Errorf("deepseek: tensor %s changed after inspection", name)
		}
		if !want.packed {
			params[name] = a
			continue
		}
		data, err := a.UInt8Data()
		a.Close()
		if err != nil {
			return nil, fmt.Errorf("deepseek: %s: %w", name, err)
		}
		packed[name] = data
	}
	o := ModelOptions{FP4Experts: make(map[ExpertID]FP4ExpertWeights)}
	weight := func(name string) FP4Weight {
		return FP4Weight{Data: packed[name], Scales: packed[strings.TrimSuffix(name, ".weight")+".scale"]}
	}
	for _, id := range options.FP4Experts {
		p := expertPrefix(id)
		o.FP4Experts[id] = FP4ExpertWeights{Gate: weight(p + "w1.weight"), Up: weight(p + "w3.weight"), Down: weight(p + "w2.weight")}
	}
	for _, p := range []struct {
		name   string
		layers []int
		out    *map[int]FP8AttentionWeight
	}{{"wkv", options.FP8AttentionKV, &o.FP8AttentionKV}, {"wq_b", options.FP8AttentionQB, &o.FP8AttentionQB}, {"wo_a", options.FP8AttentionOA, &o.FP8AttentionOA}} {
		*p.out = make(map[int]FP8AttentionWeight)
		for _, layer := range p.layers {
			w := weight(fmt.Sprintf("layers.%d.attn.%s.weight", layer, p.name))
			(*p.out)[layer] = FP8AttentionWeight{Data: w.Data, Scales: w.Scales}
		}
	}
	return NewModelWithOptions(c, params, o)
}
