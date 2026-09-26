package deepseek

import (
	"fmt"
	"math"

	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/deepseek/quant"
)

// Group boundaries must coincide with checkpoint scale blocks. Validate before
// slicing payloads or uploading any native handles, using constant scratch space.
func validateFP8OutputWeight(weight FP8AttentionWeight, groups, rank, width int) error {
	limit := int(^uint(0) >> 1)
	if groups <= 0 || rank <= 0 || width <= 0 || rank%32 != 0 || width%32 != 0 || groups > limit/rank || groups*rank > limit/width || rank > 1<<31-1 || width > 1<<31-1 {
		return fmt.Errorf("invalid grouped FP8 dimensions: groups=%d rank=%d width=%d", groups, rank, width)
	}
	rows := groups * rank
	if len(weight.Data) != rows*width || len(weight.Scales) != rows*width/1024 {
		return fmt.Errorf("incorrect grouped FP8 data or scale length")
	}
	var decoded [32]float32
	for row := 0; row < rows; row++ {
		for block := 0; block < width/32; block++ {
			i, j := row*width+block*32, row/32*(width/32)+block
			if err := quant.Decode(decoded[:], weight.Data[i:i+32], weight.Scales[j:j+1], 1, 32, quant.FP8Row32, quant.Float32); err != nil {
				return fmt.Errorf("grouped FP8 row %d block %d: %w", row, block, err)
			}
			for col, v := range decoded {
				if math.Float32bits(v) != math.Float32bits(quant.RoundBFloat16(v)) {
					return fmt.Errorf("grouped FP8 row %d column %d changes under BF16 conversion", row, block*32+col)
				}
			}
		}
	}
	return nil
}

// Called on the MLX worker during construction. Each group owns a disjoint
// contiguous set of output rows and their scales, not a full decoded matrix.
func newFP8OutputGroups(weight FP8AttentionWeight, groups, rank, width int) ([]*quant.FP8Linear, error) {
	if err := validateFP8OutputWeight(weight, groups, rank, width); err != nil {
		return nil, err
	}
	out := make([]*quant.FP8Linear, 0, groups)
	n, ns := rank*width, rank*width/1024
	for g := 0; g < groups; g++ {
		p, err := quant.NewFP8Linear(weight.Data[g*n:(g+1)*n], weight.Scales[g*ns:(g+1)*ns], rank, width, quant.FP8Block32)
		if err != nil {
			for _, p := range out {
				p.Close()
			}
			return nil, err
		}
		out = append(out, p)
	}
	return out, nil
}

func (session *Session) attentionOutputProjection(s *scope, o mlx.Array, name string) mlx.Array {
	if s.err != nil {
		return mlx.Array{}
	}
	c := session.model.config
	groups := session.model.fp8AttentionOutput[name]
	if len(groups) == 0 {
		return groupedAttentionProjection(s, o, session.model.weights[name], c)
	}
	n, width := o.Shape()[1], c.Heads*c.HeadDim/c.Groups
	o = s.add(mlx.Reshape(o, []int{n, c.Groups, width}))
	outputs := make([]mlx.Array, len(groups))
	for g, p := range groups {
		x := s.add(mlx.Reshape(s.span(o, 1, g, g+1), []int{n, width}))
		outputs[g] = s.add(p.Forward(x))
	}
	y := s.add(mlx.ConcatenateAxis(outputs, 1))
	return s.add(mlx.Reshape(y, []int{1, n, c.Groups * c.ORank}))
}
