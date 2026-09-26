package deepseek

import (
	"slices"

	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/deepseek/quant"
)

// Preserve leading dimensions so cache appends/trims operate on encoded rows,
// never dequantize and requantize old tokens. Queries also use this for heads.
func packCache(s *scope, x mlx.Array, format quant.ActivationFormat) (mlx.Array, mlx.Array) {
	if s.err != nil {
		return mlx.Array{}, mlx.Array{}
	}
	shape := x.Shape()
	cols := shape[len(shape)-1]
	matrix := s.add(mlx.Reshape(x, []int{-1, cols}))
	data, scales, err := quant.QuantizeActivationArray(matrix, format)
	data, scales = s.add(data, err), s.add(scales, err)
	if s.err != nil {
		return mlx.Array{}, mlx.Array{}
	}
	ds, ss := slices.Clone(shape), slices.Clone(shape)
	ds[len(ds)-1], ss[len(ss)-1] = data.Shape()[1], scales.Shape()[1]
	return s.add(mlx.Reshape(data, ds)), s.add(mlx.Reshape(scales, ss))
}

func unpackCache(s *scope, data, scales mlx.Array, format quant.ActivationFormat) mlx.Array {
	if s.err != nil {
		return mlx.Array{}
	}
	shape, scaleShape := data.Shape(), scales.Shape()
	d := s.add(mlx.Reshape(data, []int{-1, shape[len(shape)-1]}))
	sc := s.add(mlx.Reshape(scales, []int{-1, scaleShape[len(scaleShape)-1]}))
	y := s.add(quant.DequantizeActivationArray(d, sc, format, quant.Float32))
	if s.err != nil {
		return mlx.Array{}
	}
	shape[len(shape)-1] = y.Shape()[1]
	return s.add(mlx.Reshape(y, shape))
}

func (session *Session) appendAttentionCache(s *scope, data, scales *mlx.Array, x mlx.Array, count int, format quant.ActivationFormat) {
	if !session.options.QuantizedCaches {
		appendCache(s, data, x, count)
		return
	}
	d, sc := packCache(s, x, format)
	appendCache(s, data, d, count)
	appendCache(s, scales, sc, count)
}

func (session *Session) readAttentionCache(s *scope, data, scales mlx.Array, format quant.ActivationFormat) mlx.Array {
	if !session.options.QuantizedCaches {
		return data
	}
	return unpackCache(s, data, scales, format)
}
