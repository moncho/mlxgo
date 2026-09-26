package quant

import (
	"fmt"
	"math"

	mlx "github.com/moncho/mlxgo"
)

// QuantizeActivationArray builds a lazy, inference-only MLX quantization graph.
// x must be a float32 matrix [rows, cols] with the layout accepted by
// ActivationLayout. It returns caller-owned uint8 arrays: data [rows, cols]
// (or [rows, cols/2] for FP4), and scales [rows, cols/group]. FP4 packs the
// earlier value in the low nibble. Input values are not implicitly BF16-rounded.
//
// No tensor values are read back to Go. Shape/type failures return errors now;
// nonfinite inputs or reconstruction overflow mark the affected group's scale
// as 255, which DequantizeActivationArray reconstructs as NaNs. This differs
// from the eager CPU reference's ErrNonFinite contract. Gradients are stopped.
// This is a composed reference graph, not a fused kernel or quantized matmul.
func QuantizeActivationArray(x mlx.Array, format ActivationFormat) (data, scales mlx.Array, err error) {
	out, err := activationGraph(func(s *activationScope) []mlx.Array {
		if !s.dtype(x, mlx.Float32) {
			return nil
		}
		shape := x.Shape()
		if len(shape) != 2 {
			s.err = fmt.Errorf("quant: activation input must be a matrix")
			return nil
		}
		rows, cols := shape[0], shape[1]
		_, ns, err := ActivationLayout(rows, cols, format)
		if err != nil {
			s.err = err
			return nil
		}
		group := rows * cols / ns
		x = s.add(mlx.StopGradient(x))
		x = s.add(mlx.Reshape(x, []int{ns, group}))
		bits := s.add(mlx.View(x, mlx.UInt32))
		absBits := s.add(mlx.BitwiseAnd(bits, s.u(0x7fffffff)))
		maxBits := s.add(mlx.MaxAxis(absBits, 1, true))
		amax := s.add(mlx.View(maxBits, mlx.Float32))
		var scaleCode mlx.Array
		if format == FP4Cache16 {
			amax = s.add(mlx.Maximum(amax, s.f(6.0/512)))
			unrounded := s.add(mlx.Divide(amax, s.f(6)))
			scaleCode = s.nearest(unrounded, false)
		} else {
			floor, reciprocal := float32(1e-4), float32(1.0/448)
			if format == FP4Index32 {
				floor, reciprocal = 6*0x1p-126, 1.0/6
			}
			amaxBits := s.add(mlx.Maximum(maxBits, s.u(math.Float32bits(floor))))
			scaleCode = s.powerScale(amaxBits, reciprocal)
			scaleCode = s.add(mlx.Minimum(scaleCode, s.u(255)))
		}
		scale := s.scale(scaleCode, format)
		abs := s.add(mlx.View(absBits, mlx.Float32))
		normalized := s.add(mlx.Divide(abs, scale))
		if format != FP4Cache16 {
			// Backend arithmetic may flush subnormal operands to zero. Convert
			// their integer significands instead; very large scales can safely
			// be clamped here because the result still rounds to a zero code.
			exp := s.add(mlx.Subtract(s.u(105), s.add(mlx.Minimum(scaleCode, s.u(100)))))
			factor := s.add(mlx.View(s.add(mlx.LeftShift(exp, s.u(23))), mlx.Float32))
			small := s.add(mlx.Multiply(s.add(mlx.AsType(absBits, mlx.Float32)), factor))
			normalized = s.add(mlx.Where(s.add(mlx.Less(absBits, s.u(0x800000))), small, normalized))
		}
		codes := s.nearest(normalized, format != FP8Activation32)
		signShift := uint32(24)
		if format != FP8Activation32 {
			signShift = 28
		}
		sign := s.add(mlx.RightShift(s.add(mlx.BitwiseAnd(bits, s.u(0x80000000))), s.u(signShift)))
		codes = s.add(mlx.BitwiseOr(codes, sign))
		reconstructed := s.reconstruct(codes, scaleCode, format)
		reconBits := s.add(mlx.BitwiseAnd(s.add(mlx.View(reconstructed, mlx.UInt32)), s.u(0x7fffffff)))
		maxRecon := s.add(mlx.MaxAxis(reconBits, 1, true))
		invalid := s.add(mlx.GreaterEqual(s.add(mlx.Maximum(maxBits, maxRecon)), s.u(0x7f800000)))
		scaleCode = s.add(mlx.Where(invalid, s.u(255), scaleCode))
		if format != FP8Activation32 {
			pairs := s.add(mlx.Reshape(codes, []int{rows, cols / 2, 2}))
			low := s.add(mlx.TakeAxis(pairs, s.u(0), 2))
			high := s.add(mlx.TakeAxis(pairs, s.u(1), 2))
			codes = s.add(mlx.BitwiseOr(low, s.add(mlx.LeftShift(high, s.u(4)))))
			cols /= 2
		}
		codes = s.add(mlx.Reshape(codes, []int{rows, cols}))
		scaleCode = s.add(mlx.Reshape(scaleCode, []int{rows, ns / rows}))
		return []mlx.Array{s.add(mlx.AsType(codes, mlx.UInt8)), s.add(mlx.AsType(scaleCode, mlx.UInt8))}
	})
	if err != nil {
		return mlx.Array{}, mlx.Array{}, err
	}
	return out[0], out[1], nil
}

// DequantizeActivationArray reconstructs packed device arrays as a caller-owned
// float32 matrix. BFloat16 rounds the result to BF16 but returns float32 storage.
// data and scales must have the uint8 matrix shapes produced by
// QuantizeActivationArray. Invalid scales, NaN value codes and float32 overflow
// produce NaNs, not a synchronous Go error. No tensor values leave the device.
// Gradients are stopped; this API does not implement a straight-through estimator.
func DequantizeActivationArray(data, scales mlx.Array, format ActivationFormat, rounding Rounding) (mlx.Array, error) {
	out, err := activationGraph(func(s *activationScope) []mlx.Array {
		if !s.dtype(data, mlx.UInt8) || !s.dtype(scales, mlx.UInt8) {
			return nil
		}
		ds, ss := data.Shape(), scales.Shape()
		if len(ds) != 2 || len(ss) != 2 {
			s.err = fmt.Errorf("quant: encoded data and scales must be matrices")
			return nil
		}
		rows, cols := ds[0], ds[1]
		if format != FP8Activation32 {
			if cols > int(^uint(0)>>1)/2 {
				s.err = fmt.Errorf("quant: encoded columns overflow")
				return nil
			}
			cols *= 2
		}
		_, ns, err := ActivationLayout(rows, cols, format)
		if err != nil {
			s.err = err
			return nil
		}
		if ss[0] != rows || ss[1] != ns/rows || (rounding != Float32 && rounding != BFloat16) {
			s.err = fmt.Errorf("quant: invalid scale shape or rounding")
			return nil
		}
		codes := s.add(mlx.AsType(data, mlx.UInt32))
		if format != FP8Activation32 {
			low := s.add(mlx.BitwiseAnd(codes, s.u(15)))
			high := s.add(mlx.RightShift(codes, s.u(4)))
			codes = s.add(mlx.StackAxis([]mlx.Array{low, high}, -1))
		}
		codes = s.add(mlx.Reshape(codes, []int{ns, rows * cols / ns}))
		scaleCodes := s.add(mlx.Reshape(scales, []int{ns, 1}))
		y := s.reconstruct(codes, scaleCodes, format)
		bits := s.add(mlx.View(y, mlx.UInt32))
		invalidInput := s.add(mlx.GreaterEqual(s.add(mlx.BitwiseAnd(bits, s.u(0x7fffffff))), s.u(0x7f800000)))
		bits = s.add(mlx.Where(invalidInput, s.u(0x7fc00000), bits))
		if rounding == BFloat16 {
			odd := s.add(mlx.BitwiseAnd(s.add(mlx.RightShift(bits, s.u(16))), s.u(1)))
			bits = s.add(mlx.Add(bits, s.add(mlx.Add(s.u(0x7fff), odd))))
			bits = s.add(mlx.BitwiseAnd(bits, s.u(0xffff0000)))
		}
		invalid := s.add(mlx.GreaterEqual(s.add(mlx.BitwiseAnd(bits, s.u(0x7fffffff))), s.u(0x7f800000)))
		bits = s.add(mlx.Where(invalid, s.u(0x7fc00000), bits))
		y = s.add(mlx.View(bits, mlx.Float32))
		y = s.add(mlx.Reshape(y, []int{rows, cols}))
		return []mlx.Array{s.add(mlx.StopGradient(y))}
	})
	if err != nil {
		return mlx.Array{}, err
	}
	return out[0], nil
}

type activationScope struct {
	arrays []mlx.Array
	err    error
}

func (s *activationScope) add(a mlx.Array, err error) mlx.Array {
	if err != nil {
		if s.err == nil {
			s.err = err
		}
		return mlx.Array{}
	}
	s.arrays = append(s.arrays, a)
	return a
}

func (s *activationScope) dtype(a mlx.Array, want mlx.DType) bool {
	d, err := a.DType()
	if err != nil {
		s.err = err
	} else if d != want {
		s.err = fmt.Errorf("quant: expected %s, got %s", want, d)
	}
	return s.err == nil
}

func (s *activationScope) u(v uint32) mlx.Array {
	a := s.add(mlx.NewScalarInt(int(v)))
	return s.add(mlx.AsType(a, mlx.UInt32))
}

func (s *activationScope) f(v float32) mlx.Array { return s.add(mlx.NewScalarFloat32(v)) }

// Compute ceil_pow2(float32(amax * reciprocal)) with an exact 48-bit
// significand product and round-to-nearest-even. Keeping this in integers
// prevents compiled fast-math from changing the scale at power boundaries.
func (s *activationScope) powerScale(bits mlx.Array, reciprocal float32) mlx.Array {
	r := math.Float32bits(reciprocal)
	man := s.add(mlx.BitwiseOr(s.add(mlx.BitwiseAnd(bits, s.u(0x7fffff))), s.u(0x800000)))
	man64 := s.add(mlx.AsType(man, mlx.UInt64))
	r64 := s.add(mlx.AsType(s.u(r&0x7fffff|0x800000), mlx.UInt64))
	product := s.add(mlx.Multiply(man64, r64))
	hi := s.add(mlx.AsType(s.add(mlx.RightShift(product, s.u(47))), mlx.UInt32))
	shift := s.add(mlx.Add(s.u(23), hi))
	kept := s.add(mlx.AsType(s.add(mlx.RightShift(product, shift)), mlx.UInt32))
	mask := s.add(mlx.Subtract(s.add(mlx.LeftShift(s.u(1), shift)), s.u(1)))
	rem := s.add(mlx.AsType(s.add(mlx.BitwiseAnd(product, s.add(mlx.AsType(mask, mlx.UInt64)))), mlx.UInt32))
	half := s.add(mlx.LeftShift(s.u(1), s.add(mlx.Subtract(shift, s.u(1)))))
	odd := s.add(mlx.BitwiseAnd(kept, s.u(1)))
	round := s.add(mlx.Where(s.add(mlx.Equal(rem, half)), odd, s.add(mlx.AsType(s.add(mlx.Greater(rem, half)), mlx.UInt32))))
	kept = s.add(mlx.Add(kept, round))
	carry := s.add(mlx.RightShift(kept, s.u(24)))
	ceil := s.add(mlx.AsType(s.add(mlx.Greater(s.add(mlx.BitwiseAnd(kept, s.u(0x7fffff))), s.u(0))), mlx.UInt32))
	exp := s.add(mlx.RightShift(bits, s.u(23)))
	exp = s.add(mlx.Add(exp, s.u(r>>23)))
	exp = s.add(mlx.Subtract(exp, s.u(127)))
	return s.add(mlx.Add(s.add(mlx.Add(exp, hi)), s.add(mlx.Add(carry, ceil))))
}

// Binary search of exact midpoint thresholds avoids a [values, 127] temporary.
// A one-ULP threshold adjustment gives ties to the even code, including zero.
func (s *activationScope) nearest(x mlx.Array, fp4 bool) mlx.Array {
	last, steps, decode := 126, 7, DecodeE4M3
	if fp4 {
		last, steps, decode = 7, 3, DecodeE2M1
	}
	thresholds := make([]int32, last+1)
	for i := 0; i < last; i++ {
		bits := math.Float32bits((decode(byte(i)) + decode(byte(i+1))) / 2)
		if i%2 == 0 {
			bits++
		}
		thresholds[i] = int32(bits)
	}
	thresholds[last] = 0x7fffffff
	table := s.add(mlx.NewInt32(thresholds, []int{last + 1}))
	table = s.add(mlx.AsType(table, mlx.UInt32))
	bits := s.add(mlx.View(x, mlx.UInt32))
	lo, hi := s.u(0), s.u(uint32(last))
	for range steps {
		mid := s.add(mlx.RightShift(s.add(mlx.Add(lo, hi)), s.u(1)))
		threshold := s.add(mlx.Take(table, mid))
		up := s.add(mlx.GreaterEqual(bits, threshold))
		lo = s.add(mlx.Where(up, s.add(mlx.Add(mid, s.u(1))), lo))
		hi = s.add(mlx.Where(up, hi, mid))
	}
	return lo
}

func (s *activationScope) values(codes mlx.Array, format ActivationFormat) mlx.Array {
	n, decode := 256, DecodeE4M3
	if format != FP8Activation32 {
		n, decode = 16, DecodeE2M1
	}
	return s.lookup(codes, n, decode)
}

func (s *activationScope) reconstruct(codes, scaleCodes mlx.Array, format ActivationFormat) mlx.Array {
	values := s.values(codes, format)
	if format == FP4Cache16 {
		return s.add(mlx.Multiply(values, s.scale(scaleCodes, format)))
	}
	// E8M0 multiplication is an exact exponent adjustment. All significands
	// have enough trailing zero bits that even the smallest supported scale
	// produces exact float32 subnormals. Avoid backend flush-to-zero behavior.
	bits := s.add(mlx.View(values, mlx.UInt32))
	abs := s.add(mlx.BitwiseAnd(bits, s.u(0x7fffffff)))
	exp := s.add(mlx.AsType(s.add(mlx.RightShift(abs, s.u(23))), mlx.Int32))
	exp = s.add(mlx.Add(exp, s.add(mlx.AsType(scaleCodes, mlx.Int32))))
	exp = s.add(mlx.Subtract(exp, s.add(mlx.NewScalarInt(127))))
	man := s.add(mlx.BitwiseAnd(abs, s.u(0x7fffff)))
	uExp := s.add(mlx.AsType(exp, mlx.UInt32))
	normal := s.add(mlx.BitwiseOr(man, s.add(mlx.LeftShift(uExp, s.u(23)))))
	shift := s.add(mlx.Subtract(s.add(mlx.NewScalarInt(1)), exp))
	shift = s.add(mlx.Clip(shift, s.add(mlx.NewScalarInt(0)), s.add(mlx.NewScalarInt(31))))
	shift = s.add(mlx.AsType(shift, mlx.UInt32))
	small := s.add(mlx.RightShift(s.add(mlx.BitwiseOr(man, s.u(0x800000))), shift))
	result := s.add(mlx.Where(s.add(mlx.LessEqual(exp, s.add(mlx.NewScalarInt(0)))), small, normal))
	result = s.add(mlx.Where(s.add(mlx.Equal(abs, s.u(0))), s.u(0), result))
	result = s.add(mlx.BitwiseOr(result, s.add(mlx.BitwiseAnd(bits, s.u(0x80000000)))))
	badScale := s.add(mlx.Equal(scaleCodes, s.u(255)))
	badValue := s.add(mlx.GreaterEqual(abs, s.u(0x7f800000)))
	overflow := s.add(mlx.GreaterEqual(exp, s.add(mlx.NewScalarInt(255))))
	invalid := s.add(mlx.Where(badScale, badScale, s.add(mlx.Where(badValue, badValue, overflow))))
	result = s.add(mlx.Where(invalid, s.u(0x7fc00000), result))
	return s.add(mlx.View(result, mlx.Float32))
}

func (s *activationScope) scale(codes mlx.Array, format ActivationFormat) mlx.Array {
	return s.lookup(codes, 256, func(b byte) float32 {
		v := DecodeE8M0(b)
		if format == FP4Cache16 {
			v = DecodeE4M3(b)
			if v <= 0 {
				return float32(math.NaN())
			}
		}
		return v
	})
}

func (s *activationScope) lookup(codes mlx.Array, n int, decode func(byte) float32) mlx.Array {
	table := make([]int32, n)
	for i := range table {
		table[i] = int32(math.Float32bits(decode(byte(i))))
	}
	a := s.add(mlx.NewInt32(table, []int{n}))
	a = s.add(mlx.Take(a, codes))
	return s.add(mlx.View(a, mlx.Float32))
}

func activationGraph(fn func(*activationScope) []mlx.Array) (out []mlx.Array, err error) {
	err = mlx.Batch(func() error {
		s := &activationScope{}
		defer func() { mlx.CloseArrays(s.arrays) }()
		results := fn(s)
		if s.err != nil {
			return s.err
		}
		for _, a := range results {
			copy, err := mlx.Reshape(a, a.Shape())
			if err != nil {
				mlx.CloseArrays(out)
				out = nil
				return err
			}
			out = append(out, copy)
		}
		return nil
	})
	return out, err
}
