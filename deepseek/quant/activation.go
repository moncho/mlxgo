package quant

import (
	"fmt"
	"math"
)

// ActivationFormat describes a row-group quantizer, including its scale rule.
// These formats, shared by the CPU and MLX graph APIs, are distinct from
// checkpoint weight formats.
type ActivationFormat uint8

const (
	// FP8Activation32 uses E4M3FN values and E8M0 scales for groups of 32.
	// The absolute maximum is floored at 1e-4 before ceil-power-of-two scaling.
	FP8Activation32 ActivationFormat = iota
	// FP4Index32 uses packed E2M1 values and E8M0 scales for groups of 32.
	// The absolute maximum is floored at 6*2^-126 before power-of-two scaling.
	FP4Index32
	// FP4Cache16 uses packed E2M1 values and E4M3FN scales for groups of 16.
	// The absolute maximum is floored at 6*2^-9; scale rounding is nearest-even,
	// not ceiling to a power of two. Finite scale overflow saturates at 448.
	FP4Cache16
)

// ActivationLayout returns exact byte lengths for row-major encoded values and
// scales. Rows must be positive; columns must be divisible by the format's group
// size. FP4 packs the earlier element into the low nibble.
func ActivationLayout(rows, cols int, format ActivationFormat) (dataBytes, scaleBytes int, err error) {
	if rows <= 0 || cols <= 0 || rows > int(^uint(0)>>1)/cols {
		return 0, 0, fmt.Errorf("quant: invalid or overflowing activation dimensions %dx%d", rows, cols)
	}
	group := 32
	switch format {
	case FP8Activation32, FP4Index32:
	case FP4Cache16:
		group = 16
	default:
		return 0, 0, fmt.Errorf("quant: invalid activation format %d", format)
	}
	if cols%group != 0 {
		return 0, 0, fmt.Errorf("quant: activation columns must be divisible by %d", group)
	}
	n := rows * cols
	dataBytes = n
	if format != FP8Activation32 {
		dataBytes /= 2
	}
	return dataBytes, n / group, nil
}

// QuantizeActivation writes packed activation values and scales to caller-owned
// buffers. It models the pinned kernel's float32 scale arithmetic and saturating
// nearest-even casts, not CUDA execution or MLX graph operations. Inputs are used
// as supplied: callers modeling BF16 inputs must round them before this call.
//
// Lengths must match ActivationLayout exactly. No buffers are allocated. Errors
// leave both outputs unchanged, including late nonfinite inputs or float32
// reconstruction overflow. Signed zero is preserved. Inputs must stay immutable,
// and output buffers must not overlap or be shared during a call.
func QuantizeActivation(data, scales []byte, src []float32, rows, cols int, format ActivationFormat) error {
	nd, ns, err := ActivationLayout(rows, cols, format)
	if err != nil {
		return err
	}
	if len(src) != rows*cols || len(data) != nd || len(scales) != ns {
		return fmt.Errorf("quant: activation buffer lengths do not match layout")
	}
	group := 32
	if format == FP4Cache16 {
		group = 16
	}
	for pass := 0; pass < 2; pass++ {
		for g := 0; g < ns; g++ {
			start := g * group
			var amax float32
			for _, v := range src[start : start+group] {
				bits := math.Float32bits(v) & 0x7fffffff
				if bits >= 0x7f800000 {
					return fmt.Errorf("%w: activation group %d", ErrNonFinite, g)
				}
				amax = max(amax, math.Float32frombits(bits))
			}
			var scale float32
			var scaleCode byte
			if format == FP4Cache16 {
				scaleCode = nearestE4M3(max(amax, float32(6.0/512)) / 6)
				scale = DecodeE4M3(scaleCode)
			} else {
				var unrounded float32
				if format == FP8Activation32 {
					unrounded = max(amax, 1e-4) * float32(1.0/448)
				} else {
					unrounded = max(amax, float32(6*0x1p-126)) * float32(1.0/6)
				}
				bits := math.Float32bits(unrounded)
				exp := bits >> 23
				if bits&0x7fffff != 0 {
					exp++
				}
				scaleCode = byte(exp)
				scale = DecodeE8M0(scaleCode)
			}
			var codes [32]byte
			for j, v := range src[start : start+group] {
				var reconstructed float32
				if format == FP8Activation32 {
					codes[j] = nearestE4M3(v / scale)
					reconstructed = DecodeE4M3(codes[j]) * scale
				} else {
					codes[j] = nearestE2M1(v / scale)
					reconstructed = DecodeE2M1(codes[j]) * scale
				}
				if math.Float32bits(reconstructed)&0x7f800000 == 0x7f800000 {
					return fmt.Errorf("%w: activation reconstruction group %d", ErrNonFinite, g)
				}
			}
			if pass == 1 {
				scales[g] = scaleCode
				if format == FP8Activation32 {
					copy(data[start:start+group], codes[:group])
				} else {
					for j := 0; j < group; j += 2 {
						data[(start+j)/2] = codes[j] | codes[j+1]<<4
					}
				}
			}
		}
	}
	return nil
}

// DequantizeActivation reconstructs a row-major float32 buffer, optionally
// rounding each result to BF16. Invalid lengths, nonpositive/nonfinite scales,
// NaN codes and reconstruction overflow leave dst unchanged. No buffers are
// allocated. It does not quantize inputs or attach an MLX computation graph.
func DequantizeActivation(dst []float32, data, scales []byte, rows, cols int, format ActivationFormat, rounding Rounding) error {
	nd, ns, err := ActivationLayout(rows, cols, format)
	if err != nil {
		return err
	}
	if len(dst) != rows*cols || len(data) != nd || len(scales) != ns {
		return fmt.Errorf("quant: activation buffer lengths do not match layout")
	}
	if rounding != Float32 && rounding != BFloat16 {
		return fmt.Errorf("quant: invalid rounding %d", rounding)
	}
	group := 32
	if format == FP4Cache16 {
		group = 16
	}
	for pass := 0; pass < 2; pass++ {
		for g, code := range scales {
			scale := DecodeE8M0(code)
			if format == FP4Cache16 {
				scale = DecodeE4M3(code)
			}
			if scale <= 0 || math.Float32bits(scale)&0x7f800000 == 0x7f800000 {
				return fmt.Errorf("%w: invalid activation scale %d", ErrNonFinite, g)
			}
			for j := 0; j < group; j++ {
				i := g*group + j
				var v float32
				if format == FP8Activation32 {
					v = DecodeE4M3(data[i])
				} else {
					b := data[i/2]
					if i%2 != 0 {
						b >>= 4
					}
					v = DecodeE2M1(b)
				}
				v *= scale
				if rounding == BFloat16 {
					v = RoundBFloat16(v)
				}
				if math.Float32bits(v)&0x7f800000 == 0x7f800000 {
					return fmt.Errorf("%w: activation output %d", ErrNonFinite, i)
				}
				if pass == 1 {
					dst[i] = v
				}
			}
		}
	}
	return nil
}

// Positive encodings are ordered. Compare exact representable midpoints, using
// the low code bit for ties to even; finite overflow saturates at the last code.
func nearestCode(v float32, last int, decode func(byte) float32) byte {
	lo, hi := 0, last
	for lo < hi {
		mid := (lo + hi) / 2
		if decode(byte(mid)) < v {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	if lo == 0 {
		return 0
	}
	midpoint := (decode(byte(lo-1)) + decode(byte(lo))) / 2
	if v < midpoint || (v == midpoint && (lo-1)%2 == 0) {
		lo--
	}
	return byte(lo)
}

func nearestE4M3(v float32) byte {
	bits := math.Float32bits(v)
	return byte(bits>>24)&0x80 | nearestCode(math.Float32frombits(bits&0x7fffffff), 126, DecodeE4M3)
}

func nearestE2M1(v float32) byte {
	bits := math.Float32bits(v)
	return byte(bits>>28)&8 | nearestCode(math.Float32frombits(bits&0x7fffffff), 7, DecodeE2M1)
}
