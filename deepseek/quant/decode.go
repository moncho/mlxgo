// Package quant provides bounded CPU reference weight decoding, host activation
// quantization, lazy MLX activation quantization graphs, and an experimental
// packed FP8 linear adapter to MLX's native kernel. It is not a full-checkpoint loader.
// No tensor payloads are fetched here.
package quant

import (
	"errors"
	"fmt"
	"math"
)

// Format specifies both element encoding and scale layout. Scale bytes always
// use unsigned E8M0 (bias 127), not ordinary integer scale exponents.
type Format uint8

const (
	// FP8Block32 uses E4M3FN values and one scale per 32x32 block.
	// Partial edge blocks are allowed, with ceil(rows/32)*ceil(cols/32) scales.
	FP8Block32 Format = iota
	// FP8Row32 uses E4M3FN values and one scale per row per 32 columns,
	// as in the Engram table. Columns must be a multiple of 32.
	FP8Row32
	// FP4Row32 uses packed E2M1 values, low nibble first, with one scale per
	// row per 32 logical columns. Columns must be a multiple of 32.
	FP4Row32
)

type Rounding uint8

const (
	// Float32 keeps the scaled float32 result.
	Float32 Rounding = iota
	// BFloat16 rounds the scaled result to nearest-even BF16, returned as
	// float32. This models the reference wo_a conversion and Engram lookup.
	BFloat16
)

var ErrNonFinite = errors.New("quant: nonfinite decoded value")

// Decode writes one row-major matrix (or independently scaled chunk) into dst.
// Lengths must match exactly: dst=rows*cols; data=rows*cols for FP8 or half
// that for FP4. scales follows Format. Dimensions must be positive.
//
// It allocates no output buffer. All validation, including a nonfinite-output
// check, completes before dst is modified. NaN weights/scales and float32/BF16
// overflow return ErrNonFinite; finite underflow and signed zeros are preserved.
// Inputs must remain immutable and dst must not be shared during the call.
// For chunked FP8Block32 decoding, chunks must start on original 32-row/column
// block boundaries and include the corresponding scale blocks.
func Decode(dst []float32, data, scales []byte, rows, cols int, format Format, rounding Rounding) error {
	if rows <= 0 || cols <= 0 || rows > int(^uint(0)>>1)/cols {
		return fmt.Errorf("quant: invalid or overflowing dimensions %dx%d", rows, cols)
	}
	n := rows * cols
	if rounding != Float32 && rounding != BFloat16 {
		return fmt.Errorf("quant: invalid rounding %d", rounding)
	}
	blocks := (cols-1)/32 + 1
	var ndata, nscales int
	switch format {
	case FP8Block32:
		ndata, nscales = n, ((rows-1)/32+1)*blocks
	case FP8Row32, FP4Row32:
		if cols%32 != 0 {
			return fmt.Errorf("quant: row-scaled columns must be divisible by 32")
		}
		ndata, nscales = n, rows*blocks
		if format == FP4Row32 {
			ndata /= 2
		}
	default:
		return fmt.Errorf("quant: invalid format %d", format)
	}
	if len(dst) != n || len(data) != ndata || len(scales) != nscales {
		return fmt.Errorf("quant: lengths dst/data/scales=%d/%d/%d, expected %d/%d/%d", len(dst), len(data), len(scales), n, ndata, nscales)
	}
	for i, b := range scales {
		if b == 255 {
			return fmt.Errorf("%w: NaN scale at %d", ErrNonFinite, i)
		}
	}
	// Two passes keep failures atomic without allocating an output-sized scratch
	// buffer. Use bounded chunks; this is not a throughput-optimized kernel.
	for pass := 0; pass < 2; pass++ {
		for row := 0; row < rows; row++ {
			scaleRow := row
			if format == FP8Block32 {
				scaleRow /= 32
			}
			for col := 0; col < cols; col++ {
				i := row*cols + col
				var v float32
				if format == FP4Row32 {
					b := data[i/2]
					if i%2 != 0 {
						b >>= 4
					}
					v = DecodeE2M1(b)
				} else {
					v = DecodeE4M3(data[i])
				}
				v *= DecodeE8M0(scales[scaleRow*blocks+col/32])
				if rounding == BFloat16 {
					v = RoundBFloat16(v)
				}
				if pass == 0 {
					if math.Float32bits(v)&0x7f800000 == 0x7f800000 {
						return fmt.Errorf("%w at row %d column %d", ErrNonFinite, row, col)
					}
				} else {
					dst[i] = v
				}
			}
		}
	}
	return nil
}

// DecodeE4M3 converts one E4M3FN byte. Both signed zeros and subnormals are
// preserved. 0x7f and 0xff are NaNs; the format has no infinity encoding.
func DecodeE4M3(b byte) float32 {
	sign := uint32(b&0x80) << 24
	exponent, fraction := (b>>3)&15, b&7
	if exponent == 15 && fraction == 7 {
		return math.Float32frombits(sign | 0x7fc00000)
	}
	if exponent == 0 {
		return math.Float32frombits(sign | math.Float32bits(float32(fraction)/512))
	}
	return math.Float32frombits(sign | uint32(exponent+120)<<23 | uint32(fraction)<<20)
}

// DecodeE8M0 converts an unsigned scale byte. Zero encodes 2^-127, NOT zero;
// 254 encodes 2^127, and 255 encodes NaN.
func DecodeE8M0(b byte) float32 {
	if b == 0 {
		return math.Float32frombits(0x00400000)
	}
	if b == 255 {
		return math.Float32frombits(0x7fc00000)
	}
	return math.Float32frombits(uint32(b) << 23)
}

// DecodeE2M1 converts the low nibble; high bits are ignored. The sign bit is
// preserved even for zero. DeepSeek's conversion table canonicalizes -0 to +0,
// so equality with that table is numerical, not bitwise, for nibble 8.
func DecodeE2M1(b byte) float32 {
	values := [...]float32{0, .5, 1, 1.5, 2, 3, 4, 6}
	return math.Float32frombits(math.Float32bits(values[b&7]) | uint32(b&8)<<28)
}

// RoundBFloat16 rounds a float32 value to BF16 nearest-even and widens it back
// to float32. Infinities and signed zero are preserved. NaNs become quiet NaNs;
// their payload is not part of the API contract. Finite overflow becomes infinity.
func RoundBFloat16(v float32) float32 {
	b := math.Float32bits(v)
	if b&0x7fffffff > 0x7f800000 {
		return math.Float32frombits((b | 0x00400000) & 0xffff0000)
	}
	return math.Float32frombits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff0000)
}
