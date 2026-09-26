//go:build mlx && mlxruntime

package mlx

import (
	"math"
	"slices"
	"testing"
)

func TestRuntimeQuantizationPrimitives(t *testing.T) {
	defer SetDefaultCPU()
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", SetDefaultCPU}, {"gpu", SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			must := func(a Array, err error) Array { return mustOp(t, a, err) }
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			a := mustNewFloat32(t, []float32{1, math.Float32frombits(0x80000000), math.SmallestNonzeroFloat32, -2}, []int{2, 2})
			defer a.Close()
			transposed := must(Transpose(a))
			defer transposed.Close()
			bits := must(View(transposed, UInt32))
			defer bits.Close()
			got, err := bits.UInt32Data()
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got, []uint32{0x3f800000, 1, 0x80000000, 0xc0000000}) {
				t.Fatalf("view: %x", got)
			}
			max := must(MaxAxis(bits, 1, true))
			defer max.Close()
			got, err = max.UInt32Data()
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(max.Shape(), []int{2, 1}) || !slices.Equal(got, []uint32{0x3f800000, 0xc0000000}) {
				t.Fatalf("max: %x %v", got, max.Shape())
			}
			ints := mustNewInt32(t, []int32{3, 5, 7, 9}, []int{2, 2})
			defer ints.Close()
			counts := mustNewInt32(t, []int32{1, 2}, []int{2})
			defer counts.Close()
			for _, tc := range []struct {
				fn   func(Array, Array) (Array, error)
				want []int32
			}{
				{BitwiseAnd, []int32{1, 0, 1, 0}}, {BitwiseOr, []int32{3, 7, 7, 11}},
				{LeftShift, []int32{6, 20, 14, 36}}, {RightShift, []int32{1, 1, 3, 2}},
			} {
				y := must(tc.fn(ints, counts))
				got, err := y.Int32Data()
				y.Close()
				if err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(got, tc.want) {
					t.Fatalf("bitwise: %v != %v", got, tc.want)
				}
				bad, err := tc.fn(a, counts)
				bad.Close()
				if err == nil {
					t.Fatal("accepted floating-point bitwise input")
				}
			}
			bad, err := MaxAxis(a, 3, false)
			bad.Close()
			if err == nil {
				t.Fatal("accepted invalid axis")
			}
			bad, err = View(a, DType(-1))
			bad.Close()
			if err == nil {
				t.Fatal("accepted invalid dtype")
			}
			ints.Close()
			bad, err = View(ints, Float32)
			bad.Close()
			if err == nil {
				t.Fatal("accepted closed array")
			}
		})
	}
}
