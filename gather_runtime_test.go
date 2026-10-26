//go:build mlx && mlxruntime

package mlx

import (
	"slices"
	"testing"
)

func TestRuntimeGatherMM(t *testing.T) {
	defer SetDefaultCPU()
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", SetDefaultCPU}, {"gpu", SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			a := mustNewFloat32(t, []float32{1, 2, 3, 4, 5, 6}, []int{3, 1, 2})
			defer a.Close()
			b := mustNewFloat32(t, []float32{1, 0, 0, 1, 2, 0, 0, 3}, []int{2, 2, 2})
			defer b.Close()
			identity := mustNewFloat32(t, []float32{1, 0, 0, 1}, []int{2, 2})
			defer identity.Close()
			left := mustNewInt32(t, []int32{2, 0}, []int{2, 1})
			defer left.Close()
			right := mustNewInt32(t, []int32{1, 0}, []int{2})
			defer right.Close()
			leftOnly := mustNewInt32(t, []int32{2, 0}, []int{2})
			defer leftOnly.Close()
			rightOnly := mustNewInt32(t, []int32{0, 1, 1}, []int{3})
			defer rightOnly.Close()
			for _, test := range []struct {
				name      string
				b, li, ri Array
				sorted    bool
				shape     []int
				want      []float32
			}{
				{"both-broadcast", b, left, right, false, []int{2, 2, 1, 2}, []float32{10, 18, 5, 6, 2, 6, 1, 2}},
				{"left-only", b, leftOnly, Array{}, false, []int{2, 1, 2}, []float32{5, 6, 2, 6}},
				{"right-only-sorted", b, Array{}, rightOnly, true, []int{3, 1, 2}, []float32{1, 2, 6, 12, 10, 18}},
				{"no-indices", identity, Array{}, Array{}, false, []int{3, 1, 2}, []float32{1, 2, 3, 4, 5, 6}},
			} {
				t.Run(test.name, func(t *testing.T) {
					y, err := GatherMM(a, test.b, test.li, test.ri, test.sorted)
					if err != nil {
						t.Fatal(err)
					}
					defer y.Close()
					if !slices.Equal(y.Shape(), test.shape) {
						t.Fatalf("shape %v, want %v", y.Shape(), test.shape)
					}
					assertFloat32Data(t, y, test.want)
				})
			}
		})
	}
}

func TestRuntimeGatherMXFP4(t *testing.T) {
	defer SetDefaultCPU()
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", SetDefaultCPU}, {"gpu", SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			weights, scales := gatherFP4Fixture(t)
			defer weights.Close()
			defer scales.Close()
			data := make([]float32, 3*32)
			for i := range data {
				data[i] = float32((i/32 + 1) * (i%4 + 1))
			}
			x := mustNewFloat32(t, data, []int{3, 1, 32})
			defer x.Close()
			pair := mustNewFloat32(t, data[:64], []int{2, 1, 32})
			defer pair.Close()
			left := mustNewInt32(t, []int32{2, 0}, []int{2, 1})
			defer left.Close()
			right := mustNewInt32(t, []int32{1, 0}, []int{2})
			defer right.Close()
			rightOnly := mustNewInt32(t, []int32{0, 1, 1}, []int{3})
			defer rightOnly.Close()
			leftOnly := mustNewInt32(t, []int32{2, 0}, []int{2})
			defer leftOnly.Close()
			for _, dtype := range []DType{Float32, Float16, BFloat16} {
				t.Run(dtype.String(), func(t *testing.T) {
					input, err := AsType(x, dtype)
					if err != nil {
						t.Fatal(err)
					}
					defer input.Close()
					pairInput, err := AsType(pair, dtype)
					if err != nil {
						t.Fatal(err)
					}
					defer pairInput.Close()
					for _, test := range []struct {
						name      string
						x, li, ri Array
						sorted    bool
						shape     []int
						rows, ids []int
					}{
						{"both-broadcast", input, left, right, false, []int{2, 2, 1, 32}, []int{2, 2, 0, 0}, []int{1, 0, 1, 0}},
						{"left-only", input, leftOnly, Array{}, false, []int{2, 1, 32}, []int{2, 0}, []int{0, 1}},
						{"right-only-sorted", input, Array{}, rightOnly, true, []int{3, 1, 32}, []int{0, 1, 2}, []int{0, 1, 1}},
						{"no-indices", pairInput, Array{}, Array{}, false, []int{2, 1, 32}, []int{0, 1}, []int{0, 1}},
					} {
						t.Run(test.name, func(t *testing.T) {
							y, err := GatherMXFP4(test.x, weights, scales, test.li, test.ri, test.sorted)
							if err != nil {
								t.Fatal(err)
							}
							defer y.Close()
							if dt, err := y.DType(); err != nil || dt != dtype {
								t.Fatalf("dtype=%v err=%v, want %v", dt, err, dtype)
							}
							if !slices.Equal(y.Shape(), test.shape) {
								t.Fatalf("shape %v, want %v", y.Shape(), test.shape)
							}
							f, err := AsType(y, Float32)
							if err != nil {
								t.Fatal(err)
							}
							defer f.Close()
							want := make([]float32, len(test.rows)*32)
							for i, row := range test.rows {
								for n := range 32 {
									for k := range 32 {
										w := float32(1 + (n+test.ids[i]+k)%2)
										want[i*32+n] += data[row*32+k] * w * float32(uint32(1)<<(test.ids[i]+n%2))
									}
								}
							}
							assertFloat32Data(t, f, want)
						})
					}
				})
			}
		})
	}
}

func TestRuntimeGatherOwnershipAndValidation(t *testing.T) {
	if err := SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}
	baseline := LiveArrays()
	t.Run("packed-ownership", func(t *testing.T) {
		x := mustNewFloat32(t, makeGatherOnes(32), []int{1, 32})
		weights, scales := gatherFP4Fixture(t)
		indices := mustNewInt32(t, []int32{1}, []int{1})
		y, err := GatherMXFP4(x, weights, scales, Array{}, indices, false)
		if err != nil {
			t.Fatal(err)
		}
		defer y.Close()
		for _, a := range []Array{x, weights, scales, indices} {
			if err := a.Close(); err != nil {
				t.Fatal(err)
			}
		}
		want := make([]float32, 32)
		for i := range want {
			want[i] = 96 * float32(uint32(1)<<(i%2))
		}
		assertFloat32Data(t, y, want)
	})
	t.Run("float-ownership", func(t *testing.T) {
		x := mustNewFloat32(t, []float32{1, 2}, []int{1, 2})
		w := mustNewFloat32(t, []float32{1, 0, 0, 1, 2, 0, 0, 3}, []int{2, 2, 2})
		indices := mustNewInt32(t, []int32{1}, []int{1})
		y, err := GatherMM(x, w, Array{}, indices, false)
		if err != nil {
			t.Fatal(err)
		}
		defer y.Close()
		for _, a := range []Array{x, w, indices} {
			if err := a.Close(); err != nil {
				t.Fatal(err)
			}
		}
		assertFloat32Data(t, y, []float32{2, 6})
	})
	t.Run("invalid", func(t *testing.T) {
		x := mustNewFloat32(t, makeGatherOnes(32), []int{1, 32})
		defer x.Close()
		weights, scales := gatherFP4Fixture(t)
		defer weights.Close()
		defer scales.Close()
		badX := mustNewFloat32(t, []float32{1}, []int{1})
		defer badX.Close()
		badIndices := mustNewFloat32(t, []float32{0}, []int{1})
		defer badIndices.Close()
		badScales, err := Reshape(scales, []int{32, 2})
		if err != nil {
			t.Fatal(err)
		}
		defer badScales.Close()
		closed := mustNewInt32(t, []int32{0}, []int{1})
		closed.Close()
		for _, args := range [][5]Array{
			{Array{}, weights, scales, Array{}, Array{}},
			{badX, weights, scales, Array{}, Array{}},
			{x, weights, badScales, Array{}, Array{}},
			{x, weights, scales, badIndices, Array{}},
			{x, weights, scales, Array{}, closed},
			{x, x, scales, Array{}, Array{}},
		} {
			if y, err := GatherMXFP4(args[0], args[1], args[2], args[3], args[4], false); err == nil {
				y.Close()
				t.Fatal("invalid packed gather accepted")
			}
		}
		for _, args := range [][4]Array{
			{Array{}, x, Array{}, Array{}},
			{badX, x, Array{}, Array{}},
			{x, x, Array{}, Array{}},
			{x, weights, badIndices, Array{}},
			{x, weights, Array{}, closed},
		} {
			if y, err := GatherMM(args[0], args[1], args[2], args[3], false); err == nil {
				y.Close()
				t.Fatal("invalid gather accepted")
			}
		}
	})
	if got := LiveArrays(); got != baseline {
		t.Fatalf("live arrays %d, want %d", got, baseline)
	}
}

func gatherFP4Fixture(t *testing.T) (Array, Array) {
	t.Helper()
	data := make([]byte, 2*32*16)
	codes := make([]byte, 2*32)
	for e := range 2 {
		for n := range 32 {
			code := byte(0x42)
			if (e+n)%2 != 0 {
				code = 0x24
			}
			for k := range 16 {
				data[(e*32+n)*16+k] = code
			}
			codes[e*32+n] = byte(127 + e + n%2)
		}
	}
	bytes, err := NewUInt8(data, []int{2, 32, 16})
	if err != nil {
		t.Fatal(err)
	}
	defer bytes.Close()
	weights, err := View(bytes, UInt32)
	if err != nil {
		t.Fatal(err)
	}
	scales, err := NewUInt8(codes, []int{2, 32, 1})
	if err != nil {
		weights.Close()
		t.Fatal(err)
	}
	return weights, scales
}

func makeGatherOnes(n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = 1
	}
	return x
}
