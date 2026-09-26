//go:build mlx && mlxruntime

package mlx

import (
	"reflect"
	"testing"
)

func TestRuntimeMXFP8Matmul(t *testing.T) {
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", SetDefaultCPU}, {"gpu", SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			data := make([]byte, 32*32)
			for i := range data {
				data[i] = 0x38
			}
			bytes, err := NewUInt8(data, []int{32, 32})
			if err != nil {
				t.Fatal(err)
			}
			defer bytes.Close()
			w, err := View(bytes, UInt32)
			if err != nil {
				t.Fatal(err)
			}
			defer w.Close()
			codes := make([]byte, 32)
			for i := range codes {
				codes[i] = 127
			}
			s, err := NewUInt8(codes, []int{32, 1})
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			for _, dtype := range []DType{Float32, Float16, BFloat16} {
				x, err := Ones([]int{3, 32}, dtype)
				if err != nil {
					t.Fatal(err)
				}
				defer x.Close()
				y, err := MXFP8Matmul(x, w, s)
				if err != nil {
					t.Fatal(err)
				}
				defer y.Close()
				if dt, err := y.DType(); err != nil || dt != dtype {
					t.Fatalf("output dtype=%v err=%v", dt, err)
				}
				if !reflect.DeepEqual(y.Shape(), []int{3, 32}) {
					t.Fatal("output shape")
				}
				f, err := AsType(y, Float32)
				if err != nil {
					t.Fatal(err)
				}
				defer f.Close()
				v, err := f.Float32Data()
				if err != nil {
					t.Fatal(err)
				}
				for _, n := range v {
					if n != 32 {
						t.Fatalf("got %g want 32", n)
					}
				}
			}
			x, err := Ones([]int{1, 32}, Float32)
			if err != nil {
				t.Fatal(err)
			}
			defer x.Close()
			badRank, err := Reshape(x, []int{32})
			if err != nil {
				t.Fatal(err)
			}
			defer badRank.Close()
			badDType, err := AsType(x, Int32)
			if err != nil {
				t.Fatal(err)
			}
			defer badDType.Close()
			badWidth, err := Ones([]int{1, 64}, Float32)
			if err != nil {
				t.Fatal(err)
			}
			defer badWidth.Close()
			badScales, err := Reshape(s, []int{1, 32})
			if err != nil {
				t.Fatal(err)
			}
			defer badScales.Close()
			for _, args := range [][3]Array{{Array{}, w, s}, {badRank, w, s}, {badDType, w, s}, {badWidth, w, s}, {x, bytes, s}, {x, w, badScales}, {x, w, Array{}}} {
				if y, err := MXFP8Matmul(args[0], args[1], args[2]); err == nil {
					y.Close()
					t.Fatal("invalid argument accepted")
				}
			}
			if err := w.Close(); err != nil {
				t.Fatal(err)
			}
			if y, err := MXFP8Matmul(x, w, s); err == nil {
				y.Close()
				t.Fatal("closed weight accepted")
			}
		})
	}
}
