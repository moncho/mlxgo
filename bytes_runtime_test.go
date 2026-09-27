//go:build mlx && mlxruntime

package mlx

import (
	"slices"
	"testing"
)

func TestRuntimeUInt8Data(t *testing.T) {
	defer SetDefaultCPU()
	for _, set := range []func() error{SetDefaultCPU, SetDefaultGPU} {
		if err := set(); err != nil {
			t.Fatal(err)
		}
		a, err := NewUInt8([]byte{1, 2, 255, 4}, []int{2, 2})
		if err != nil {
			t.Fatal(err)
		}
		b, err := Transpose(a)
		if err != nil {
			t.Fatal(err)
		}
		got, err := b.UInt8Data()
		if err != nil || !slices.Equal(got, []byte{1, 255, 2, 4}) {
			t.Fatal(got, err)
		}
		got[0] = 99
		again, err := b.UInt8Data()
		if err != nil || again[0] != 1 {
			t.Fatal("not an owned copy", err)
		}
		b.Close()
		a.Close()
		if _, err := a.UInt8Data(); err == nil {
			t.Fatal("closed read accepted")
		}
		one, err := NewUInt8([]byte{7}, []int{1})
		if err != nil {
			t.Fatal(err)
		}
		view, err := BroadcastTo(one, []int{8})
		if err != nil {
			t.Fatal(err)
		}
		got, err = view.UInt8Data()
		if err != nil || !slices.Equal(got, []byte{7, 7, 7, 7, 7, 7, 7, 7}) {
			t.Fatal(got, err)
		}
		view.Close()
		one.Close()
		wrong, err := NewFloat32([]float32{1}, []int{1})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := wrong.UInt8Data(); err == nil {
			t.Fatal("dtype mismatch accepted")
		}
		wrong.Close()
		empty, err := NewUInt8(nil, []int{0})
		if err != nil {
			t.Fatal(err)
		}
		got, err = empty.UInt8Data()
		empty.Close()
		if err != nil || len(got) != 0 {
			t.Fatal(got, err)
		}
	}
}
