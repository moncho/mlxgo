//go:build mlx && mlxruntime

package mlx

import (
	"slices"
	"testing"
)

func TestSortAxes(t *testing.T) {
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", SetDefaultCPU}, {"gpu", SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			x := mustNewFloat32(t, []float32{3, 1, 2, 0, 5, 4}, []int{2, 3})
			defer x.Close()
			for _, test := range []struct {
				axis   int
				values []float32
				ids    []int32
			}{{-1, []float32{1, 2, 3, 0, 4, 5}, []int32{1, 2, 0, 0, 2, 1}}, {0, []float32{0, 1, 2, 3, 5, 4}, []int32{1, 0, 0, 0, 1, 1}}} {
				y, err := SortAxis(x, test.axis)
				if err != nil {
					t.Fatal(err)
				}
				defer y.Close()
				assertFloat32Data(t, y, test.values)
				idx, err := ArgSortAxis(x, test.axis)
				if err != nil {
					t.Fatal(err)
				}
				defer idx.Close()
				cast, err := AsType(idx, Int32)
				if err != nil {
					t.Fatal(err)
				}
				defer cast.Close()
				got, err := cast.Int32Data()
				if err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(got, test.ids) {
					t.Fatalf("sort indices %v, want %v", got, test.ids)
				}
			}
			if a, err := SortAxis(x, 3); err == nil {
				a.Close()
				t.Fatal("accepted invalid axis")
			}
		})
	}
}
