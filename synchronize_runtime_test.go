//go:build mlx && mlxruntime

package mlx

import "testing"

func TestRuntimeSynchronize(t *testing.T) {
	for _, device := range []DeviceType{DeviceCPU, DeviceGPU} {
		if err := SetDefaultDevice(device, 0); err != nil {
			t.Fatal(err)
		}
		if err := Synchronize(); err != nil {
			t.Fatal(err)
		}
		x, err := Ones([]int{128}, Float32)
		if err != nil {
			t.Fatal(err)
		}
		y, err := Add(x, x)
		x.Close()
		if err != nil {
			t.Fatal(err)
		}
		if err := AsyncEval(y); err != nil {
			y.Close()
			t.Fatal(err)
		}
		if err := Synchronize(); err != nil {
			y.Close()
			t.Fatal(err)
		}
		want := make([]float32, 128)
		for i := range want {
			want[i] = 2
		}
		assertFloat32Data(t, y, want)
		y.Close()
		if err := Synchronize(); err != nil {
			t.Fatal(err)
		}
	}
}
