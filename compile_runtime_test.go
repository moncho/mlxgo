//go:build mlx && mlxruntime

package mlx

import "testing"

func TestCompiledClosureOwnership(t *testing.T) {
	closureRegistry.Lock()
	before := len(closureRegistry.m)
	closureRegistry.Unlock()
	f, err := Compile(func(in []Array) ([]Array, error) {
		out, err := Square(in[0])
		if err != nil {
			return nil, err
		}
		return []Array{out}, nil
	}, true)
	if err != nil {
		t.Fatal(err)
	}
	for _, values := range [][]float32{{2}, {3, 4}} {
		x := mustNewFloat32(t, values, []int{len(values)})
		out, err := f.Apply(x)
		if err != nil {
			t.Fatal(err)
		}
		want := make([]float32, len(values))
		for i, v := range values {
			want[i] = v * v
		}
		assertFloat32Data(t, out[0], want)
		_ = CloseArrays(out)
		_ = x.Close()
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	closureRegistry.Lock()
	after := len(closureRegistry.m)
	closureRegistry.Unlock()
	if after != before {
		t.Fatalf("compiled closure leaked callback payloads: before=%d after=%d", before, after)
	}
}
