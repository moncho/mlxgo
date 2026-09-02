package mlx

import "testing"

func TestDTypeString(t *testing.T) {
	tests := map[DType]string{
		Bool:      "bool",
		Int32:     "int32",
		Int64:     "int64",
		Float32:   "float32",
		Float64:   "float64",
		Complex64: "complex64",
		DType(99): "unknown",
	}

	for dtype, want := range tests {
		if got := dtype.String(); got != want {
			t.Fatalf("%v.String() = %q, want %q", int(dtype), got, want)
		}
	}
}
