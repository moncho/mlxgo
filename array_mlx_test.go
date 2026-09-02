//go:build mlx

package mlx

import "testing"

func TestMLXBuildValidatesShapesBeforeCallingNativeCode(t *testing.T) {
	tests := []struct {
		name  string
		shape []int
	}{
		{name: "negative", shape: []int{2, -1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := cDataShape(tt.shape); err == nil {
				t.Fatal("expected invalid shape to fail before native allocation")
			}
		})
	}
}

func TestMLXBuildAllowsScalarAndEmptyShapes(t *testing.T) {
	for _, shape := range [][]int{nil, []int{}, []int{2, 0}} {
		if _, err := cDataShape(shape); err != nil {
			t.Fatalf("cDataShape(%v) returned error: %v", shape, err)
		}
	}
}

func TestMLXBuildValidatesElementCountBeforeCallingNativeCode(t *testing.T) {
	if _, err := NewFloat32([]float32{1, 2, 3}, []int{2, 2}); err == nil {
		t.Fatal("expected mismatched data and shape to fail")
	}
}

func TestMLXBuildValidatesReshapeShape(t *testing.T) {
	tests := []struct {
		name    string
		shape   []int
		wantErr bool
	}{
		{name: "valid", shape: []int{2, -1}, wantErr: false},
		{name: "scalar", shape: nil, wantErr: false},
		{name: "zero", shape: []int{0, 2}, wantErr: false},
		{name: "two inferred", shape: []int{-1, -1}, wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := cReshapeShape(tt.shape)
			if (err != nil) != tt.wantErr {
				t.Fatalf("cReshapeShape(%v) error = %v, wantErr %v", tt.shape, err, tt.wantErr)
			}
		})
	}
}

func TestMLXBuildMapsSupportedDTypes(t *testing.T) {
	for _, dtype := range []DType{Bool, Int32, Int64, Float32, Float64} {
		if _, err := cDType(dtype); err != nil {
			t.Fatalf("cDType(%s) returned error: %v", dtype, err)
		}
	}

	if _, err := cDType(DType(99)); err == nil {
		t.Fatal("expected unsupported dtype to fail")
	}
}

func TestMLXBuildValidatesNewAPIInputsBeforeCallingNativeCode(t *testing.T) {
	if _, err := Zeros([]int{-1}, Float32); err == nil {
		t.Fatal("expected Zeros with invalid shape to fail")
	}
	if _, err := Ones([]int{1, -1}, Float32); err == nil {
		t.Fatal("expected Ones with invalid shape to fail")
	}
	if _, err := Full([]int{1, -1}, 1, Float32); err == nil {
		t.Fatal("expected Full with invalid shape to fail")
	}
	if _, err := Concatenate(nil); err == nil {
		t.Fatal("expected Concatenate with no arrays to fail")
	}
	if _, err := Stack(nil); err == nil {
		t.Fatal("expected Stack with no arrays to fail")
	}
	if _, err := Load(""); err == nil {
		t.Fatal("expected Load with empty path to fail")
	}
	if err := Save("", Array{}); err == nil {
		t.Fatal("expected Save with empty path to fail")
	}
	if _, err := LoadSafetensors(""); err == nil {
		t.Fatal("expected LoadSafetensors with empty path to fail")
	}
	if err := SaveSafetensors("", nil, nil); err == nil {
		t.Fatal("expected SaveSafetensors with empty path to fail")
	}
	if err := SaveSafetensors("weights.safetensors", nil, nil); err == nil {
		t.Fatal("expected SaveSafetensors with no arrays to fail")
	}
	if _, err := BroadcastTo(Array{}, []int{-1}); err == nil {
		t.Fatal("expected BroadcastTo with invalid shape to fail")
	}
	if _, err := ExpandDimsAxes(Array{}, nil); err == nil {
		t.Fatal("expected ExpandDimsAxes with empty axes to fail")
	}
	if _, err := RandomNormal([]int{-1}, Float32, 0, 1); err == nil {
		t.Fatal("expected RandomNormal with invalid shape to fail")
	}
	if _, err := RandomUniform([]int{2, 2}, DType(99), 0, 1); err == nil {
		t.Fatal("expected RandomUniform with invalid dtype to fail")
	}
	if _, err := RandomBernoulli([]int{-1}, 0.5); err == nil {
		t.Fatal("expected RandomBernoulli with invalid shape to fail")
	}
}
