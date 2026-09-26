package deepseek

import (
	"strings"
	"testing"
)

func TestFP8OutputBF16Guard(t *testing.T) {
	for _, tt := range []struct {
		name        string
		code, scale byte
		wantErr     string
	}{
		{"normal", 0x38, 127, ""},
		{"negative zero", 0x80, 0, ""},
		{"exact BF16 subnormal", 0x38, 0, ""},
		{"BF16 underflow", 0x01, 0, "BF16 conversion"},
		{"BF16 rounding", 0x09, 0, "BF16 conversion"},
		{"nonfinite weight", 0x7f, 127, "nonfinite"},
		{"nonfinite scale", 0x38, 255, "nonfinite"},
		{"overflow", 0x7e, 254, "nonfinite"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			w := FP8AttentionWeight{Data: make([]byte, 32*32), Scales: []byte{tt.scale}}
			for i := range w.Data {
				w.Data[i] = tt.code
			}
			err := validateFP8OutputWeight(w, 1, 32, 32)
			if tt.wantErr == "" && err != nil || tt.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tt.wantErr)) {
				t.Fatalf("error=%v want %q", err, tt.wantErr)
			}
		})
	}
}

func TestFP8OutputLayoutValidation(t *testing.T) {
	w := FP8AttentionWeight{Data: make([]byte, 64*32), Scales: []byte{127, 127}}
	for _, dims := range [][3]int{{0, 32, 32}, {2, 16, 32}, {1, 32, 31}, {1, 64, 64}, {int(^uint(0) >> 1), 32, 32}} {
		if err := validateFP8OutputWeight(w, dims[0], dims[1], dims[2]); err == nil {
			t.Fatal("accepted invalid layout", dims)
		}
	}
	if err := validateFP8OutputWeight(w, 2, 32, 32); err != nil {
		t.Fatal(err)
	}
	w.Scales = w.Scales[:1]
	if err := validateFP8OutputWeight(w, 2, 32, 32); err == nil {
		t.Fatal("accepted truncated scales")
	}
	w.Scales = []byte{127, 0}
	w.Data[len(w.Data)-1] = 1
	if err := validateFP8OutputWeight(w, 2, 32, 32); err == nil || !strings.Contains(err.Error(), "BF16 conversion") {
		t.Fatal("did not validate last group's BF16 rounding", err)
	}
}
