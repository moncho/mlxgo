package quant

import (
	"errors"
	"testing"
)

func TestFP8LinearRejectsInvalidWeights(t *testing.T) {
	for _, tc := range []struct {
		name         string
		rows, cols   int
		format       Format
		data, scales []byte
		nonfinite    bool
	}{
		{"zero", 0, 32, FP8Block32, nil, nil, false},
		{"negative", -32, 32, FP8Block32, nil, nil, false},
		{"partial row", 31, 32, FP8Block32, nil, nil, false},
		{"partial group", 32, 33, FP8Block32, nil, nil, false},
		{"overflow", int(^uint(0)>>1) - 31, 64, FP8Block32, nil, nil, false},
		{"FP4", 32, 32, FP4Row32, make([]byte, 512), make([]byte, 32), false},
		{"unknown format", 32, 32, Format(255), nil, nil, false},
		{"short data", 32, 32, FP8Block32, make([]byte, 1023), []byte{127}, false},
		{"short scales", 32, 32, FP8Row32, make([]byte, 1024), []byte{127}, false},
		{"NaN scale", 32, 32, FP8Block32, make([]byte, 1024), []byte{255}, true},
		{"NaN weight", 32, 32, FP8Block32, append([]byte{127}, make([]byte, 1023)...), []byte{127}, true},
		{"weight overflow", 32, 32, FP8Block32, append([]byte{126}, make([]byte, 1023)...), []byte{254}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			l, err := NewFP8Linear(tc.data, tc.scales, tc.rows, tc.cols, tc.format)
			if err == nil || l != nil {
				if l != nil {
					l.Close()
				}
				t.Fatal("invalid weight accepted")
			}
			if tc.nonfinite && !errors.Is(err, ErrNonFinite) {
				t.Fatalf("expected ErrNonFinite: %v", err)
			}
		})
	}
}
