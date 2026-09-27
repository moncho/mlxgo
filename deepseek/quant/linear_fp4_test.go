package quant

import (
	"errors"
	"testing"
)

func TestFP4LinearInvalid(t *testing.T) {
	for _, dims := range [][2]int{{0, 32}, {32, 0}, {-32, 32}, {31, 32}, {32, 33}, {int(^uint(0)>>1) - 31, 64}} {
		if p, err := NewFP4Linear(nil, nil, dims[0], dims[1]); p != nil || err == nil {
			t.Fatal("invalid dimensions accepted", dims, err)
		}
	}
	for _, name := range []string{"short data", "short scales", "NaN scale", "overflow"} {
		t.Run(name, func(t *testing.T) {
			d, s := make([]byte, 512), make([]byte, 32)
			for i := range s {
				s[i] = 127
			}
			switch name {
			case "short data":
				d = d[:511]
			case "short scales":
				s = s[:31]
			case "NaN scale":
				s[31] = 255
			case "overflow":
				s[31] = 254
				d[511] = 0x77
			}
			p, err := NewFP4Linear(d, s, 32, 32)
			if err == nil || p != nil {
				if p != nil {
					p.Close()
				}
				t.Fatal("invalid weights accepted")
			}
			if (name == "NaN scale" || name == "overflow") && !errors.Is(err, ErrNonFinite) {
				t.Fatal(err)
			}
		})
	}
}
