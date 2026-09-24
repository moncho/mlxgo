package quant

import (
	"bytes"
	"errors"
	"io"
	"math"
	"testing"
)

func TestReadMatrixFixtures(t *testing.T) {
	for _, c := range readFixture(t).Cases {
		t.Run(c.Name, func(t *testing.T) {
			format, rounding := options(t, c)
			got, err := ReadMatrix(bytes.NewReader(c.Data), bytes.NewReader(c.Scales), c.Rows, c.Cols, format, rounding, 1<<20)
			if err != nil {
				t.Fatal(err)
			}
			want := decodeCase(t, c)
			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("element %d differs", i)
				}
			}
		})
	}
}

type noRead struct{ t *testing.T }

func (r noRead) Read([]byte) (int, error) {
	r.t.Fatal("read before validation")
	return 0, io.EOF
}

func TestReadMatrixBudgetAndValidation(t *testing.T) {
	const exact = 32*32*4 + 32*32 + 1
	r := noRead{t}
	for _, budget := range []int64{-1, 0, exact - 1} {
		out, err := ReadMatrix(r, r, 32, 32, FP8Block32, Float32, budget)
		if out != nil || !errors.Is(err, ErrMemoryBudget) {
			t.Fatal(out, err)
		}
	}
	for _, dims := range [][2]int{{0, 32}, {32, -1}, {int(^uint(0) >> 1), 2}} {
		if _, err := ReadMatrix(r, r, dims[0], dims[1], FP8Block32, Float32, math.MaxInt64); err == nil {
			t.Fatal("accepted invalid dimensions")
		}
	}
	for _, bad := range []struct {
		f    Format
		mode Rounding
		cols int
	}{{99, Float32, 32}, {FP8Block32, 99, 32}, {FP4Row32, Float32, 31}} {
		if _, err := ReadMatrix(r, r, 32, bad.cols, bad.f, bad.mode, 1<<20); err == nil {
			t.Fatal("accepted invalid options")
		}
	}
	if _, err := ReadMatrix(bytes.NewReader(make([]byte, 1024)), bytes.NewReader([]byte{127}), 32, 32, FP8Block32, Float32, exact); err != nil {
		t.Fatal("exact budget rejected", err)
	}
}

func TestReadMatrixBadStreams(t *testing.T) {
	for _, mode := range []string{"short weight", "short scale", "long weight", "long scale", "nan", "late nan"} {
		t.Run(mode, func(t *testing.T) {
			data, scales := make([]byte, 64*32), []byte{127, 127}
			switch mode {
			case "short weight":
				data = data[:len(data)-1]
			case "short scale":
				scales = scales[:1]
			case "long weight":
				data = append(data, 0)
			case "long scale":
				scales = append(scales, 127)
			case "nan":
				data[0] = 127
			case "late nan":
				data[len(data)-1] = 127
			}
			if out, err := ReadMatrix(bytes.NewReader(data), bytes.NewReader(scales), 64, 32, FP8Block32, Float32, 1<<20); out != nil || err == nil {
				t.Fatal("returned partial/successful matrix", err)
			}
		})
	}
}
