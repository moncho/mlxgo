package quant

import (
	"compress/gzip"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"hash"
	"math"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"
)

type matrixCase struct {
	Name       string   `json:"name"`
	Format     string   `json:"format"`
	Rows       int      `json:"rows"`
	Cols       int      `json:"cols"`
	Rounding   string   `json:"rounding"`
	Data       []byte   `json:"data"`
	Scales     []byte   `json:"scales"`
	Expected   []uint32 `json:"expected_bits"`
	Input      []uint32 `json:"input_bits"`
	Projection []uint32 `json:"projection_bits"`
}

type fixture struct {
	Torch           string       `json:"torch_version"`
	Revision        string       `json:"source_revision"`
	ConvertSHA      string       `json:"conversion_sha256"`
	FP8             []uint32     `json:"fp8_bits"`
	Scales          []uint32     `json:"scale_bits"`
	FP4             []uint32     `json:"fp4_bits"`
	FP8Scaled       string       `json:"fp8_scaled_sha256"`
	FP8ScaledBF16   string       `json:"fp8_scaled_bf16_sha256"`
	FP4Scaled       string       `json:"fp4_scaled_sha256"`
	FP4ScaledBF16   string       `json:"fp4_scaled_bf16_sha256"`
	BF16Ties        string       `json:"bf16_ties_sha256"`
	RoundingInputs  []uint32     `json:"rounding_inputs"`
	RoundingOutputs []uint32     `json:"rounding_outputs"`
	Cases           []matrixCase `json:"cases"`
}

func readFixture(t testing.TB) fixture {
	t.Helper()
	f, err := os.Open("testdata/decoding.json.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	z, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	defer z.Close()
	var out fixture
	if err = json.NewDecoder(z).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if strings.SplitN(out.Torch, "+", 2)[0] != "2.14.0" || out.Revision != "df42c109f1defefcbfcedbe7d905718a12266e40" || out.ConvertSHA != "035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76" {
		t.Fatal("unexpected reference provenance")
	}
	return out
}

func sameBits(got float32, want uint32) bool {
	if want&0x7fffffff > 0x7f800000 {
		return math.IsNaN(float64(got))
	}
	return math.Float32bits(got) == want
}

func TestEveryEncoding(t *testing.T) {
	f := readFixture(t)
	for _, tc := range []struct {
		name   string
		want   []uint32
		decode func(byte) float32
		n      int
	}{
		{"E4M3", f.FP8, DecodeE4M3, 256}, {"E8M0", f.Scales, DecodeE8M0, 256}, {"E2M1", f.FP4, DecodeE2M1, 16},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if len(tc.want) != tc.n {
				t.Fatal("incomplete fixture")
			}
			for b, want := range tc.want {
				if got := tc.decode(byte(b)); !sameBits(got, want) {
					t.Fatalf("code %02x: got %08x want %08x", b, math.Float32bits(got), want)
				}
			}
		})
	}
	if DecodeE8M0(0) == 0 || !math.Signbit(float64(DecodeE2M1(8))) {
		t.Fatal("zero semantics changed")
	}
}

func hashValue(h hash.Hash, v float32) {
	b := math.Float32bits(v)
	if b&0x7fffffff > 0x7f800000 {
		b = 0x7fc00000
	}
	var bytes [4]byte
	binary.LittleEndian.PutUint32(bytes[:], b)
	h.Write(bytes[:])
}

func TestEveryCodeScalePair(t *testing.T) {
	f := readFixture(t)
	for _, tc := range []struct {
		name      string
		decode    func(byte) float32
		n         int
		f32, bf16 string
	}{
		{"FP8", DecodeE4M3, 256, f.FP8Scaled, f.FP8ScaledBF16},
		{"FP4", DecodeE2M1, 16, f.FP4Scaled, f.FP4ScaledBF16},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h, hb := sha256.New(), sha256.New()
			for s := 0; s < 256; s++ {
				for b := 0; b < tc.n; b++ {
					v := tc.decode(byte(b)) * DecodeE8M0(byte(s))
					hashValue(h, v)
					hashValue(hb, RoundBFloat16(v))
				}
			}
			if hex.EncodeToString(h.Sum(nil)) != tc.f32 || hex.EncodeToString(hb.Sum(nil)) != tc.bf16 {
				t.Fatal("scaled decode differs from PyTorch CPU across exhaustive code/scale pairs")
			}
		})
	}
}

func TestBFloat16Rounding(t *testing.T) {
	f := readFixture(t)
	if len(f.RoundingInputs) != len(f.RoundingOutputs) {
		t.Fatal("incomplete rounding fixture")
	}
	for i, b := range f.RoundingInputs {
		if got := RoundBFloat16(math.Float32frombits(b)); !sameBits(got, f.RoundingOutputs[i]) {
			t.Fatalf("round %08x: got %08x want %08x", b, math.Float32bits(got), f.RoundingOutputs[i])
		}
	}
	h := sha256.New()
	for upper := uint32(0); upper < 65536; upper++ {
		for _, low := range []uint32{0x7fff, 0x8000, 0x8001} {
			hashValue(h, RoundBFloat16(math.Float32frombits(upper<<16|low)))
		}
	}
	if hex.EncodeToString(h.Sum(nil)) != f.BF16Ties {
		t.Fatal("exhaustive BF16 tie rounding differs from PyTorch CPU")
	}
}

func options(t testing.TB, c matrixCase) (Format, Rounding) {
	t.Helper()
	f, ok := map[string]Format{"fp8_block32": FP8Block32, "fp8_row32": FP8Row32, "fp4_row32": FP4Row32}[c.Format]
	if !ok {
		t.Fatal("unknown fixture format")
	}
	r, ok := map[string]Rounding{"float32": Float32, "bfloat16": BFloat16}[c.Rounding]
	if !ok {
		t.Fatal("unknown fixture rounding")
	}
	return f, r
}

func decodeCase(t testing.TB, c matrixCase) []float32 {
	t.Helper()
	f, r := options(t, c)
	dst := make([]float32, c.Rows*c.Cols)
	if err := Decode(dst, c.Data, c.Scales, c.Rows, c.Cols, f, r); err != nil {
		t.Fatal(err)
	}
	if len(dst) != len(c.Expected) {
		t.Fatal("incomplete matrix fixture")
	}
	for i, want := range c.Expected {
		// The pinned FP4_TABLE canonicalizes negative zero. All nonzero
		// finite values and every FP8 result still require bitwise equality.
		if f == FP4Row32 && want&0x7fffffff == 0 && dst[i] == 0 {
			continue
		}
		if !sameBits(dst[i], want) {
			t.Fatalf("element %d: got %08x want %08x", i, math.Float32bits(dst[i]), want)
		}
	}
	return dst
}

func TestMatrixReference(t *testing.T) {
	f := readFixture(t)
	if len(f.Cases) != 12 {
		t.Fatal("incomplete matrix cases")
	}
	for _, c := range f.Cases {
		t.Run(c.Name, func(t *testing.T) { decodeCase(t, c) })
	}
}

func TestChunkBoundaries(t *testing.T) {
	for _, c := range readFixture(t).Cases {
		f, r := options(t, c)
		whole := decodeCase(t, c)
		chunkRows := 1
		if f == FP8Block32 {
			chunkRows = 32
		}
		bytesPerRow := c.Cols
		if f == FP4Row32 {
			bytesPerRow /= 2
		}
		scaleCols := (c.Cols + 31) / 32
		for row := 0; row < c.Rows; row += chunkRows {
			n := min(chunkRows, c.Rows-row)
			srow, srows := row, n
			if f == FP8Block32 {
				srow = row / 32
				srows = 1
			}
			dst := make([]float32, n*c.Cols)
			if err := Decode(dst, c.Data[row*bytesPerRow:(row+n)*bytesPerRow], c.Scales[srow*scaleCols:(srow+srows)*scaleCols], n, c.Cols, f, r); err != nil {
				t.Fatal(err)
			}
			for i, v := range dst {
				if math.Float32bits(v) != math.Float32bits(whole[row*c.Cols+i]) {
					t.Fatal("chunk differs", c.Name, row, i)
				}
			}
		}
	}
}

func TestInvalidInputsAreAtomic(t *testing.T) {
	for _, mode := range []string{"zero rows", "negative cols", "overflow dims", "bad format", "bad rounding", "short data", "long data", "short scales", "long scales", "short dst", "long dst", "row tail", "late NaN", "NaN scale", "overflow value", "packed NaN scale", "packed overflow"} {
		t.Run(mode, func(t *testing.T) {
			rows, cols := 2, 32
			format, rounding := FP8Row32, Float32
			dst := make([]float32, 64)
			data := make([]byte, 64)
			scales := []byte{127, 127}
			switch mode {
			case "zero rows":
				rows = 0
			case "negative cols":
				cols = -1
			case "overflow dims":
				rows = int(^uint(0) >> 1)
				cols = 2
			case "bad format":
				format = 255
			case "bad rounding":
				rounding = 255
			case "short data":
				data = data[:63]
			case "long data":
				data = append(data, 0)
			case "short scales":
				scales = scales[:1]
			case "long scales":
				scales = append(scales, 127)
			case "short dst":
				dst = dst[:63]
			case "long dst":
				dst = append(dst, 0)
			case "row tail":
				cols = 31
			case "late NaN":
				data[63] = 127
			case "NaN scale":
				scales[1] = 255
			case "overflow value":
				data[63] = 126
				scales[1] = 254
			case "packed NaN scale", "packed overflow":
				format = FP4Row32
				data = make([]byte, 32)
				data[31] = 0x70
				scales[1] = 255
				if mode == "packed overflow" {
					scales[1] = 254
				}
			}
			for i := range dst {
				dst[i] = float32(i) + .5
			}
			before := slices.Clone(dst)
			err := Decode(dst, data, scales, rows, cols, format, rounding)
			if err == nil || !slices.Equal(dst, before) {
				t.Fatal("missing error or modified destination", err)
			}
			if (mode == "late NaN" || mode == "NaN scale" || mode == "overflow value" || format == FP4Row32) && !errors.Is(err, ErrNonFinite) {
				t.Fatal("lost nonfinite error", err)
			}
		})
	}
}

func TestConcurrentDecodeAndAllocations(t *testing.T) {
	c := readFixture(t).Cases[0]
	want := decodeCase(t, c)
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			dst := make([]float32, len(want))
			for range 20 {
				if err := Decode(dst, c.Data, c.Scales, c.Rows, c.Cols, FP8Block32, Float32); err != nil {
					t.Error(err)
					return
				}
				for i, v := range dst {
					if math.Float32bits(v) != math.Float32bits(want[i]) {
						t.Errorf("concurrent decode differs at %d", i)
						return
					}
				}
			}
		})
	}
	wg.Wait()
	dst := make([]float32, c.Rows*c.Cols)
	if n := testing.AllocsPerRun(10, func() {
		if err := Decode(dst, c.Data, c.Scales, c.Rows, c.Cols, FP8Block32, Float32); err != nil {
			t.Error(err)
		}
	}); n != 0 {
		t.Fatalf("decoder allocated %g times", n)
	}
}

func FuzzDecode(f *testing.F) {
	f.Add([]byte{1, 2, 127}, byte(0), byte(1), byte(32), byte(0))
	f.Fuzz(func(t *testing.T, seed []byte, mode, rb, cb, round byte) {
		if len(seed) == 0 {
			return
		}
		rows, cols := int(rb%65), int(cb%97)
		format := Format(mode % 4)
		rounding := Rounding(round % 3)
		n := rows * cols
		dst := make([]float32, n)
		for i := range dst {
			dst[i] = .25
		}
		ndata := n
		if format == FP4Row32 {
			ndata /= 2
		}
		data := make([]byte, ndata)
		for i := range data {
			data[i] = seed[i%len(seed)]
		}
		nscales := rows * ((cols + 31) / 32)
		if format == FP8Block32 {
			nscales = ((rows + 31) / 32) * ((cols + 31) / 32)
		}
		scales := make([]byte, nscales)
		for i := range scales {
			scales[i] = seed[(i+1)%len(seed)]
		}
		err := Decode(dst, data, scales, rows, cols, format, rounding)
		for _, v := range dst {
			if err != nil && v != .25 {
				t.Fatal("partial write")
			}
			if err == nil && (math.IsNaN(float64(v)) || math.IsInf(float64(v), 0)) {
				t.Fatal("nonfinite success")
			}
		}
	})
}
