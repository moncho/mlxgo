package quant

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"slices"
	"sync"
	"testing"
)

type activationCase struct {
	Name     string   `json:"name"`
	Format   string   `json:"format"`
	Rows     int      `json:"rows"`
	Cols     int      `json:"cols"`
	Input    []uint32 `json:"input_bits"`
	Data     []byte   `json:"data"`
	Scales   []byte   `json:"scales"`
	Expected []uint32 `json:"expected_bits"`
	BF16     []uint32 `json:"bf16_bits"`
}

func readActivationFixture(t testing.TB, path string, real bool) []activationCase {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	z, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	defer z.Close()
	var r struct {
		Schema       int              `json:"schema"`
		Torch        string           `json:"torch_version"`
		Revision     string           `json:"source_revision"`
		KernelSHA    string           `json:"kernel_sha256"`
		AttentionSHA string           `json:"attention_reference_sha256"`
		ManifestSHA  string           `json:"attention_manifest_sha256"`
		Cases        []activationCase `json:"cases"`
	}
	d := json.NewDecoder(io.LimitReader(z, 32<<20))
	if err := d.Decode(&r); err != nil {
		t.Fatal(err)
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		t.Fatal("trailing reference", err)
	}
	if r.Schema != 1 || r.Torch != "2.14.0" || r.Revision != "df42c109f1defefcbfcedbe7d905718a12266e40" || r.KernelSHA != "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455" {
		t.Fatal("unexpected activation provenance")
	}
	count := 19
	if real {
		count = 27
		if r.AttentionSHA != "24edc3f81b8be9b32aae07cdea63af3565f5da78fcb6cdf22d11dbbd5ab23aa2" || r.ManifestSHA != "655a7c7542b6a671aed5e964c939bc5e4f01c1cd5bbed056b52ac52346fbf1f4" {
			t.Fatal("unexpected real cache provenance")
		}
	} else if r.AttentionSHA != "" || r.ManifestSHA != "" {
		t.Fatal("real data in synthetic fixture")
	}
	if len(r.Cases) != count {
		t.Fatal("incomplete activation fixture")
	}
	seen := map[string]bool{}
	for _, c := range r.Cases {
		if seen[c.Name] {
			t.Fatal("duplicate activation case")
		}
		seen[c.Name] = true
	}
	return r.Cases
}

func activationOptions(t testing.TB, c activationCase) (ActivationFormat, []float32) {
	t.Helper()
	f, ok := map[string]ActivationFormat{"fp8_activation32": FP8Activation32, "fp4_index32": FP4Index32, "fp4_cache16": FP4Cache16}[c.Format]
	if !ok {
		t.Fatal("invalid fixture format")
	}
	x := make([]float32, len(c.Input))
	for i, b := range c.Input {
		x[i] = math.Float32frombits(b)
	}
	return f, x
}

func checkActivationCase(t testing.TB, c activationCase) []float32 {
	t.Helper()
	format, x := activationOptions(t, c)
	nd, ns, err := ActivationLayout(c.Rows, c.Cols, format)
	if err != nil {
		t.Fatal(err)
	}
	data, scales := make([]byte, nd), make([]byte, ns)
	if err := QuantizeActivation(data, scales, x, c.Rows, c.Cols, format); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(data, c.Data) {
		for i, v := range data {
			if i >= len(c.Data) || v != c.Data[i] {
				t.Fatalf("encoded value %d: got %02x, expected data %x", i, v, c.Data[max(0, i-2):min(len(c.Data), i+3)])
			}
		}
		t.Fatal("encoded value length")
	}
	if !bytes.Equal(scales, c.Scales) {
		t.Fatalf("scales differ: got %x want %x", scales, c.Scales)
	}
	got := make([]float32, len(x))
	for _, rounding := range []Rounding{Float32, BFloat16} {
		want := c.Expected
		if rounding == BFloat16 {
			want = c.BF16
		}
		if len(want) != len(got) {
			t.Fatal("incomplete decoded fixture")
		}
		if err := DequantizeActivation(got, data, scales, c.Rows, c.Cols, format, rounding); err != nil {
			t.Fatal(err)
		}
		for i, v := range got {
			if math.Float32bits(v) != want[i] {
				t.Fatalf("rounding %d value %d: got %08x want %08x", rounding, i, math.Float32bits(v), want[i])
			}
		}
	}
	return got
}

func TestActivationReference(t *testing.T) {
	for _, c := range readActivationFixture(t, "testdata/activation.json.gz", false) {
		t.Run(c.Name, func(t *testing.T) { checkActivationCase(t, c) })
	}
}

func TestRealCacheQuantization(t *testing.T) {
	path := os.Getenv("MLXGO_DEEPSEEK_ACTIVATION_REFERENCE")
	if path == "" {
		t.Skip("set MLXGO_DEEPSEEK_ACTIVATION_REFERENCE to the real cache quantization oracle")
	}
	for _, c := range readActivationFixture(t, path, true) {
		t.Run(c.Name, func(t *testing.T) { checkActivationCase(t, c) })
	}
}

func TestActivationChunking(t *testing.T) {
	for _, c := range readActivationFixture(t, "testdata/activation.json.gz", false) {
		format, x := activationOptions(t, c)
		nd, ns, err := ActivationLayout(1, c.Cols, format)
		if err != nil {
			t.Fatal(err)
		}
		for row := 0; row < c.Rows; row++ {
			data, scales := make([]byte, nd), make([]byte, ns)
			if err := QuantizeActivation(data, scales, x[row*c.Cols:(row+1)*c.Cols], 1, c.Cols, format); err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(data, c.Data[row*nd:(row+1)*nd]) || !bytes.Equal(scales, c.Scales[row*ns:(row+1)*ns]) {
				t.Fatal("chunk boundary mismatch", c.Name, row)
			}
		}
	}
}

func TestActivationErrorsAreAtomic(t *testing.T) {
	for _, mode := range []string{"zero rows", "negative cols", "overflow dims", "bad format", "tail", "short input", "long input", "short data", "long data", "short scales", "long scales", "late NaN", "late infinity", "reconstruction overflow"} {
		t.Run(mode, func(t *testing.T) {
			rows, cols, format := 2, 32, FP8Activation32
			x := make([]float32, 64)
			data := bytes.Repeat([]byte{99}, 64)
			scales := []byte{99, 99}
			switch mode {
			case "zero rows":
				rows = 0
			case "negative cols":
				cols = -1
			case "overflow dims":
				rows = int(^uint(0) >> 1)
			case "bad format":
				format = 255
			case "tail":
				cols = 31
			case "short input":
				x = x[:63]
			case "long input":
				x = append(x, 0)
			case "short data":
				data = data[:63]
			case "long data":
				data = append(data, 99)
			case "short scales":
				scales = scales[:1]
			case "long scales":
				scales = append(scales, 99)
			case "late NaN":
				x[63] = float32(math.NaN())
			case "late infinity":
				x[63] = float32(math.Inf(-1))
			case "reconstruction overflow":
				x[63] = math.MaxFloat32
			}
			db, sb := slices.Clone(data), slices.Clone(scales)
			err := QuantizeActivation(data, scales, x, rows, cols, format)
			if err == nil || !bytes.Equal(db, data) || !bytes.Equal(sb, scales) {
				t.Fatal("non-atomic/missing error", err)
			}
			if (mode == "late NaN" || mode == "late infinity" || mode == "reconstruction overflow") && !errors.Is(err, ErrNonFinite) {
				t.Fatal("lost nonfinite error", err)
			}
		})
	}
	for _, mode := range []string{"NaN code", "NaN scale", "zero scale", "negative scale", "negative zero scale", "overflow", "bad rounding", "short dst", "short data", "short scales"} {
		t.Run("decode_"+mode, func(t *testing.T) {
			format, rounding := FP8Activation32, Float32
			dst := make([]float32, 64)
			data := make([]byte, 64)
			scales := []byte{127, 127}
			switch mode {
			case "NaN code":
				data[63] = 127
			case "NaN scale":
				scales[1] = 255
			case "zero scale", "negative scale", "negative zero scale":
				format = FP4Cache16
				data = data[:32]
				scales = []byte{56, 56, 56, 0}
				if mode == "negative scale" {
					scales[3] = 184
				}
				if mode == "negative zero scale" {
					scales[3] = 128
				}
			case "overflow":
				data[63] = 126
				scales[1] = 254
			case "bad rounding":
				rounding = 255
			case "short dst":
				dst = dst[:63]
			case "short data":
				data = data[:63]
			case "short scales":
				scales = scales[:1]
			}
			for i := range dst {
				dst[i] = .25
			}
			err := DequantizeActivation(dst, data, scales, 2, 32, format, rounding)
			if err == nil {
				t.Fatal("accepted invalid decode")
			}
			for _, v := range dst {
				if v != .25 {
					t.Fatal("partial decode write")
				}
			}
		})
	}
}

func TestActivationConcurrencyAndAllocations(t *testing.T) {
	for _, format := range []ActivationFormat{FP8Activation32, FP4Index32, FP4Cache16} {
		x := make([]float32, 64)
		for i := range x {
			x[i] = float32(i-32) / 7
		}
		nd, ns, err := ActivationLayout(2, 32, format)
		if err != nil {
			t.Fatal(err)
		}
		data, scales, out := make([]byte, nd), make([]byte, ns), make([]float32, 64)
		run := func(data, scales []byte, out []float32) {
			if err := QuantizeActivation(data, scales, x, 2, 32, format); err != nil {
				t.Error(err)
			}
			if err := DequantizeActivation(out, data, scales, 2, 32, format, Float32); err != nil {
				t.Error(err)
			}
		}
		if n := testing.AllocsPerRun(10, func() { run(data, scales, out) }); n != 0 {
			t.Fatalf("allocated %g times", n)
		}
		var wg sync.WaitGroup
		for range 8 {
			wg.Go(func() {
				d, s, o := make([]byte, nd), make([]byte, ns), make([]float32, 64)
				for range 20 {
					run(d, s, o)
					if !bytes.Equal(d, data) || !bytes.Equal(s, scales) || !slices.Equal(o, out) {
						t.Error("concurrent result differs")
					}
				}
			})
		}
		wg.Wait()
	}
}

func TestActivationLayoutAndZeroScales(t *testing.T) {
	for _, tc := range []struct {
		format             ActivationFormat
		cols, data, scales int
		zeroScale          byte
	}{
		{FP8Activation32, 64, 192, 6, 105}, {FP4Index32, 64, 96, 6, 1}, {FP4Cache16, 48, 72, 9, 1},
	} {
		nd, ns, err := ActivationLayout(3, tc.cols, tc.format)
		if err != nil || nd != tc.data || ns != tc.scales {
			t.Fatal("wrong layout", nd, ns, err)
		}
		data, scales := make([]byte, nd), make([]byte, ns)
		if err := QuantizeActivation(data, scales, make([]float32, 3*tc.cols), 3, tc.cols, tc.format); err != nil {
			t.Fatal(err)
		}
		for _, s := range scales {
			if s != tc.zeroScale {
				t.Fatal("wrong all-zero scale", s, tc.zeroScale)
			}
		}
	}
}

func FuzzQuantizeActivation(f *testing.F) {
	f.Add([]byte{0, 0, 128, 63, 0, 0, 0, 128}, byte(0))
	f.Add([]byte{255, 255, 127, 127, 0, 0, 128, 127}, byte(2))
	f.Fuzz(func(t *testing.T, seed []byte, mode byte) {
		if len(seed) < 4 {
			return
		}
		format := ActivationFormat(mode % 3)
		x := make([]float32, 64)
		for i := range x {
			offset := (i % (len(seed) / 4)) * 4
			x[i] = math.Float32frombits(binary.LittleEndian.Uint32(seed[offset : offset+4]))
		}
		nd, ns, err := ActivationLayout(2, 32, format)
		if err != nil {
			t.Fatal(err)
		}
		data, scales := bytes.Repeat([]byte{99}, nd), bytes.Repeat([]byte{99}, ns)
		err = QuantizeActivation(data, scales, x, 2, 32, format)
		if err != nil {
			for _, v := range append(data, scales...) {
				if v != 99 {
					t.Fatal("partial write on error")
				}
			}
			return
		}
		out := make([]float32, 64)
		if err := DequantizeActivation(out, data, scales, 2, 32, format, Float32); err != nil {
			t.Fatal("encoded output cannot be decoded", err)
		}
		for _, v := range out {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatal("nonfinite success")
			}
		}
		for i, v := range x {
			offset := (i % (len(seed) / 4)) * 4
			if math.Float32bits(v) != binary.LittleEndian.Uint32(seed[offset:offset+4]) {
				t.Fatal("input modified")
			}
		}
	})
}
