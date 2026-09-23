package quant

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"
)

const sampleRevision = "df42c109f1defefcbfcedbe7d905718a12266e40"
const sampleConvertSHA = "035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76"

type releasedCase struct {
	Name         string    `json:"name"`
	Rows         int       `json:"rows"`
	Cols         int       `json:"cols"`
	Groups       int       `json:"groups"`
	Format       string    `json:"format"`
	Rounding     string    `json:"rounding"`
	DataSHA      string    `json:"data_sha256"`
	ScalesSHA    string    `json:"scales_sha256"`
	Float32SHA   string    `json:"float32_sha256"`
	DecodedSHA   string    `json:"decoded_sha256"`
	Projection   []float64 `json:"projection"`
	ProjectionL1 []float64 `json:"projection_l1"`
}

func readReleasedReference(t *testing.T) (string, []releasedCase) {
	t.Helper()
	dir := os.Getenv("MLXGO_DEEPSEEK_SAMPLE_DIR")
	if dir == "" {
		t.Skip("set MLXGO_DEEPSEEK_SAMPLE_DIR to an absolute sample directory with reference.json")
	}
	var ref struct {
		Schema        int            `json:"schema"`
		Revision      string         `json:"revision"`
		TorchVersion  string         `json:"torch_version"`
		ConversionSHA string         `json:"conversion_sha256"`
		ManifestSHA   string         `json:"manifest_sha256"`
		Cases         []releasedCase `json:"cases"`
	}
	b := readBoundedSample(t, filepath.Join(dir, "reference.json"), 4<<20)
	if err := json.Unmarshal(b, &ref); err != nil {
		t.Fatal(err)
	}
	if ref.Schema != 1 || ref.Revision != sampleRevision || ref.ConversionSHA != sampleConvertSHA || ref.TorchVersion != "2.14.0" {
		t.Fatal("reference provenance/version mismatch")
	}
	checkSampleHash(t, readBoundedSample(t, filepath.Join(dir, "manifest.json"), 16<<10), ref.ManifestSHA)
	specs := []releasedCase{
		{Name: "layers.0.attn.wkv", Rows: 512, Cols: 5120, Groups: 1, Format: "fp8_block32", Rounding: "float32"},
		{Name: "layers.0.ffn.experts.0.w1", Rows: 2304, Cols: 5120, Groups: 1, Format: "fp4_row32", Rounding: "float32"},
		{Name: "layers.0.attn.wo_a", Rows: 8192, Cols: 4096, Groups: 8, Format: "fp8_block32", Rounding: "bfloat16"},
	}
	if len(ref.Cases) != len(specs) {
		t.Fatal("expected exactly three reference cases")
	}
	for i, s := range specs {
		c := ref.Cases[i]
		if c.Name != s.Name || c.Rows != s.Rows || c.Cols != s.Cols || c.Groups != s.Groups || c.Format != s.Format || c.Rounding != s.Rounding {
			t.Fatalf("unexpected reference layout: %+v", c)
		}
		if len(c.Projection) != 3*c.Rows || len(c.ProjectionL1) != 3*c.Rows {
			t.Fatal("incorrect reference projection length")
		}
		for j, v := range c.Projection {
			l1 := c.ProjectionL1[j]
			if math.IsNaN(v) || math.IsInf(v, 0) || math.IsNaN(l1) || math.IsInf(l1, 0) || l1 < math.Abs(v) {
				t.Fatal("invalid projection reference")
			}
		}
	}
	return dir, ref.Cases
}

func readBoundedSample(t *testing.T, path string, limit int64) []byte {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	b, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil || int64(len(b)) > limit {
		t.Fatalf("cannot read bounded sample %s: %v", path, err)
	}
	return b
}

func checkSampleHash(t *testing.T, data []byte, expected string) {
	t.Helper()
	h := sha256.Sum256(data)
	if hex.EncodeToString(h[:]) != expected {
		t.Fatalf("sample SHA-256 mismatch: got %x want %s", h, expected)
	}
}

func releasedBytes(t *testing.T, dir string, c releasedCase) ([]byte, []byte, Format) {
	t.Helper()
	n, ns, format := c.Rows*c.Cols, (c.Rows/32)*(c.Cols/32), FP8Block32
	if c.Format == "fp4_row32" {
		n, ns, format = n/2, c.Rows*(c.Cols/32), FP4Row32
	}
	data := readBoundedSample(t, filepath.Join(dir, c.Name+".weight.bin"), int64(n))
	scales := readBoundedSample(t, filepath.Join(dir, c.Name+".scale.bin"), int64(ns))
	if len(data) != n || len(scales) != ns {
		t.Fatal("incorrect sample length")
	}
	checkSampleHash(t, data, c.DataSHA)
	checkSampleHash(t, scales, c.ScalesSHA)
	return data, scales, format
}

// Canonicalize signed zero only; all nonzero values must agree bit-for-bit.
func appendDecodedHash(w io.Writer, values []float32) {
	var b [4096]byte
	for len(values) > 0 {
		n := min(len(values), len(b)/4)
		for i, v := range values[:n] {
			bits := math.Float32bits(v)
			if v == 0 {
				bits = 0
			}
			binary.LittleEndian.PutUint32(b[i*4:], bits)
		}
		_, _ = w.Write(b[:n*4])
		values = values[n:]
	}
}

func releasedInputs(c releasedCase) []float32 {
	x := make([]float32, c.Groups*3*c.Cols)
	for g := 0; g < c.Groups; g++ {
		base := g * 3 * c.Cols
		for col := 0; col < c.Cols; col++ {
			x[base+col] = float32((col*37+g*17)%257-128) / 128
			x[base+c.Cols+col] = float32((col*13+g*29)%127-63) / 64
		}
		x[base+2*c.Cols], x[base+2*c.Cols+31], x[base+3*c.Cols-1] = 1, -2, .5
	}
	return x
}

func TestReleasedSampleDecoding(t *testing.T) {
	dir, cases := readReleasedReference(t)
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			data, scales, format := releasedBytes(t, dir, c)
			modes := []Rounding{Float32}
			if c.Rounding == "bfloat16" {
				modes = append(modes, BFloat16)
			}
			for _, mode := range modes {
				h := sha256.New()
				chunk := make([]float32, 32*c.Cols)
				for row := 0; row < c.Rows; row += 32 {
					start, end := row*c.Cols, (row+32)*c.Cols
					ss, se := row/32*(c.Cols/32), (row/32+1)*(c.Cols/32)
					if format == FP4Row32 {
						start, end = start/2, end/2
						ss, se = row*(c.Cols/32), (row+32)*(c.Cols/32)
					}
					if err := Decode(chunk, data[start:end], scales[ss:se], 32, c.Cols, format, mode); err != nil {
						t.Fatal(err)
					}
					appendDecodedHash(h, chunk)
				}
				want := c.Float32SHA
				if mode == BFloat16 {
					want = c.DecodedSHA
				}
				if got := hex.EncodeToString(h.Sum(nil)); got != want {
					t.Fatalf("decoded checksum mismatch for mode %d: got %s want %s", mode, got, want)
				}
				t.Logf("%d real weights match reference; rounding=%d", c.Rows*c.Cols, mode)
			}
		})
	}
}
