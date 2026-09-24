package deepseek

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

	"github.com/moncho/mlxgo/deepseek/quant"
)

type expertMatrixReference struct {
	Name       string `json:"name"`
	Rows       int    `json:"rows"`
	Cols       int    `json:"cols"`
	DataSHA    string `json:"data_sha256"`
	ScalesSHA  string `json:"scales_sha256"`
	DecodedSHA string `json:"decoded_sha256"`
}

type expertCaseReference struct {
	Name    string    `json:"name"`
	Limit   float32   `json:"limit"`
	Routing []float32 `json:"routing"`
	Output  []float64 `json:"output"`
	L1      []float64 `json:"output_l1"`
}

type expertReference struct {
	Schema      int                     `json:"schema"`
	Revision    string                  `json:"revision"`
	Torch       string                  `json:"torch_version"`
	ModelSHA    string                  `json:"model_sha256"`
	ConvertSHA  string                  `json:"conversion_sha256"`
	ManifestSHA string                  `json:"manifest_sha256"`
	Dim         int                     `json:"dim"`
	Inter       int                     `json:"inter_dim"`
	Matrices    []expertMatrixReference `json:"matrices"`
	Cases       []expertCaseReference   `json:"cases"`
}

func readExpertReference(t *testing.T) (string, expertReference) {
	t.Helper()
	dir := os.Getenv("MLXGO_DEEPSEEK_EXPERT_DIR")
	if dir == "" {
		t.Skip("set MLXGO_DEEPSEEK_EXPERT_DIR to an absolute expert sample directory with expert-reference.json")
	}
	read := func(name string, limit int64) []byte {
		f, err := os.Open(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		b, err := io.ReadAll(io.LimitReader(f, limit+1))
		if err != nil || int64(len(b)) > limit {
			t.Fatalf("invalid bounded reference: %v", err)
		}
		return b
	}
	var ref expertReference
	if err := json.Unmarshal(read("expert-reference.json", 8<<20), &ref); err != nil {
		t.Fatal(err)
	}
	h := sha256.Sum256(read("manifest.json", 16<<10))
	if ref.Schema != 1 || ref.Revision != ReferenceRevision || ref.Torch != "2.14.0" || ref.Dim != 5120 || ref.Inter != 2304 || len(ref.Matrices) != 3 || len(ref.Cases) != 3 || hex.EncodeToString(h[:]) != ref.ManifestSHA || ref.ModelSHA != "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65" || ref.ConvertSHA != "035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76" {
		t.Fatal("expert reference provenance/layout mismatch")
	}
	for i, m := range ref.Matrices {
		rows, cols := ref.Inter, ref.Dim
		if i == 1 {
			rows, cols = cols, rows
		}
		if m.Name != "layers.0.ffn.experts.0."+[]string{"w1", "w2", "w3"}[i] || m.Rows != rows || m.Cols != cols {
			t.Fatal("unexpected expert matrix")
		}
	}
	for i, c := range ref.Cases {
		limit := float32(10)
		if i == 2 {
			limit = 0
		}
		if c.Name != []string{"clipped_unweighted", "clipped_routed", "unclipped_routed"}[i] || c.Limit != limit || len(c.Routing) != 6 || len(c.Output) != 6*ref.Dim || len(c.L1) != len(c.Output) {
			t.Fatal("unexpected expert case")
		}
		for j, r := range c.Routing {
			want := float32(1)
			if i != 0 {
				want = []float32{1, .25, .75, 0, 1, 1.5}[j]
			}
			if r != want {
				t.Fatal("unexpected routing weights")
			}
		}
		for j, v := range c.Output {
			if math.IsNaN(v) || math.IsInf(v, 0) || math.IsNaN(c.L1[j]) || math.IsInf(c.L1[j], 0) || c.L1[j] < 0 {
				t.Fatal("invalid expert oracle")
			}
		}
	}
	return dir, ref
}

func matrixSHA(values []float32) string {
	h := sha256.New()
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
		h.Write(b[:n*4])
		values = values[n:]
	}
	return hex.EncodeToString(h.Sum(nil))
}

func loadExpertMatrix(t *testing.T, dir string, m expertMatrixReference) []float32 {
	t.Helper()
	d, err := os.Open(filepath.Join(dir, m.Name+".weight.bin"))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	s, err := os.Open(filepath.Join(dir, m.Name+".scale.bin"))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	dh, sh := sha256.New(), sha256.New()
	data, err := quant.ReadMatrix(io.TeeReader(d, dh), io.TeeReader(s, sh), m.Rows, m.Cols, quant.FP4Row32, quant.Float32, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	if hex.EncodeToString(dh.Sum(nil)) != m.DataSHA || hex.EncodeToString(sh.Sum(nil)) != m.ScalesSHA || matrixSHA(data) != m.DecodedSHA {
		t.Fatal("expert matrix checksum mismatch")
	}
	return data
}

func expertInputs(dim int) []float32 {
	x := make([]float32, 6*dim)
	for c := 0; c < dim; c++ {
		x[c] = float32((c*37)%257-128) / 128
		x[dim+c] = float32((c*13)%127-63) / 64
		x[2*dim+c], x[3*dim+c] = x[c]*64, x[dim+c]*-64
	}
	x[5*dim], x[5*dim+31], x[6*dim-1] = 1, -2, .5
	return x
}

func TestReleasedExpertDecoding(t *testing.T) {
	dir, ref := readExpertReference(t)
	for _, m := range ref.Matrices {
		t.Run(m.Name, func(t *testing.T) {
			data := loadExpertMatrix(t, dir, m)
			t.Logf("%d decoded weights match reference using a 64 MiB reader budget", len(data))
		})
	}
}
