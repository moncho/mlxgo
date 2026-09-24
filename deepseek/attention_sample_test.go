package deepseek

import (
	"compress/gzip"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/moncho/mlxgo/deepseek/quant"
)

type attentionTensorReference struct {
	Name       string `json:"name"`
	Shape      []int  `json:"shape"`
	DType      string `json:"dtype"`
	DataSHA    string `json:"data_sha256"`
	ScalesSHA  string `json:"scales_sha256"`
	DecodedSHA string `json:"decoded_sha256"`
}

type attentionCaseReference struct {
	Prefill    int       `json:"prefill"`
	Output     []float32 `json:"output"`
	Cache      []float32 `json:"cache"`
	Compressed []float32 `json:"compressed"`
	Keys       []float32 `json:"keys"`
	Pending    []float32 `json:"pending"`
}

type attentionIndexReference struct {
	Position int     `json:"position"`
	Indices  []int32 `json:"indices"`
}

type attentionReference struct {
	Schema      int    `json:"schema"`
	Revision    string `json:"revision"`
	ModelSHA    string `json:"model_sha256"`
	ConfigSHA   string `json:"config_sha256"`
	Torch       string `json:"torch_version"`
	ManifestSHA string `json:"manifest_sha256"`
	Config      struct {
		Config
		NormEpsilon float32 `json:"norm_eps"`
	} `json:"config"`
	Tensors    []attentionTensorReference `json:"tensors"`
	Tokens     int                        `json:"tokens"`
	Full       []float32                  `json:"full_output"`
	Cases      []attentionCaseReference   `json:"cases"`
	Layer      int                        `json:"layer"`
	IndexKeys  []float32                  `json:"index_keys"`
	IndexCases []attentionIndexReference  `json:"index_cases"`
}

func readAttentionReference(t *testing.T) (string, attentionReference) {
	return readAttentionLayerReference(t, 0)
}

func readAttentionLayerReference(t *testing.T, layer int) (string, attentionReference) {
	t.Helper()
	env := "MLXGO_DEEPSEEK_ATTENTION_DIR"
	if layer == 2 {
		env = "MLXGO_DEEPSEEK_COMPRESSED_ATTENTION_DIR"
	}
	dir := os.Getenv(env)
	if dir == "" {
		t.Skip("set " + env + " to an absolute attention sample directory")
	}
	f, err := os.Open(filepath.Join(dir, "attention-reference.json.gz"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	z, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	defer z.Close()
	d := json.NewDecoder(io.LimitReader(z, 128<<20))
	var r attentionReference
	if err := d.Decode(&r); err != nil {
		t.Fatal(err)
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		t.Fatal("trailing reference", err)
	}
	fm, err := os.Open(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer fm.Close()
	b, err := io.ReadAll(io.LimitReader(fm, (32<<10)+1))
	if err != nil || len(b) > 32<<10 {
		t.Fatal("invalid manifest", err)
	}
	h := sha256.Sum256(b)
	if r.Schema != 1 || r.Revision != ReferenceRevision || r.Torch != "2.14.0" || r.ModelSHA != "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65" || r.ConfigSHA != "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879" || hex.EncodeToString(h[:]) != r.ManifestSHA {
		t.Fatal("attention reference provenance mismatch")
	}
	c := &r.Config.Config
	if c.Dim != 5120 || c.Heads != 64 || c.QRank != 1280 || c.HeadDim != 512 || c.RopeDim != 64 || c.Groups != 8 || c.ORank != 1024 || c.Window != 128 || c.RopeTheta != 10000 || r.Config.NormEpsilon != float32(1e-20) || r.Tokens != 131 || r.Layer != layer {
		t.Fatal("unexpected attention configuration")
	}
	if layer == 0 {
		if c.Layers != 1 || c.MaxSeq != 131 || !slices.Equal(c.Ratios, []int{0}) {
			t.Fatal("unexpected sliding-window config")
		}
	} else {
		if c.Layers != 3 || c.MaxSeq != 1281 || !slices.Equal(c.Ratios, []int{0, 0, 2}) || !slices.Equal(c.KVSources, []int{2}) || !slices.Equal(c.IndexSources, []int{2}) || c.CompressTheta != 160000 || c.OriginalSeq != 65536 || c.RopeFactor != 16 || c.BetaFast != 32 || c.BetaSlow != 1 || c.IndexHeads != 32 || c.IndexDim != 128 || c.IndexTopK != 512 || c.CandidateSource != 20 || c.CandidateBlocks != 2048 || c.CandidateSize != 8 {
			t.Fatal("unexpected compressed attention config")
		}
	}
	c.Hyper.NormEpsilon = r.Config.NormEpsilon
	want := map[string][]int{
		"layers.0.attn.wq_a.weight": {1280, 5120}, "layers.0.attn.wq_b.weight": {32768, 1280},
		"layers.0.attn.wkv.weight": {512, 5120}, "layers.0.attn.wo_a.weight": {8192, 4096},
		"layers.0.attn.wo_b.weight": {5120, 8192}, "layers.0.attn.q_norm.weight": {1280},
		"layers.0.attn.kv_norm.weight": {512}, "layers.0.attn.attn_sink": {64}, "layers.0.attn_norm.weight": {5120},
	}
	if layer == 2 {
		remapped := make(map[string][]int)
		for name, shape := range want {
			remapped[strings.Replace(name, "layers.0.", "layers.2.", 1)] = shape
		}
		want = remapped
		for name, shape := range map[string][]int{
			"compressor.norm.weight": {512}, "compressor.wkv.weight": {512, 5120}, "compressor.wgate.weight": {512, 5120},
			"indexer.k_norm.weight": {128}, "indexer.wk.weight": {128, 512}, "indexer.weights_proj.weight": {32, 5120}, "indexer.wq_b.weight": {4096, 1280},
		} {
			want["layers.2.attn."+name] = shape
		}
	}
	if len(r.Tensors) != len(want) {
		t.Fatal("incomplete attention weights")
	}
	for _, tensor := range r.Tensors {
		shape, ok := want[tensor.Name]
		if !ok || !slices.Equal(shape, tensor.Shape) {
			t.Fatal("unexpected/duplicate attention tensor")
		}
		dtype := "BF16"
		if len(shape) == 2 {
			dtype = "F8_E4M3"
			if strings.Contains(tensor.Name, ".compressor.") || strings.HasSuffix(tensor.Name, "indexer.wk.weight") || strings.HasSuffix(tensor.Name, "indexer.weights_proj.weight") {
				dtype = "BF16"
			}
		} else if strings.HasSuffix(tensor.Name, "attn_sink") {
			dtype = "F32"
		}
		if tensor.DType != dtype {
			t.Fatal("unexpected storage dtype")
		}
		delete(want, tensor.Name)
	}
	prefills := []int{1, 127, 128, 129}
	if layer == 2 {
		prefills = []int{1, 2, 127, 128, 129}
	}
	if len(r.Full) != 131*5120 || len(r.Cases) != len(prefills) {
		t.Fatal("incorrect attention oracle size")
	}
	finite := func(values []float32) {
		for _, v := range values {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatal("nonfinite reference")
			}
		}
	}
	finite(r.Full)
	for i, c := range r.Cases {
		if c.Prefill != prefills[i] || len(c.Output) != (c.Prefill+2)*5120 || len(c.Cache) != min(c.Prefill+2, 128)*512 {
			t.Fatal("incorrect attention schedule")
		}
		finite(c.Output)
		finite(c.Cache)
		if layer == 2 {
			end := c.Prefill + 2
			if len(c.Compressed) != end/2*512 || len(c.Keys) != end/2*128 || len(c.Pending) != end%2*5120 {
				t.Fatal("incorrect compressed cache size")
			}
			finite(c.Compressed)
			finite(c.Keys)
			finite(c.Pending)
		}
	}
	if layer == 2 {
		if len(r.IndexKeys) != 640*128 || len(r.IndexCases) != 3 {
			t.Fatal("incorrect index probe size")
		}
		finite(r.IndexKeys)
		for i, probe := range r.IndexCases {
			if probe.Position != []int{1022, 1024, 1280}[i] || len(probe.Indices) != min(probe.Position/2, 512) {
				t.Fatal("incorrect index probe")
			}
			for j, id := range probe.Indices {
				if id < 0 || int(id) >= probe.Position/2 || (j > 0 && probe.Indices[j-1] >= id) {
					t.Fatal("invalid index selection")
				}
			}
		}
	}
	return dir, r
}

func loadAttentionTensor(t *testing.T, dir string, m attentionTensorReference) []float32 {
	t.Helper()
	d, err := os.Open(filepath.Join(dir, m.Name+".bin"))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	h := sha256.New()
	var values []float32
	if m.DType == "F8_E4M3" {
		s, err := os.Open(filepath.Join(dir, strings.TrimSuffix(m.Name, ".weight")+".scale.bin"))
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		sh := sha256.New()
		mode := quant.Float32
		if strings.HasSuffix(m.Name, "wo_a.weight") {
			mode = quant.BFloat16
		}
		values, err = quant.ReadMatrix(io.TeeReader(d, h), io.TeeReader(s, sh), m.Shape[0], m.Shape[1], quant.FP8Block32, mode, 192<<20)
		if err != nil {
			t.Fatal(err)
		}
		if hex.EncodeToString(sh.Sum(nil)) != m.ScalesSHA {
			t.Fatal("attention scale hash mismatch")
		}
	} else {
		width := 2
		if m.DType == "F32" {
			width = 4
		}
		count := 1
		for _, dim := range m.Shape {
			count *= dim
		}
		b, err := io.ReadAll(io.LimitReader(io.TeeReader(d, h), int64(count*width+1)))
		if err != nil || len(b) != count*width {
			t.Fatal("invalid BF16/F32 data", err)
		}
		values = make([]float32, count)
		for i := range values {
			var bits uint32
			if width == 2 {
				bits = uint32(binary.LittleEndian.Uint16(b[2*i:])) << 16
			} else {
				bits = binary.LittleEndian.Uint32(b[4*i:])
			}
			values[i] = math.Float32frombits(bits)
		}
	}
	if hex.EncodeToString(h.Sum(nil)) != m.DataSHA || matrixSHA(values) != m.DecodedSHA {
		t.Fatal("attention weight hash mismatch:", m.Name)
	}
	return values
}

func attentionInputs(start, n, dim int) []float32 {
	x := make([]float32, n*dim)
	for i := 0; i < n; i++ {
		pos := start + i
		for j := 0; j < dim; j++ {
			v := float32((j*37+pos*17+(j%13)*pos*7)%257-128) / 128
			if pos == 0 {
				v = 0
			}
			if pos == 2 {
				v *= 8
			}
			x[i*dim+j] = v
		}
	}
	return x
}

func TestReleasedAttentionWeights(t *testing.T) {
	dir, ref := readAttentionReference(t)
	for _, m := range ref.Tensors {
		t.Run(m.Name, func(t *testing.T) { t.Logf("%d values match reference", len(loadAttentionTensor(t, dir, m))) })
	}
}

func TestReleasedCompressedAttentionWeights(t *testing.T) {
	dir, ref := readAttentionLayerReference(t, 2)
	for _, m := range ref.Tensors {
		t.Run(m.Name, func(t *testing.T) { t.Logf("%d values match reference", len(loadAttentionTensor(t, dir, m))) })
	}
}
