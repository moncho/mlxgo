//go:build mlx && mlxruntime

package deepseek

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"testing"

	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/deepseek/quant"
)

func TestReleasedFP8QueryProjection(t *testing.T) {
	path := os.Getenv("MLXGO_DEEPSEEK_QUERY_REFERENCE")
	if path == "" {
		t.Skip("set MLXGO_DEEPSEEK_QUERY_REFERENCE to the independent query-projection-reference.json.gz")
	}
	dir, attention := readAttentionReference(t)
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
	var ref struct {
		Schema      int       `json:"schema"`
		Revision    string    `json:"revision"`
		Torch       string    `json:"torch_version"`
		ManifestSHA string    `json:"manifest_sha256"`
		HeaderSHA   string    `json:"header_sha256"`
		Name        string    `json:"name"`
		Rows        int       `json:"rows"`
		Cols        int       `json:"cols"`
		DataSHA     string    `json:"data_sha256"`
		ScalesSHA   string    `json:"scales_sha256"`
		DecodedSHA  string    `json:"decoded_sha256"`
		InputRecipe string    `json:"input_recipe"`
		Projection  []float64 `json:"projection"`
		L1          []float64 `json:"projection_l1"`
	}
	d := json.NewDecoder(io.LimitReader(z, 16<<20))
	if err := d.Decode(&ref); err != nil {
		t.Fatal(err)
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		t.Fatal("trailing reference", err)
	}
	if ref.Schema != 1 || ref.Revision != ReferenceRevision || ref.Torch != "2.14.0" || ref.ManifestSHA != attention.ManifestSHA || ref.HeaderSHA != "ff66dd94d7eb6ef5cc1457b2ac13b422c14e9891914c786af995edaa4f10a614" || ref.Name != "layers.0.attn.wq_b.weight" || ref.Rows != 32768 || ref.Cols != 1280 || ref.InputRecipe != "sample_three_rows_v1" {
		t.Fatal("query projection reference provenance mismatch")
	}
	if len(ref.Projection) != 3*ref.Rows || len(ref.L1) != len(ref.Projection) {
		t.Fatal("incorrect oracle dimensions")
	}
	for i, want := range ref.Projection {
		if math.IsNaN(want) || math.IsInf(want, 0) || math.IsNaN(ref.L1[i]) || math.IsInf(ref.L1[i], 0) || ref.L1[i] < math.Abs(want) {
			t.Fatal("nonfinite/invalid oracle", i)
		}
	}
	var metadata attentionTensorReference
	for _, m := range attention.Tensors {
		if m.Name == ref.Name {
			metadata = m
		}
	}
	if metadata.Name == "" || metadata.DataSHA != ref.DataSHA || metadata.ScalesSHA != ref.ScalesSHA || metadata.DecodedSHA != ref.DecodedSHA {
		t.Fatal("query projection hashes differ from independent attention oracle")
	}
	weight := readPackedAttentionWeight(t, dir, metadata)
	defer mlx.SetDefaultCPU()
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			layer, err := quant.NewFP8Linear(weight.Data, weight.Scales, ref.Rows, ref.Cols, quant.FP8Block32)
			if err != nil {
				t.Fatal(err)
			}
			defer layer.Close()
			for _, tokens := range []int{1, 3, 32, 128} {
				t.Run(fmt.Sprintf("tokens%d", tokens), func(t *testing.T) {
					input := make([]float32, tokens*ref.Cols)
					for m := 0; m < tokens; m++ {
						for col := 0; col < ref.Cols; col++ {
							i := m*ref.Cols + col
							switch m % 3 {
							case 0:
								input[i] = float32(col*37%257-128) / 128
							case 1:
								input[i] = float32(col*13%127-63) / 64
							case 2:
								if col == 0 {
									input[i] = 1
								}
								if col == 31 {
									input[i] = -2
								}
								if col == ref.Cols-1 {
									input[i] = .5
								}
							}
						}
					}
					x, err := mlx.NewFloat32(input, []int{tokens, ref.Cols})
					if err != nil {
						t.Fatal(err)
					}
					defer x.Close()
					y, err := layer.Forward(x)
					if err != nil {
						t.Fatal(err)
					}
					defer y.Close()
					got, err := y.Float32Data()
					if err != nil {
						t.Fatal(err)
					}
					if len(got) != tokens*ref.Rows {
						t.Fatal("wrong output length")
					}
					var maxAbs, maxNormalized float64
					for i, v := range got {
						j := i % len(ref.Projection)
						delta := math.Abs(float64(v) - ref.Projection[j])
						l1 := ref.L1[j]
						if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || delta > 1e-5+2e-6*l1 {
							t.Fatalf("query projection %d: got %g want %g abs_error=%g l1=%g", i, v, ref.Projection[j], delta, l1)
						}
						maxAbs = math.Max(maxAbs, delta)
						maxNormalized = math.Max(maxNormalized, delta/math.Max(1, l1))
					}
					t.Logf("%d projections max_abs=%g max_l1_normalized=%g", len(got), maxAbs, maxNormalized)
				})
			}
		})
	}
}
