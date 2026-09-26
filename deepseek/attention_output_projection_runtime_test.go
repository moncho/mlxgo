//go:build mlx && mlxruntime

package deepseek

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/deepseek/quant"
)

func TestFP8OutputGroupIsolation(t *testing.T) {
	defer mlx.SetDefaultCPU()
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			// Distinct group weights and an input supported on only one group.
			w := FP8AttentionWeight{Data: make([]byte, 2*32*32), Scales: []byte{127, 127}}
			for i := range w.Data {
				w.Data[i] = 0x38
				if i >= 32*32 {
					w.Data[i] = 0x40
				}
			}
			m := &Model{config: Config{Groups: 2, ORank: 32, Heads: 2, HeadDim: 32}}
			defer m.Close()
			name := "layers.0.attn.wo_a.weight"
			if err := mlx.Batch(func() error { return m.initFP8Attention(name, []int{64, 32}, w) }); err != nil {
				t.Fatal(err)
			}
			clear(w.Data)
			clear(w.Scales)
			session := &Session{model: m}
			for active := 0; active < 2; active++ {
				input := make([]float32, 3*2*32)
				for n := 0; n < 3; n++ {
					for col := 0; col < 32; col++ {
						input[(n*2+active)*32+col] = float32(n + 1)
					}
				}
				x, err := mlx.NewFloat32(input, []int{1, 3, 2, 32})
				if err != nil {
					t.Fatal(err)
				}
				defer x.Close()
				y, err := one(func(s *scope) mlx.Array { return session.attentionOutputProjection(s, x, name) })
				if err != nil {
					t.Fatal(err)
				}
				defer y.Close()
				if !slices.Equal(y.Shape(), []int{1, 3, 64}) {
					t.Fatal("wrong grouped shape", y.Shape())
				}
				// Lazy graph owns all group dependencies even after model close.
				if active == 1 {
					if err := m.Close(); err != nil {
						t.Fatal(err)
					}
				}
				got, err := y.Float32Data()
				if err != nil {
					t.Fatal(err)
				}
				for i, v := range got {
					var want float32
					if i%64/32 == active {
						want = float32(32 * (active + 1) * (i/64 + 1))
					}
					if v != want {
						t.Fatalf("group leakage/order: output %d=%g want %g", i, v, want)
					}
				}
			}
		})
	}
}

func TestReleasedFP8OutputProjection(t *testing.T) {
	dir := os.Getenv("MLXGO_DEEPSEEK_SAMPLE_DIR")
	if dir == "" {
		t.Skip("set MLXGO_DEEPSEEK_SAMPLE_DIR to the existing sample with reference.json")
	}
	read := func(name string, limit int64) []byte {
		f, err := os.Open(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		b, err := io.ReadAll(io.LimitReader(f, limit+1))
		if err != nil || int64(len(b)) > limit {
			t.Fatalf("read %s: %v", name, err)
		}
		return b
	}
	checkHash := func(b []byte, want string) {
		h := sha256.Sum256(b)
		if hex.EncodeToString(h[:]) != want {
			t.Fatalf("hash mismatch: got %x want %s", h, want)
		}
	}
	var ref struct {
		Schema        int    `json:"schema"`
		Revision      string `json:"revision"`
		Torch         string `json:"torch_version"`
		ConversionSHA string `json:"conversion_sha256"`
		ManifestSHA   string `json:"manifest_sha256"`
		Cases         []struct {
			Name               string `json:"name"`
			Rows, Cols, Groups int
			Format, Rounding   string
			DataSHA            string    `json:"data_sha256"`
			ScalesSHA          string    `json:"scales_sha256"`
			Float32SHA         string    `json:"float32_sha256"`
			DecodedSHA         string    `json:"decoded_sha256"`
			Projection         []float64 `json:"projection"`
			L1                 []float64 `json:"projection_l1"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(read("reference.json", 4<<20), &ref); err != nil {
		t.Fatal(err)
	}
	if ref.Schema != 1 || ref.Revision != ReferenceRevision || ref.Torch != "2.14.0" || ref.ConversionSHA != "035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76" || len(ref.Cases) != 3 {
		t.Fatal("output reference provenance mismatch")
	}
	checkHash(read("manifest.json", 16<<10), ref.ManifestSHA)
	c := ref.Cases[2]
	if c.Name != "layers.0.attn.wo_a" || c.Rows != 8192 || c.Cols != 4096 || c.Groups != 8 || c.Format != "fp8_block32" || c.Rounding != "bfloat16" || len(c.Projection) != 3*c.Rows || len(c.L1) != len(c.Projection) {
		t.Fatal("output reference layout mismatch")
	}
	for i, v := range c.Projection {
		if math.IsNaN(v) || math.IsInf(v, 0) || math.IsNaN(c.L1[i]) || math.IsInf(c.L1[i], 0) || c.L1[i] < math.Abs(v) {
			t.Fatal("invalid oracle", i)
		}
	}
	w := FP8AttentionWeight{Data: read(c.Name+".weight.bin", int64(c.Rows*c.Cols)), Scales: read(c.Name+".scale.bin", int64(c.Rows*c.Cols/1024))}
	checkHash(w.Data, c.DataSHA)
	checkHash(w.Scales, c.ScalesSHA)
	if err := validateFP8OutputWeight(w, c.Groups, c.Rows/c.Groups, c.Cols); err != nil {
		t.Fatal(err)
	}
	// Independently verify that skipping BF16 conversion is exact for this
	// entire sample, not just for the projected input rows. Keep decoding bounded.
	h := sha256.New()
	decoded := make([]float32, 32*c.Cols)
	bytes := make([]byte, 4*len(decoded))
	for row := 0; row < c.Rows; row += 32 {
		if err := quant.Decode(decoded, w.Data[row*c.Cols:(row+32)*c.Cols], w.Scales[row*c.Cols/1024:(row+32)*c.Cols/1024], 32, c.Cols, quant.FP8Block32, quant.Float32); err != nil {
			t.Fatal(err)
		}
		for i, v := range decoded {
			if v == 0 {
				v = 0
			}
			binary.LittleEndian.PutUint32(bytes[i*4:], math.Float32bits(v))
		}
		h.Write(bytes)
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != c.Float32SHA || got != c.DecodedSHA {
		t.Fatal("FP8 and official BF16 weights differ", got)
	}
	t.Logf("%d weights match both float32 and official BF16 decoded hashes", c.Rows*c.Cols)
	defer mlx.SetDefaultCPU()
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			config := Config{Groups: c.Groups, ORank: c.Rows / c.Groups, Heads: 64, HeadDim: 512}
			m := &Model{config: config}
			defer m.Close()
			name := c.Name + ".weight"
			if err := mlx.Batch(func() error { return m.initFP8Attention(name, []int{c.Rows, c.Cols}, w) }); err != nil {
				t.Fatal(err)
			}
			session := &Session{model: m}
			for _, tokens := range []int{1, 3, 32, 128} {
				t.Run(fmt.Sprintf("tokens%d", tokens), func(t *testing.T) {
					input := make([]float32, tokens*c.Groups*c.Cols)
					for n := 0; n < tokens; n++ {
						for g := 0; g < c.Groups; g++ {
							for col := 0; col < c.Cols; col++ {
								i := (n*c.Groups+g)*c.Cols + col
								switch n % 3 {
								case 0:
									input[i] = float32((col*37+g*17)%257-128) / 128
								case 1:
									input[i] = float32((col*13+g*29)%127-63) / 64
								case 2:
									if col == 0 {
										input[i] = 1
									}
									if col == 31 {
										input[i] = -2
									}
									if col == c.Cols-1 {
										input[i] = .5
									}
								}
							}
						}
					}
					x, err := mlx.NewFloat32(input, []int{1, tokens, config.Heads, config.HeadDim})
					if err != nil {
						t.Fatal(err)
					}
					defer x.Close()
					y, err := one(func(s *scope) mlx.Array { return session.attentionOutputProjection(s, x, name) })
					if err != nil {
						t.Fatal(err)
					}
					defer y.Close()
					got, err := y.Float32Data()
					if err != nil {
						t.Fatal(err)
					}
					if len(got) != tokens*c.Rows {
						t.Fatal("incorrect output length")
					}
					var maxAbs, maxNormalized float64
					for i, v := range got {
						n, g, r := i/c.Rows, (i%c.Rows)/config.ORank, i%config.ORank
						j := (g*3+n%3)*config.ORank + r
						delta := math.Abs(float64(v) - c.Projection[j])
						l1 := c.L1[j]
						if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || delta > 1e-5+2e-6*l1 {
							t.Fatalf("output %d got %g want %g error=%g l1=%g", i, v, c.Projection[j], delta, l1)
						}
						maxAbs = math.Max(maxAbs, delta)
						maxNormalized = math.Max(maxNormalized, delta/math.Max(1, l1))
					}
					t.Logf("%d grouped projections: max_abs=%g max_l1_normalized=%g", len(got), maxAbs, maxNormalized)
				})
			}
		})
	}
}
