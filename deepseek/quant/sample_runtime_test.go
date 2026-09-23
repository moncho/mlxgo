//go:build mlx && mlxruntime

package quant

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"testing"

	mlx "github.com/moncho/mlxgo"
)

func TestReleasedSampleProjections(t *testing.T) {
	dir, cases := readReleasedReference(t)
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			for _, c := range cases {
				t.Run(c.Name, func(t *testing.T) {
					data, scales, format := releasedBytes(t, dir, c)
					var got []float32
					err := mlx.Batch(func() error {
						decoded := make([]float32, c.Rows*c.Cols)
						mode := Float32
						if c.Rounding == "bfloat16" {
							mode = BFloat16
						}
						if err := Decode(decoded, data, scales, c.Rows, c.Cols, format, mode); err != nil {
							return err
						}
						h := sha256.New()
						appendDecodedHash(h, decoded)
						if hex.EncodeToString(h.Sum(nil)) != c.DecodedSHA {
							// Do not call t.Fatal from the MLX worker goroutine.
							return fmt.Errorf("sample decoding on MLX worker differs from reference")
						}
						w, err := mlx.NewFloat32(decoded, []int{c.Groups, c.Rows / c.Groups, c.Cols})
						if err != nil {
							return err
						}
						defer w.Close()
						if mode == BFloat16 {
							bf, err := mlx.AsType(w, mlx.BFloat16)
							if err != nil {
								return err
							}
							defer bf.Close()
							wf, err := mlx.AsType(bf, mlx.Float32)
							if err != nil {
								return err
							}
							defer wf.Close()
							w = wf
						}
						read, err := w.Float32Data()
						if err != nil {
							return err
						}
						h.Reset()
						appendDecodedHash(h, read)
						if hex.EncodeToString(h.Sum(nil)) != c.DecodedSHA {
							return fmt.Errorf("MLX upload/BF16 round trip differs from reference")
						}
						wt, err := mlx.TransposeAxes(w, []int{0, 2, 1})
						if err != nil {
							return err
						}
						defer wt.Close()
						x, err := mlx.NewFloat32(releasedInputs(c), []int{c.Groups, 3, c.Cols})
						if err != nil {
							return err
						}
						defer x.Close()
						y, err := mlx.Matmul(x, wt)
						if err != nil {
							return err
						}
						defer y.Close()
						got, err = y.Float32Data()
						return err
					})
					if err != nil {
						t.Fatal(err)
					}
					if len(got) != len(c.Projection) {
						t.Fatal("projection shape changed")
					}
					var maxAbs, maxNormalized float64
					for i, v := range got {
						err := math.Abs(float64(v) - c.Projection[i])
						l1 := c.ProjectionL1[i]
						if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || err > 1e-5+2e-6*l1 {
							t.Fatalf("projection %d: got %g want %.12g error %g l1 %g", i, v, c.Projection[i], err, l1)
						}
						maxAbs = math.Max(maxAbs, err)
						maxNormalized = math.Max(maxNormalized, err/math.Max(1, l1))
					}
					t.Logf("%d projections, groups=%d: max_abs_error=%g max_l1_normalized_error=%g", len(got), c.Groups, maxAbs, maxNormalized)
				})
			}
		})
	}
}
