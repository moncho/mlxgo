//go:build mlx && mlxruntime

package quant

import (
	"math"
	"os"
	"strings"
	"testing"

	mlx "github.com/moncho/mlxgo"
)

// Quantization remains on the host. This tests worker compatibility and native
// storage of the reconstructed values, not a Metal quantizer or quantized GEMM.
func TestActivationReconstructionInMLX(t *testing.T) {
	cases := readActivationFixture(t, "testdata/activation.json.gz", false)
	if path := os.Getenv("MLXGO_DEEPSEEK_ACTIVATION_REFERENCE"); path != "" {
		cases = readActivationFixture(t, path, true)
	}
	for _, device := range []struct {
		name string
		set  func() error
	}{{"cpu", mlx.SetDefaultCPU}, {"gpu", mlx.SetDefaultGPU}} {
		t.Run(device.name, func(t *testing.T) {
			if err := device.set(); err != nil {
				t.Fatal(err)
			}
			for _, c := range cases {
				// Host tests cover denormals exactly; this smoke test makes no
				// assumptions about Metal's denormal conversion behavior.
				if strings.HasSuffix(c.Name, "_tiny") || strings.HasSuffix(c.Name, "_scale_boundaries") {
					continue
				}
				t.Run(c.Name, func(t *testing.T) {
					format, x := activationOptions(t, c)
					err := mlx.Batch(func() error {
						nd, ns, err := ActivationLayout(c.Rows, c.Cols, format)
						if err != nil {
							return err
						}
						data, scales, dst := make([]byte, nd), make([]byte, ns), make([]float32, len(x))
						if err := QuantizeActivation(data, scales, x, c.Rows, c.Cols, format); err != nil {
							return err
						}
						if err := DequantizeActivation(dst, data, scales, c.Rows, c.Cols, format, BFloat16); err != nil {
							return err
						}
						a, err := mlx.NewFloat32(dst, []int{c.Rows, c.Cols})
						if err != nil {
							return err
						}
						defer a.Close()
						bf, err := mlx.AsType(a, mlx.BFloat16)
						if err != nil {
							return err
						}
						defer bf.Close()
						back, err := mlx.AsType(bf, mlx.Float32)
						if err != nil {
							return err
						}
						defer back.Close()
						got, err := back.Float32Data()
						if err != nil {
							return err
						}
						if len(got) != len(c.BF16) {
							t.Error("native shape changed")
							return nil
						}
						for i, v := range got {
							if math.Float32bits(v) != c.BF16[i] {
								t.Errorf("native value %d: %08x != %08x", i, math.Float32bits(v), c.BF16[i])
								break
							}
						}
						return nil
					})
					if err != nil {
						t.Fatal(err)
					}
				})
			}
		})
	}
}
