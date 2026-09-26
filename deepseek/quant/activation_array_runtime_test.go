//go:build mlx && mlxruntime

package quant

import (
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"sync"
	"testing"

	mlx "github.com/moncho/mlxgo"
)

func TestActivationArrayReference(t *testing.T) {
	cases := readActivationFixture(t, "testdata/activation.json.gz", false)
	if path := os.Getenv("MLXGO_DEEPSEEK_ACTIVATION_REFERENCE"); path != "" {
		cases = readActivationFixture(t, path, true)
	}
	defer mlx.SetDefaultCPU()
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
					format, x := activationOptions(t, c)
					if err := checkActivationArray(c, x, format); err != nil {
						t.Fatal(err)
					}
				})
			}
		})
	}
}

func TestActivationArrayRandomAndConcurrent(t *testing.T) {
	defer mlx.SetDefaultCPU()
	for _, set := range []func() error{mlx.SetDefaultCPU, mlx.SetDefaultGPU} {
		if err := set(); err != nil {
			t.Fatal(err)
		}
		for _, format := range []ActivationFormat{FP8Activation32, FP4Index32, FP4Cache16} {
			rng := rand.New(rand.NewPCG(123, 456))
			x := make([]float32, 256*32)
			for g := 0; g < 256; g++ {
				exponent := rng.IntN(277) - 149
				for j := 0; j < 32; j++ {
					x[g*32+j] = float32(math.Ldexp(rng.Float64()*2-1, exponent))
				}
			}
			c := activationArrayCase(t, x, 256, 32, format)
			if err := checkActivationArray(c, x, format); err != nil {
				t.Fatalf("format %d: %v", format, err)
			}
		}
		var wg sync.WaitGroup
		errs := make(chan error, 8)
		for worker := range 8 {
			format := ActivationFormat(worker % 3)
			x := make([]float32, 32)
			for i := range x {
				x[i] = float32(i-worker) / 7
			}
			c := activationArrayCase(t, x, 1, 32, format)
			wg.Go(func() {
				for range 25 {
					if err := checkActivationArray(c, x, format); err != nil {
						errs <- err
						return
					}
				}
			})
		}
		wg.Wait()
		close(errs)
		for err := range errs {
			t.Error(err)
		}
	}
}

func activationArrayCase(t *testing.T, x []float32, rows, cols int, format ActivationFormat) activationCase {
	t.Helper()
	nd, ns, err := ActivationLayout(rows, cols, format)
	if err != nil {
		t.Fatal(err)
	}
	c := activationCase{Rows: rows, Cols: cols, Data: make([]byte, nd), Scales: make([]byte, ns)}
	if err := QuantizeActivation(c.Data, c.Scales, x, rows, cols, format); err != nil {
		t.Fatal(err)
	}
	y := make([]float32, len(x))
	for _, r := range []Rounding{Float32, BFloat16} {
		if err := DequantizeActivation(y, c.Data, c.Scales, rows, cols, format, r); err != nil {
			t.Fatal(err)
		}
		bits := make([]uint32, len(y))
		for i, v := range y {
			bits[i] = math.Float32bits(v)
		}
		if r == Float32 {
			c.Expected = bits
		} else {
			c.BF16 = bits
		}
	}
	return c
}

func TestActivationArrayNonfinite(t *testing.T) {
	defer mlx.SetDefaultCPU()
	for _, set := range []func() error{mlx.SetDefaultCPU, mlx.SetDefaultGPU} {
		if err := set(); err != nil {
			t.Fatal(err)
		}
		for _, f := range []ActivationFormat{FP8Activation32, FP4Index32, FP4Cache16} {
			err := mlx.Batch(func() error {
				x := make([]float32, 6*32)
				x[0], x[32], x[64], x[96], x[128], x[160] = 1, float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)), math.MaxFloat32, math.Float32frombits(0xffffffff)
				a, err := mlx.NewFloat32(x, []int{6, 32})
				if err != nil {
					return err
				}
				defer a.Close()
				d, sc, err := QuantizeActivationArray(a, f)
				if err != nil {
					return err
				}
				defer d.Close()
				defer sc.Close()
				for _, r := range []Rounding{Float32, BFloat16} {
					y, err := DequantizeActivationArray(d, sc, f, r)
					if err != nil {
						return err
					}
					got, err := y.Float32Data()
					y.Close()
					if err != nil {
						return err
					}
					group := 32
					if f == FP4Cache16 {
						group = 16
					}
					for i, v := range got {
						row, col := i/32, i%32
						wantNaN := row != 0 && !(row == 4 && f == FP4Cache16) && col < group
						if math.IsNaN(float64(v)) != wantNaN {
							return fmt.Errorf("format %d rounding %d value %d: nonfinite policy got %v", f, r, i, v)
						}
					}
				}
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestActivationArrayGradientAndCompile(t *testing.T) {
	defer mlx.SetDefaultCPU()
	for _, set := range []func() error{mlx.SetDefaultCPU, mlx.SetDefaultGPU} {
		if err := set(); err != nil {
			t.Fatal(err)
		}
		for _, f := range []ActivationFormat{FP8Activation32, FP4Index32, FP4Cache16} {
			fn := func(in []mlx.Array) ([]mlx.Array, error) {
				d, sc, err := QuantizeActivationArray(in[0], f)
				if err != nil {
					return nil, err
				}
				defer d.Close()
				defer sc.Close()
				y, err := DequantizeActivationArray(d, sc, f, Float32)
				if err != nil {
					return nil, err
				}
				defer y.Close()
				loss, err := mlx.Sum(y, false)
				if err != nil {
					return nil, err
				}
				return []mlx.Array{loss}, nil
			}
			x, err := mlx.Ones([]int{2, 32}, mlx.Float32)
			if err != nil {
				t.Fatal(err)
			}
			vg, err := mlx.NewValueAndGrad(fn)
			if err != nil {
				x.Close()
				t.Fatal(err)
			}
			values, grads, err := vg.Apply(x)
			if err == nil {
				var got []float32
				got, err = grads[0].Float32Data()
				for _, v := range got {
					if v != 0 {
						t.Errorf("unexpected gradient %v", v)
					}
				}
			}
			mlx.CloseArrays(values)
			mlx.CloseArrays(grads)
			vg.Close()
			if err != nil {
				x.Close()
				t.Fatal(err)
			}
			compiled, err := mlx.Compile(fn, false)
			if err != nil {
				x.Close()
				t.Fatal(err)
			}
			values, err = compiled.Apply(x)
			if err == nil {
				err = values[0].Eval()
			}
			mlx.CloseArrays(values)
			compiled.Close()
			x.Close()
			if err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestActivationArrayCompiledReference(t *testing.T) {
	cases := readActivationFixture(t, "testdata/activation.json.gz", false)
	defer mlx.SetDefaultCPU()
	for _, set := range []func() error{mlx.SetDefaultCPU, mlx.SetDefaultGPU} {
		if err := set(); err != nil {
			t.Fatal(err)
		}
		for _, f := range []ActivationFormat{FP8Activation32, FP4Index32, FP4Cache16} {
			compiled, err := mlx.Compile(func(in []mlx.Array) ([]mlx.Array, error) {
				d, sc, err := QuantizeActivationArray(in[0], f)
				if err != nil {
					return nil, err
				}
				return []mlx.Array{d, sc}, nil
			}, false)
			if err != nil {
				t.Fatal(err)
			}
			encode := func(a mlx.Array, _ ActivationFormat) (mlx.Array, mlx.Array, error) {
				out, err := compiled.Apply(a)
				if err != nil {
					return mlx.Array{}, mlx.Array{}, err
				}
				return out[0], out[1], nil
			}
			for _, c := range cases {
				format, x := activationOptions(t, c)
				if format != f {
					continue
				}
				err := checkActivationArrayWith(c, x, f, encode)
				if err != nil {
					compiled.Close()
					t.Fatalf("%s: %v", c.Name, err)
				}
			}
			rng := rand.New(rand.NewPCG(987, 654))
			x := make([]float32, 256*32)
			for g := range 256 {
				exp := rng.IntN(277) - 149
				for j := range 32 {
					x[g*32+j] = float32(math.Ldexp(rng.Float64()*2-1, exp))
				}
			}
			c := activationArrayCase(t, x, 256, 32, f)
			if err := checkActivationArrayWith(c, x, f, encode); err != nil {
				compiled.Close()
				t.Fatalf("compiled random format %d: %v", f, err)
			}
			compiled.Close()
		}
	}
}

func TestActivationArrayValidation(t *testing.T) {
	if err := mlx.SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}
	x, err := mlx.Ones([]int{1, 32}, mlx.Float32)
	if err != nil {
		t.Fatal(err)
	}
	defer x.Close()
	for _, tc := range []struct {
		shape  []int
		dtype  mlx.DType
		format ActivationFormat
	}{
		{[]int{32}, mlx.Float32, FP8Activation32},
		{[]int{1, 31}, mlx.Float32, FP8Activation32},
		{[]int{0, 32}, mlx.Float32, FP8Activation32},
		{[]int{1, 32}, mlx.BFloat16, FP8Activation32},
		{[]int{1, 32}, mlx.Int32, FP8Activation32},
		{[]int{1, 32}, mlx.Float32, ActivationFormat(255)},
	} {
		a, err := mlx.Zeros(tc.shape, tc.dtype)
		if err != nil {
			t.Fatal(err)
		}
		d, sc, err := QuantizeActivationArray(a, tc.format)
		a.Close()
		d.Close()
		sc.Close()
		if err == nil {
			t.Errorf("accepted invalid quantization: %+v", tc)
		}
	}
	d, sc, err := QuantizeActivationArray(x, FP8Activation32)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	defer sc.Close()
	for _, tc := range []struct {
		data, scales mlx.Array
		format       ActivationFormat
		rounding     Rounding
	}{
		{x, sc, FP8Activation32, Float32}, {d, x, FP8Activation32, Float32},
		{d, d, FP8Activation32, Float32}, {d, sc, ActivationFormat(255), Float32},
		{d, sc, FP8Activation32, Rounding(255)},
	} {
		y, err := DequantizeActivationArray(tc.data, tc.scales, tc.format, tc.rounding)
		y.Close()
		if err == nil {
			t.Error("accepted invalid dequantization")
		}
	}
	x.Close()
	bad, badScale, err := QuantizeActivationArray(x, FP8Activation32)
	bad.Close()
	badScale.Close()
	if err == nil {
		t.Fatal("accepted closed input")
	}
}

func TestActivationArrayDecodeCodes(t *testing.T) {
	scaleBytes := []byte{0, 1, 2, 16, 64, 105, 126, 127, 128, 129, 254, 255}
	defer mlx.SetDefaultCPU()
	for _, set := range []func() error{mlx.SetDefaultCPU, mlx.SetDefaultGPU} {
		if err := set(); err != nil {
			t.Fatal(err)
		}
		for _, f := range []ActivationFormat{FP8Activation32, FP4Index32, FP4Cache16} {
			n, group, packedCols := 256, 32, 32
			if f != FP8Activation32 {
				n, packedCols = 16, 16
			}
			if f == FP4Cache16 {
				group, packedCols = 16, 8
			}
			rows := n * len(scaleBytes)
			data := make([]int32, rows*packedCols)
			scales := make([]int32, rows)
			for row := range rows {
				code := row % n
				if f != FP8Activation32 {
					code |= code << 4
				}
				for j := range packedCols {
					data[row*packedCols+j] = int32(code)
				}
				scales[row] = int32(scaleBytes[row/n])
			}
			err := mlx.Batch(func() error {
				s := &activationScope{}
				defer func() { mlx.CloseArrays(s.arrays) }()
				d := s.add(mlx.AsType(s.add(mlx.NewInt32(data, []int{rows, packedCols})), mlx.UInt8))
				sc := s.add(mlx.AsType(s.add(mlx.NewInt32(scales, []int{rows, 1})), mlx.UInt8))
				if s.err != nil {
					return s.err
				}
				for _, r := range []Rounding{Float32, BFloat16} {
					y, err := DequantizeActivationArray(d, sc, f, r)
					if err != nil {
						return err
					}
					got, err := y.Float32Data()
					y.Close()
					if err != nil {
						return err
					}
					for row := range rows {
						packed := make([]byte, packedCols)
						for j := range packed {
							packed[j] = byte(data[row*packedCols+j])
						}
						want := make([]float32, group)
						err := DequantizeActivation(want, packed, []byte{byte(scales[row])}, 1, group, f, r)
						for j, w := range want {
							g := got[row*group+j]
							if err != nil {
								if !math.IsNaN(float64(g)) {
									return fmt.Errorf("format %d row %d rounding %d: invalid code decoded to %v", f, row, r, g)
								}
							} else if math.Float32bits(g) != math.Float32bits(w) {
								return fmt.Errorf("format %d row %d rounding %d: %08x != %08x", f, row, r, math.Float32bits(g), math.Float32bits(w))
							}
						}
					}
				}
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
		}
	}
}

func checkActivationArray(c activationCase, x []float32, format ActivationFormat) error {
	return checkActivationArrayWith(c, x, format, QuantizeActivationArray)
}

func checkActivationArrayWith(c activationCase, x []float32, format ActivationFormat, encode func(mlx.Array, ActivationFormat) (mlx.Array, mlx.Array, error)) error {
	// Deliberately separate graph construction and evaluation dispatches so
	// concurrent tests exercise ordinary unpinned caller goroutines.
	a, err := mlx.NewFloat32(x, []int{c.Rows, c.Cols})
	if err != nil {
		return err
	}
	data, scales, err := encode(a, format)
	a.Close() // Packed graphs must retain their input dependencies.
	if err != nil {
		return err
	}
	defer data.Close()
	defer scales.Close()
	for _, part := range []struct {
		name string
		a    mlx.Array
		want []byte
	}{
		{"data", data, c.Data}, {"scales", scales, c.Scales},
	} {
		v, err := mlx.AsType(part.a, mlx.Int32)
		if err != nil {
			return err
		}
		got, err := v.Int32Data()
		v.Close()
		if err != nil {
			return err
		}
		if len(got) != len(part.want) {
			return fmt.Errorf("%s length %d != %d", part.name, len(got), len(part.want))
		}
		for i, g := range got {
			if g != int32(part.want[i]) {
				return fmt.Errorf("%s[%d] = %02x, want %02x", part.name, i, g, part.want[i])
			}
		}
	}
	for _, r := range []Rounding{Float32, BFloat16} {
		y, err := DequantizeActivationArray(data, scales, format, r)
		if err != nil {
			return err
		}
		got, err := y.Float32Data()
		y.Close()
		if err != nil {
			return err
		}
		want := c.Expected
		if r == BFloat16 {
			want = c.BF16
		}
		if len(got) != len(want) {
			return fmt.Errorf("reconstruction length")
		}
		for i, v := range got {
			if math.Float32bits(v) != want[i] {
				return fmt.Errorf("rounding %d value[%d] = %08x, want %08x", r, i, math.Float32bits(v), want[i])
			}
		}
	}
	return nil
}
