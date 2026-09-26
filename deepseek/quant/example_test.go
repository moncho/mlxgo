package quant_test

import (
	"fmt"

	"github.com/moncho/mlxgo/deepseek/quant"
)

func ExampleDecode() {
	data := make([]byte, 32)
	for i := range data {
		data[i] = 0x38
	} // E4M3FN 1.0.
	scales := []byte{127} // E8M0 2^(127-127) = 1.
	dst := make([]float32, 32)
	err := quant.Decode(dst, data, scales, 1, 32, quant.FP8Row32, quant.BFloat16)
	fmt.Println(dst[0], dst[31], err)
	// Output: 1 1 <nil>
}

func ExampleQuantizeActivation() {
	src := make([]float32, 16)
	src[0], src[1] = 6, 3
	nd, ns, err := quant.ActivationLayout(1, 16, quant.FP4Cache16)
	if err != nil {
		panic(err)
	}
	data, scales := make([]byte, nd), make([]byte, ns)
	if err := quant.QuantizeActivation(data, scales, src, 1, 16, quant.FP4Cache16); err != nil {
		panic(err)
	}
	dst := make([]float32, 16)
	if err := quant.DequantizeActivation(dst, data, scales, 1, 16, quant.FP4Cache16, quant.BFloat16); err != nil {
		panic(err)
	}
	fmt.Printf("%02x %02x %v\n", data[0], scales[0], dst[:2])
	// Output: 57 38 [6 3]
}
