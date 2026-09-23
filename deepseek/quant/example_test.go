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
