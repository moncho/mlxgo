//go:build mlx

package mlx

/*
#cgo darwin,arm64 CFLAGS: -I/opt/homebrew/include
#include <stdlib.h>
#include <stdint.h>
#include <mlx/c/mlx.h>
*/
import "C"

import (
	"fmt"
	"unsafe"
)

//export mlxgoClosureApply
func mlxgoClosureApply(res *C.mlx_vector_array, inputs C.mlx_vector_array, payload unsafe.Pointer) C.int {
	state := lookupClosurePayload(payload)
	if state == nil {
		return 1
	}

	goInputs, err := vectorToArrays(inputs, false)
	if err != nil {
		state.setErr(err)
		return 1
	}
	for i := range goInputs {
		defer goInputs[i].Close()
	}

	outputs, err := state.fn(goInputs)
	if err != nil {
		state.setErr(err)
		return 1
	}
	if len(outputs) == 0 {
		state.setErr(fmt.Errorf("mlxgo: closure returned no outputs"))
		return 1
	}

	if err := setArrayVector(res, outputs); err != nil {
		state.setErr(err)
		return 1
	}

	for i := range outputs {
		if outputs[i].state != nil && outputs[i].state.owned {
			_ = outputs[i].Close()
		}
	}
	return 0
}

//export mlxgoClosureFree
func mlxgoClosureFree(payload unsafe.Pointer) {
	freeClosurePayload(payload)
}
