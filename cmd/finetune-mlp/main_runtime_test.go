//go:build mlx && mlxruntime

package main

import (
	"bytes"
	"testing"

	"github.com/moncho/mlxgo"
)

func TestFineTuneEndToEnd(t *testing.T) {
	defer mlx.SetDefaultGPU()
	for _, device := range []string{"cpu", "gpu"} {
		t.Run(device, func(t *testing.T) {
			var log bytes.Buffer
			if err := run(t.TempDir(), device, &log); err != nil {
				t.Fatalf("%v\n%s", err, log.String())
			}
			t.Log(log.String())
		})
	}
}
