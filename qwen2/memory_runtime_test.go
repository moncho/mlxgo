//go:build mlx && mlxruntime && darwin

package qwen2

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"github.com/moncho/mlxgo"
)

func TestQwenMemoryPlateau(t *testing.T) {
	dir := os.Getenv("MLXGO_QWEN2_DIR")
	if dir == "" || os.Getenv("MLXGO_QWEN2_MEMORY") != "1" {
		t.Skip("set MLXGO_QWEN2_DIR and MLXGO_QWEN2_MEMORY=1 for 200-token RSS regression")
	}
	if err := mlx.SetDefaultGPU(); err != nil {
		t.Fatal(err)
	}
	c, err := LoadConfig(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	w, err := Load(dir, c)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	rss := func() int64 {
		t.Helper()
		runtime.GC()
		out, err := exec.Command("/bin/ps", "-o", "rss=", "-p", strconv.Itoa(os.Getpid())).Output()
		if err != nil {
			t.Fatal(err)
		}
		kb, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		return kb * 1024
	}
	decode := func() {
		t.Helper()
		cache := NewKVCache(c.NumLayers)
		defer cache.Close()
		ids := []int32{151644, 872, 198, 785}
		for range 200 {
			logits, err := Forward(w, c, ids, cache)
			if err != nil {
				t.Fatal(err)
			}
			next, err := mlx.ArgmaxAxis(logits, -1, false)
			if err != nil {
				t.Fatal(err)
			}
			if err := mlx.Eval(append(cache.Arrays(), next)...); err != nil {
				t.Fatal(err)
			}
			data, err := next.UInt32Data()
			if err != nil {
				t.Fatal(err)
			}
			ids = []int32{int32(data[0])}
			_ = mlx.CloseArrays([]mlx.Array{logits, next})
		}
	}
	decode() // Warm allocator and kernels before measuring retained memory.
	baseline := rss()
	for round := 1; round <= 2; round++ {
		decode()
		after := rss()
		t.Logf("round %d RSS: %.1f -> %.1f MiB", round, float64(baseline)/(1<<20), float64(after)/(1<<20))
		if after-baseline > 64<<20 {
			t.Fatal(fmt.Sprintf("RSS retained more than 64 MiB beyond warm baseline: %d bytes", after-baseline))
		}
	}
}
