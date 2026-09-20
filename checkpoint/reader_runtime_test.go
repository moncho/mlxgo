//go:build mlx && mlxruntime

package checkpoint

import (
	"errors"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"

	mlx "github.com/moncho/mlxgo"
)

func TestReaderOwnershipAndConcurrency(t *testing.T) {
	if err := mlx.SetDefaultCPU(); err != nil {
		t.Fatal(err)
	}
	r, err := Open(twoShards(t))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	a, err := r.Get("a")
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	b, err := r.Get("b")
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			for range 20 {
				for _, name := range []string{"a", "b"} {
					x, err := r.Get(name)
					if err != nil {
						t.Error(err)
						return
					}
					got, err := x.Float32Data()
					x.Close()
					want := []float32{1, 2}
					if name == "b" {
						want = []float32{3}
					}
					if err != nil || !slices.Equal(got, want) {
						t.Errorf("%s=%v: %v", name, got, err)
						return
					}
				}
			}
		})
	}
	wg.Wait()
	if _, err = r.Get("absent"); err == nil {
		t.Fatal("missing name accepted")
	}
	if err = r.Close(); err != nil {
		t.Fatal(err)
	}
	if err = r.Close(); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		a    mlx.Array
		want []float32
	}{{a, []float32{1, 2}}, {b, []float32{3}}} {
		got, err := tc.a.Float32Data()
		if err != nil || !slices.Equal(got, tc.want) {
			t.Fatal("reader invalidated returned arrays", got, err)
		}
	}
	if _, err = r.Get("a"); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	if _, err = (*Reader)(nil).Get("a"); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
}

func TestReaderDetectsChangedShard(t *testing.T) {
	dir := twoShards(t)
	r, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	path := filepath.Join(dir, "one.safetensors")
	if err = os.Rename(path, path+".old"); err != nil {
		t.Fatal(err)
	}
	safeFile(t, path, `{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}`, floats(1, 2))
	if a, err := r.Get("a"); err == nil {
		a.Close()
		t.Fatal("accepted replacement shard")
	}
}

func TestReaderCloseDuringGet(t *testing.T) {
	r, err := Open(twoShards(t))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	var wg sync.WaitGroup
	wg.Go(func() {
		a, err := r.Get("a")
		if err != nil && !errors.Is(err, ErrClosed) {
			t.Error(err)
		}
		a.Close()
	})
	wg.Go(func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	})
	wg.Wait()
}
