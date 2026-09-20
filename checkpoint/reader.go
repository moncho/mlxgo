package checkpoint

import (
	"errors"
	"fmt"
	"sync"

	mlx "github.com/moncho/mlxgo"
)

var ErrClosed = errors.New("checkpoint: reader is closed")

// Reader routes tensor names to shards. It holds at most one native shard map
// at a time. Returned arrays are caller-owned and survive switching shards or
// closing the reader. Native lifetime is serialized by the MLX worker.
type Reader struct {
	mu       sync.Mutex
	manifest *Manifest
	current  string
	file     *mlx.SafeTensors
	closed   bool
}

// Open validates all metadata but defers native shard loading until Get.
func Open(dir string) (*Reader, error) {
	m, err := Inspect(dir)
	if err != nil {
		return nil, err
	}
	return &Reader{manifest: m}, nil
}

func (r *Reader) Get(name string) (out mlx.Array, err error) {
	err = mlx.Batch(func() error {
		if r == nil {
			return ErrClosed
		}
		// Lock inside Batch so waiting callers cannot block the MLX worker.
		// The mutex also protects the stub build, whose Batch runs inline.
		r.mu.Lock()
		defer r.mu.Unlock()
		if r.closed || r.manifest == nil {
			return ErrClosed
		}
		t, ok := r.manifest.tensors[name]
		if !ok {
			return fmt.Errorf("checkpoint: missing tensor %q", name)
		}
		if e := r.manifest.unchanged(t.File); e != nil {
			return e
		}
		if r.file == nil || r.current != t.File {
			if r.file != nil {
				if e := r.file.Close(); e != nil {
					return e
				}
				r.file = nil
				r.current = ""
			}
			f, e := mlx.LoadSafetensors(r.manifest.paths[t.File])
			if e != nil {
				return fmt.Errorf("checkpoint: %s: %w", t.File, e)
			}
			r.file = f
			r.current = t.File
		}
		var e error
		out, e = r.file.Get(name)
		return e
	})
	return out, err
}

func (r *Reader) Close() error {
	if r == nil {
		return nil
	}
	return mlx.Batch(func() error {
		r.mu.Lock()
		defer r.mu.Unlock()
		if r.closed {
			return nil
		}
		r.closed = true
		if r.file != nil {
			err := r.file.Close()
			r.file = nil
			return err
		}
		return nil
	})
}
