//go:build mlx

package mlx

import (
	"errors"
	"sync"
	"sync/atomic"
	"testing"
)

func TestMLXBuildRunMLXUsesWorkerThread(t *testing.T) {
	const workers = 8
	const iterations = 100

	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				if err := runMLX(func() error {
					if !onMLXThread() {
						return errors.New("not on MLX worker thread")
					}
					return runMLX(func() error {
						if !onMLXThread() {
							return errors.New("reentrant call left MLX worker thread")
						}
						return nil
					})
				}); err != nil {
					errs <- err
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errs)

	for err := range errs {
		t.Fatal(err)
	}
}

func TestMLXBuildRunMLXSerializesCalls(t *testing.T) {
	const workers = 8
	const iterations = 100

	var active atomic.Int32
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				if err := runMLX(func() error {
					current := active.Add(1)
					defer active.Add(-1)
					if current != 1 {
						return errors.New("MLX worker ran calls concurrently")
					}
					return nil
				}); err != nil {
					errs <- err
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errs)

	for err := range errs {
		t.Fatal(err)
	}
}
