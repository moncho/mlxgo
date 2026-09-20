//go:build !mlx

package checkpoint

import (
	"strings"
	"sync"
	"testing"
)

func TestStubInspectionAndGet(t *testing.T) {
	r, err := Open(twoShards(t))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if _, err = r.Get("a"); err == nil || !strings.Contains(err.Error(), "-tags mlx") {
		t.Fatal("stub pretended to load native tensor", err)
	}
}

func TestStubConcurrentGetAndClose(t *testing.T) {
	r, err := Open(twoShards(t))
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			for range 100 {
				_, _ = r.Get("a")
				_ = r.Close()
			}
		})
	}
	wg.Wait()
}
