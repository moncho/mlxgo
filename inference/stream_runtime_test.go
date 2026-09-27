//go:build mlx && mlxruntime

package inference

import (
	"errors"
	"slices"
	"strings"
	"testing"

	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/lm"
)

func TestTextStreaming(t *testing.T) {
	if err := mlx.SetDefaultGPU(); err != nil {
		t.Fatal(err)
	}
	dir, _ := exportQwenFixture(t)
	m, err := Open(dir, Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	for _, o := range []lm.SamplingOptions{{}, {Temperature: .8, TopP: .9, Seed: 17}} {
		want, err := m.GenerateWith("hello", 4, o)
		if err != nil {
			t.Fatal(err)
		}
		var text strings.Builder
		got, err := m.Stream("hello", 4, o, func(s string) error { text.WriteString(s); return nil })
		if err != nil || !slices.Equal(got.Tokens, want.Tokens) || got.Text != want.Text || text.String() != want.Text {
			t.Fatalf("stream=%+v text=%q want=%+v err=%v", got, text.String(), want, err)
		}
	}
	m.decode = func(ids []int32) string { return strings.Repeat("x", len(ids)) }
	m.eos = nil
	calls := 0
	stop := errors.New("consumer stopped")
	r, err := m.Stream("hello", 4, lm.SamplingOptions{}, func(string) error {
		calls++
		ch := make(chan error, 1)
		go func() {
			a, e := mlx.NewScalarFloat32(1)
			if e == nil {
				e = a.Close()
			}
			ch <- e
		}()
		if e := <-ch; e != nil {
			return e
		}
		if calls == 2 {
			return stop
		}
		return nil
	})
	if !errors.Is(err, stop) || calls != 2 || len(r.Tokens) != 2 || r.Text != "xx" {
		t.Fatalf("partial=%+v calls=%d err=%v", r, calls, err)
	}
}
