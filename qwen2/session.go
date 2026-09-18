package qwen2

import (
	"encoding/json"
	"fmt"

	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/lm"
)

// Session adapts the existing Qwen implementation to the shared inference API.
// Weights are borrowed and must remain open for the session's lifetime.
// Step evaluates logits and caches before returning. Use separate sessions for
// separate sequences; do not share a session across goroutines.
type Session struct {
	weights  *Weights
	config   Config
	cache    *KVCache
	closed   bool
	adapters *Adapters
}

// NewSessionWithAdapters borrows validated adapters and base weights. Both must
// remain open until the session closes; only cache state is owned here.
func NewSessionWithAdapters(w *Weights, c Config, a *Adapters) (*Session, error) {
	if err := a.validate(); err != nil {
		return nil, err
	}
	want, _ := json.Marshal(c)
	got, _ := json.Marshal(a.config)
	if string(want) != string(got) {
		return nil, fmt.Errorf("qwen2: adapter configuration mismatch")
	}
	s, err := NewSession(w, c)
	if err != nil {
		return nil, err
	}
	s.adapters = a
	return s, nil
}

var _ lm.Session = (*Session)(nil)

func NewSession(w *Weights, c Config) (*Session, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	if w == nil || w.closed || len(w.Layers) != c.NumLayers {
		return nil, fmt.Errorf("qwen2: closed or incompatible weights")
	}
	return &Session{weights: w, config: c, cache: NewKVCache(c.NumLayers)}, nil
}
func (s *Session) Position() int {
	if s == nil || s.cache == nil {
		return 0
	}
	return s.cache.Offset
}
func (s *Session) Step(tokens []int32) (mlx.Array, error) {
	if s == nil || s.closed {
		return mlx.Array{}, fmt.Errorf("qwen2: closed session")
	}
	var out mlx.Array
	err := mlx.Batch(func() error {
		var err error
		if s.adapters != nil {
			out, err = s.adapters.Forward(s.weights, tokens, s.cache)
		} else {
			out, err = Forward(s.weights, s.config, tokens, s.cache)
		}
		if err != nil {
			return err
		}
		if err = mlx.Eval(append(s.cache.Arrays(), out)...); err != nil {
			out.Close()
			out = mlx.Array{}
			s.cache.invalid = true
		}
		return err
	})
	return out, err
}
func (s *Session) Close() error {
	if s == nil || s.closed {
		return nil
	}
	s.closed = true
	return s.cache.Close()
}
