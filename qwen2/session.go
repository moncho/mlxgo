package qwen2

import (
	"fmt"
	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/lm"
)

// Session adapts the existing Qwen implementation to the shared inference API.
// Weights are borrowed and must remain open for the session's lifetime.
type Session struct {
	weights *Weights
	config  Config
	cache   *KVCache
	closed  bool
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
	return Forward(s.weights, s.config, tokens, s.cache)
}
func (s *Session) Close() error {
	if s == nil || s.closed {
		return nil
	}
	s.closed = true
	return s.cache.Close()
}
