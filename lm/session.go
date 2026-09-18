// Package lm defines the small inference contract shared by architecture
// adapters. Tokenization, checkpoint loading, and cache layouts stay with each
// architecture. Returned arrays are always caller-owned.
package lm

import (
	"fmt"
	mlx "github.com/moncho/mlxgo"
)

// Session owns the incremental state for one sequence. Step consumes tokens and
// returns float logits [1,1,vocabulary] for the last token. Architectures may
// restrict post-prefill calls to one token. Do not share a session across
// goroutines; use separate sessions. Close releases caches, not model weights.
type Session interface {
	Step(tokens []int32) (mlx.Array, error)
	Position() int
	Close() error
}

// Greedy returns at most count new token IDs, stopping after an EOS ID. It uses
// an existing session, which remains caller-owned; tokenization is external.
// The last returned token has not been consumed by the session.
func Greedy(s Session, prompt []int32, count int, eos ...int32) ([]int32, error) {
	if s == nil || len(prompt) == 0 || count < 0 {
		return nil, fmt.Errorf("lm: nonempty prompt, session and nonnegative count required")
	}
	if count == 0 {
		return []int32{}, nil
	}
	var result []int32
	input := prompt
	for i := 0; i < count; i++ {
		logits, err := s.Step(input)
		if err != nil {
			return nil, err
		}
		shape := logits.Shape()
		if len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] < 1 {
			logits.Close()
			return nil, fmt.Errorf("lm: expected logits [1,1,vocabulary], got %v", shape)
		}
		best, err := mlx.ArgmaxAxis(logits, -1, false)
		logits.Close()
		if err != nil {
			return nil, err
		}
		ids, err := mlx.AsType(best, mlx.Int32)
		best.Close()
		if err != nil {
			return nil, err
		}
		data, err := ids.Int32Data()
		ids.Close()
		if err != nil {
			return nil, err
		}
		if len(data) != 1 {
			return nil, fmt.Errorf("lm: expected one token")
		}
		token := data[0]
		result = append(result, token)
		for _, stop := range eos {
			if token == stop {
				return result, nil
			}
		}
		input = []int32{token}
	}
	return result, nil
}
