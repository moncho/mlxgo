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
	return Stream(s, prompt, count, SamplingOptions{}, nil, eos...)
}

// Stream emits on the calling goroutine, outside Batch, and returns partial tokens on error.
func Stream(s Session, prompt []int32, count int, o SamplingOptions, emit func(int32) error, eos ...int32) ([]int32, error) {
	if err := o.Validate(); err != nil {
		return nil, err
	}
	if s == nil || len(prompt) == 0 || count < 0 {
		return nil, fmt.Errorf("lm: nonempty prompt, session and nonnegative count required")
	}
	if count == 0 {
		return []int32{}, nil
	}
	pick := greedyPick
	if o.Temperature > 0 {
		if err := mlx.RandomSeed(o.Seed); err != nil {
			return nil, err
		}
		pick = func(a mlx.Array) (int32, error) { return sampledPick(a, o) }
	}
	var result []int32
	input := prompt
	for i := 0; i < count; i++ {
		logits, err := s.Step(input)
		if err != nil {
			return result, err
		}
		shape := logits.Shape()
		if len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] < 1 {
			logits.Close()
			return result, fmt.Errorf("lm: expected logits [1,1,vocabulary], got %v", shape)
		}
		token, err := pick(logits)
		logits.Close()
		if err != nil {
			return result, err
		}
		result = append(result, token)
		if emit != nil {
			if err := emit(token); err != nil {
				return result, fmt.Errorf("lm: emit token: %w", err)
			}
		}
		for _, stop := range eos {
			if token == stop {
				return result, nil
			}
		}
		input = []int32{token}
	}
	return result, nil
}

func greedyPick(logits mlx.Array) (int32, error) {
	best, err := mlx.ArgmaxAxis(logits, -1, false)
	if err != nil {
		return 0, err
	}
	defer best.Close()
	return tokenID(best)
}

func tokenID(a mlx.Array) (int32, error) {
	ids, err := mlx.AsType(a, mlx.Int32)
	if err != nil {
		return 0, err
	}
	defer ids.Close()
	data, err := ids.Int32Data()
	if err != nil {
		return 0, err
	}
	if len(data) != 1 {
		return 0, fmt.Errorf("lm: expected one token")
	}
	return data[0], nil
}
