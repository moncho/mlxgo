package lm

import (
	"fmt"
	"math"
	"slices"

	mlx "github.com/moncho/mlxgo"
)

type choiceNode struct {
	edges    map[int32]*choiceNode
	allowed  []int32
	terminal bool
}

// Choices is an immutable trie of complete allowed token sequences. It may be
// reused concurrently, but each generation still needs its own Session.
type Choices struct {
	root     *choiceNode
	maxLen   int
	maxToken int32
}

// NewChoices copies nonempty, nonnegative token sequences. Include a stop token
// in every sequence when termination is model-selected. Duplicate sequences and
// sequences that are prefixes of others are rejected; appending EOS resolves
// ordinary text-prefix labels such as "card" and "card_arrival".
func NewChoices(sequences [][]int32) (*Choices, error) {
	if len(sequences) == 0 {
		return nil, fmt.Errorf("lm: at least one choice required")
	}
	c := &Choices{root: &choiceNode{edges: map[int32]*choiceNode{}}}
	for _, sequence := range sequences {
		if len(sequence) == 0 {
			return nil, fmt.Errorf("lm: empty choice")
		}
		node := c.root
		for _, token := range sequence {
			if token < 0 || node.terminal {
				return nil, fmt.Errorf("lm: negative token or prefix-ambiguous choices")
			}
			if node.edges[token] == nil {
				node.edges[token] = &choiceNode{edges: map[int32]*choiceNode{}}
				node.allowed = append(node.allowed, token)
				slices.Sort(node.allowed)
			}
			node = node.edges[token]
			c.maxToken = max(c.maxToken, token)
		}
		if node.terminal || len(node.edges) != 0 {
			return nil, fmt.Errorf("lm: duplicate or prefix-ambiguous choices")
		}
		node.terminal = true
		c.maxLen = max(c.maxLen, len(sequence))
	}
	return c, nil
}

// Greedy selects the highest-logit allowed next token until a complete choice
// is reached. It does not maximize whole-sequence likelihood. Ties choose the
// lowest token ID. The cap must fit the longest choice; successful calls always
// return an exact choice, including its stop token. Errors may return a partial
// sequence, which must not be treated as a valid label. The session remains
// caller-owned; the last returned token has not been consumed by it.
func (c *Choices) Greedy(s Session, prompt []int32, cap int) ([]int32, error) {
	if c == nil || c.root == nil || s == nil || len(prompt) == 0 || cap < c.maxLen {
		return nil, fmt.Errorf("lm: choices, session, prompt, and cap fitting every choice required")
	}
	node, input := c.root, prompt
	var result []int32
	for !node.terminal {
		logits, err := s.Step(input)
		if err != nil {
			return result, err
		}
		token, err := constrainedPick(logits, node.allowed, c.maxToken)
		_ = logits.Close()
		if err != nil {
			return result, err
		}
		result = append(result, token)
		node = node.edges[token]
		input = []int32{token}
	}
	return result, nil
}

func constrainedPick(logits mlx.Array, allowed []int32, maxToken int32) (token int32, err error) {
	err = mlx.Batch(func() error {
		shape := logits.Shape()
		if len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] <= int(maxToken) {
			return fmt.Errorf("lm: expected logits [1,1,vocabulary] containing every choice token, got %v", shape)
		}
		dtype, err := logits.DType()
		if err != nil {
			return err
		}
		if dtype != mlx.Float16 && dtype != mlx.BFloat16 && dtype != mlx.Float32 && dtype != mlx.Float64 {
			return fmt.Errorf("lm: floating logits required")
		}
		s := &scope{}
		defer func() { _ = mlx.CloseArrays(s.arrays) }()
		indices := s.add(mlx.NewInt32(allowed, []int{len(allowed)}))
		x := s.add(mlx.TakeAxis(logits, indices, -1))
		if s.err != nil {
			return s.err
		}
		var values []float64
		if dtype == mlx.Float64 {
			values, err = x.Float64Data()
		} else {
			x = s.add(mlx.AsType(x, mlx.Float32))
			if s.err != nil {
				return s.err
			}
			var floats []float32
			floats, err = x.Float32Data()
			for _, value := range floats {
				values = append(values, float64(value))
			}
		}
		if err != nil {
			return err
		}
		best := -1
		for i, value := range values {
			if math.IsNaN(value) || math.IsInf(value, 1) {
				return fmt.Errorf("lm: nonfinite allowed logit")
			}
			if !math.IsInf(value, -1) && (best < 0 || value > values[best]) {
				best = i
			}
		}
		if best < 0 {
			return fmt.Errorf("lm: no finite allowed logit")
		}
		token = allowed[best]
		return nil
	})
	return token, err
}
