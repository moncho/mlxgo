package deepseek

import (
	"fmt"
	"math"
	"math/big"
	"slices"

	mlx "github.com/moncho/mlxgo"
)

// EngramConfig stores prepared hashing metadata from the pinned reference.
// TokenMap contains normalized/compressed IDs for every original token; PadID
// is an ORIGINAL token ID. Primes are flattened [ngram-2,head] within each layer.
// Multipliers are the exact NumPy-generated odd int64 values, not Go RNG values.
// This is an experimental bundle contract, not the released checkpoint schema.
type EngramConfig struct {
	Layers              []int     `json:"layers"`
	MaxNGram            int       `json:"max_ngram"`
	Heads               int       `json:"heads"`
	HeadDim             int       `json:"head_dim"`
	Rows                []int     `json:"rows"`
	Primes              [][]int64 `json:"primes"`
	Multipliers         [][]int64 `json:"multipliers"`
	TokenMap            []int32   `json:"token_map"`
	CompressedVocabSize int       `json:"compressed_vocab_size"`
	PadID               int32     `json:"pad_id"`
}

func (c EngramConfig) columns() int { return (c.MaxNGram - 1) * c.Heads }

// Validate rejects ambiguous layouts and multiplication/index overflow before
// any native allocation. Metadata must be exported with the matching tokenizer.
func (c EngramConfig) Validate() error {
	if len(c.Layers) < 1 || len(c.Layers) > 128 || c.MaxNGram < 2 || c.MaxNGram > 16 || c.Heads < 1 || c.Heads > 64 || c.HeadDim < 1 || c.HeadDim > 1<<20 {
		return fmt.Errorf("deepseek: invalid Engram dimensions")
	}
	if len(c.Rows) != len(c.Layers) || len(c.Primes) != len(c.Layers) || len(c.Multipliers) != len(c.Layers) {
		return fmt.Errorf("deepseek: Engram requires one table, bucket layout and multiplier row per layer")
	}
	if len(c.TokenMap) == 0 || c.CompressedVocabSize < 1 || c.CompressedVocabSize > len(c.TokenMap) || c.PadID < 0 || int(c.PadID) >= len(c.TokenMap) {
		return fmt.Errorf("deepseek: invalid Engram token map or padding ID")
	}
	seenIDs := make([]bool, c.CompressedVocabSize)
	for _, id := range c.TokenMap {
		if id < 0 || int(id) >= c.CompressedVocabSize {
			return fmt.Errorf("deepseek: Engram compressed ID out of range")
		}
		seenIDs[id] = true
	}
	for _, seen := range seenIDs {
		if !seen {
			return fmt.Errorf("deepseek: Engram compressed vocabulary must be dense")
		}
	}
	seenPrimes := make(map[int64]bool)
	for i, layer := range c.Layers {
		if layer < 0 || layer >= 128 || (i > 0 && c.Layers[i-1] >= layer) || c.Rows[i] < 1 || c.Rows[i] > math.MaxInt32 {
			return fmt.Errorf("deepseek: invalid Engram layer or table size")
		}
		if len(c.Primes[i]) != c.columns() || len(c.Multipliers[i]) != c.MaxNGram {
			return fmt.Errorf("deepseek: invalid Engram hash layout")
		}
		var total int64
		for _, p := range c.Primes[i] {
			if p < 2 || p > math.MaxInt32 || seenPrimes[p] || !big.NewInt(p).ProbablyPrime(0) {
				return fmt.Errorf("deepseek: Engram bucket sizes must be distinct primes")
			}
			seenPrimes[p] = true
			total += p
		}
		if total > int64(c.Rows[i]) {
			return fmt.Errorf("deepseek: Engram buckets exceed table rows")
		}
		for _, m := range c.Multipliers[i] {
			if m <= 0 || m%2 == 0 || m > math.MaxInt64/int64(c.CompressedVocabSize) {
				return fmt.Errorf("deepseek: invalid or overflowing Engram multiplier")
			}
		}
	}
	return nil
}

func (c EngramConfig) clone() EngramConfig {
	c.Layers = slices.Clone(c.Layers)
	c.Rows = slices.Clone(c.Rows)
	c.TokenMap = slices.Clone(c.TokenMap)
	c.Primes = slices.Clone(c.Primes)
	c.Multipliers = slices.Clone(c.Multipliers)
	for i := range c.Primes {
		c.Primes[i] = slices.Clone(c.Primes[i])
	}
	for i := range c.Multipliers {
		c.Multipliers[i] = slices.Clone(c.Multipliers[i])
	}
	return c
}

// EngramHasher holds one sequence's bounded token history. It remaps original
// IDs before hashing. A false mask entry blocks lookback across that position,
// including across chunk boundaries. Use separate hashers for separate sequences;
// do not share one across goroutines. No MLX runtime is required.
type EngramHasher struct {
	config   EngramConfig
	history  []int32
	position int
}

// NewEngramHasher validates and copies metadata; later caller edits cannot
// change the mapping of an existing sequence.
func NewEngramHasher(c EngramConfig) (*EngramHasher, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	return &EngramHasher{config: c.clone()}, nil
}

// Position counts consumed positions, including masked positions.
func (h *EngramHasher) Position() int {
	if h == nil {
		return 0
	}
	return h.position
}
// Reset drops history and starts a new sequence with the same metadata.
func (h *EngramHasher) Reset() {
	if h != nil {
		h.history = nil
		h.position = 0
	}
}

// Hash consumes a nonempty chunk and returns row-major [tokens,layers,columns]
// table indices. A nil mask means all tokens are text. Invalid input leaves
// history unchanged. Constants and token normalization are supplied by config;
// hashing itself is exact integer arithmetic with no floating-point conversion.
func (h *EngramHasher) Hash(tokens []int32, mask []bool) ([]int32, error) {
	if h == nil || len(h.config.Layers) == 0 {
		return nil, fmt.Errorf("deepseek: uninitialized Engram hasher")
	}
	c := h.config
	if len(tokens) == 0 || (mask != nil && len(mask) != len(tokens)) || len(tokens) > math.MaxInt-h.position || len(tokens) > math.MaxInt/len(c.Layers)/c.columns() {
		return nil, fmt.Errorf("deepseek: invalid Engram token/mask length")
	}
	for _, id := range tokens {
		if id < 0 || int(id) >= len(c.TokenMap) {
			return nil, fmt.Errorf("deepseek: Engram token %d out of range", id)
		}
	}
	history := make([]int32, len(h.history), len(h.history)+len(tokens))
	copy(history, h.history)
	for i, id := range tokens {
		v := c.TokenMap[id]
		if mask != nil && !mask[i] {
			v = -1
		}
		history = append(history, v)
	}
	out := make([]int32, len(tokens)*len(c.Layers)*c.columns())
	lookback := make([]int64, c.MaxNGram)
	for i := range tokens {
		blocked := false
		for shift := range lookback {
			p := len(h.history) + i - shift
			blocked = blocked || p < 0 || (p >= 0 && history[p] == -1)
			v := c.TokenMap[c.PadID]
			if !blocked {
				v = history[p]
			}
			lookback[shift] = int64(v)
		}
		for layer := range c.Layers {
			rolling := lookback[0] * c.Multipliers[layer][0]
			offset := int64(0)
			for shift := 1; shift < c.MaxNGram; shift++ {
				rolling ^= lookback[shift] * c.Multipliers[layer][shift]
				for head := 0; head < c.Heads; head++ {
					col := (shift-1)*c.Heads + head
					p := c.Primes[layer][col]
					out[(i*len(c.Layers)+layer)*c.columns()+col] = int32(offset + rolling%p)
					offset += p
				}
			}
		}
	}
	h.history = slices.Clone(history[max(0, len(history)-(c.MaxNGram-1)):])
	h.position += len(tokens)
	return out, nil
}

// EngramWeights uses float32 tables and [output,input] projection weights.
// Arrays are borrowed. The released FP8 table format is not supported here.
type EngramWeights struct{ Embedding, Projection, QueryNorm, KeyNorm mlx.Array }

// Engram adds a conditional n-gram memory contribution to x [batch,seq,hc,dim].
// Hash IDs are flattened [batch,seq,columns]. A nil mask means all positions are
// text; false entries pass x through unchanged. Gradients flow through x and all
// weight arrays, not through discrete token hashing. The result is caller-owned.
func Engram(x mlx.Array, hashIDs []int32, mask []bool, w EngramWeights, epsilon float32) (mlx.Array, error) {
	return one(func(s *scope) mlx.Array {
		s.check(x, "Engram residual", -1, -1, -1, -1)
		s.check(w.Embedding, "Engram table", -1, -1)
		if s.err != nil {
			return mlx.Array{}
		}
		shape, table := x.Shape(), w.Embedding.Shape()
		n, hc, dim := shape[0]*shape[1], shape[2], shape[3]
		if len(hashIDs) == 0 || len(hashIDs)%n != 0 || (mask != nil && len(mask) != n) || !positive(epsilon) {
			s.err = fmt.Errorf("deepseek: invalid Engram indices, mask or epsilon")
			return mlx.Array{}
		}
		cols := len(hashIDs) / n
		s.check(w.Projection, "Engram projection", (hc+1)*dim, cols*table[1])
		s.check(w.QueryNorm, "Engram query norm", hc, dim)
		s.check(w.KeyNorm, "Engram key norm", hc, dim)
		for _, id := range hashIDs {
			if id < 0 || int(id) >= table[0] {
				s.err = fmt.Errorf("deepseek: Engram table index out of range")
				break
			}
		}
		if s.err != nil {
			return mlx.Array{}
		}
		ids := s.add(mlx.NewInt32(hashIDs, []int{shape[0], shape[1], cols}))
		values := s.add(mlx.TakeAxis(w.Embedding, ids, 0))
		values = s.add(mlx.Reshape(values, []int{shape[0], shape[1], cols * table[1]}))
		kv := s.linear(values, w.Projection)
		key := s.add(mlx.Reshape(s.span(kv, -1, 0, hc*dim), shape))
		value := s.add(mlx.ExpandDims(s.span(kv, -1, hc*dim, (hc+1)*dim), -2))
		rstd := func(a mlx.Array) mlx.Array {
			variance := s.add(mlx.Add(s.add(mlx.MeanAxis(s.add(mlx.Square(a)), -1, false)), s.scalar(epsilon)))
			return s.add(mlx.Divide(s.scalar(1), s.add(mlx.Sqrt(variance))))
		}
		weight := s.add(mlx.Multiply(w.QueryNorm, w.KeyNorm))
		dot := s.add(mlx.SumAxis(s.add(mlx.Multiply(s.add(mlx.Multiply(x, weight)), key)), -1, false))
		dot = s.add(mlx.Multiply(dot, s.add(mlx.Multiply(rstd(x), rstd(key)))))
		dot = s.add(mlx.Multiply(dot, s.scalar(float32(1/math.Sqrt(float64(dim))))))
		magnitude := s.add(mlx.Sqrt(s.add(mlx.Maximum(s.add(mlx.Abs(dot)), s.scalar(1e-6)))))
		signed := s.add(mlx.Where(s.add(mlx.Less(dot, s.scalar(0))), s.add(mlx.Negative(magnitude)), magnitude))
		gate := s.add(mlx.Sigmoid(signed))
		if mask != nil {
			flags := make([]int32, n)
			for i, v := range mask {
				if v {
					flags[i] = 1
				}
			}
			keep := s.add(mlx.NewInt32(flags, []int{shape[0], shape[1], 1}))
			gate = s.add(mlx.Where(keep, gate, s.scalar(0)))
		}
		return s.add(mlx.Add(x, s.add(mlx.Multiply(s.add(mlx.ExpandDims(gate, -1)), value))))
	})
}
