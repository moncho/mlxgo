package inference

import (
	"fmt"
	"time"

	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/lm"
)

// Result contains generated output and forward-pass timing, including worker
// queue time but excluding token selection, text encoding and decoding.
type Result struct {
	Text                          string
	Tokens                        []int32
	PrefillTokens                 int
	PrefillSeconds, DecodeSeconds float64
}

// Generate uses the selected architecture's text encoding and a common greedy
// decoder. Qwen EOS tokens are omitted from Result.Tokens and Text.
func (m *Model) Generate(prompt string, maxTokens int) (Result, error) {
	return m.GenerateWith(prompt, maxTokens, lm.SamplingOptions{})
}

// GenerateWith generates text with optional temperature and nucleus sampling.
func (m *Model) GenerateWith(prompt string, maxTokens int, o lm.SamplingOptions) (Result, error) {
	if err := o.Validate(); err != nil {
		return Result{}, err
	}
	if m == nil {
		return Result{}, ErrClosed
	}
	if !m.info.Text {
		return Result{}, ErrTextUnsupported
	}
	if maxTokens <= 0 {
		return Result{}, fmt.Errorf("inference: max tokens must be positive")
	}
	ids := m.encode(prompt)
	r, err := m.generate(ids, maxTokens, o, m.eos)
	if err != nil {
		return r, err
	}
	if len(r.Tokens) > 0 {
		for _, id := range m.eos {
			if r.Tokens[len(r.Tokens)-1] == id {
				r.Tokens = r.Tokens[:len(r.Tokens)-1]
				break
			}
		}
	}
	r.Text = m.decode(r.Tokens)
	return r, nil
}

// GenerateTokens works for every supported architecture without a tokenizer.
// Returned tokens include any matching stop token. No text is decoded.
func (m *Model) GenerateTokens(prompt []int32, maxTokens int, stop ...int32) (Result, error) {
	return m.GenerateTokensWith(prompt, maxTokens, lm.SamplingOptions{}, stop...)
}

// GenerateTokensWith samples raw IDs, including a matching stop token.
func (m *Model) GenerateTokensWith(prompt []int32, maxTokens int, o lm.SamplingOptions, stop ...int32) (Result, error) {
	return m.generate(prompt, maxTokens, o, stop)
}

func (m *Model) generate(prompt []int32, maxTokens int, o lm.SamplingOptions, stop []int32) (r Result, err error) {
	if err := o.Validate(); err != nil {
		return r, err
	}
	if m == nil {
		return r, ErrClosed
	}
	if len(prompt) == 0 || maxTokens <= 0 {
		return r, fmt.Errorf("inference: nonempty prompt and positive max tokens required")
	}
	if len(prompt) > m.info.MaxPositions || maxTokens > m.info.MaxPositions-len(prompt)+1 {
		return r, fmt.Errorf("inference: prompt and generation exceed %d context positions", m.info.MaxPositions)
	}
	for _, id := range append(append([]int32{}, prompt...), stop...) {
		if id < 0 || int(id) >= m.info.VocabSize {
			return r, fmt.Errorf("inference: token %d outside vocabulary", id)
		}
	}
	s, err := m.NewSession()
	if err != nil {
		return r, err
	}
	defer s.Close()
	r.PrefillTokens = len(prompt)
	timed := &timedSession{Session: s, result: &r}
	r.Tokens, err = lm.Sample(timed, prompt, maxTokens, o, stop...)
	return r, err
}

type timedSession struct {
	lm.Session
	result *Result
	steps  int
}

func (s *timedSession) Step(tokens []int32) (mlx.Array, error) {
	start := time.Now()
	a, err := s.Session.Step(tokens)
	if err == nil {
		err = a.Eval()
		if err != nil {
			a.Close()
			a = mlx.Array{}
		}
	}
	if s.steps == 0 {
		s.result.PrefillSeconds += time.Since(start).Seconds()
	} else {
		s.result.DecodeSeconds += time.Since(start).Seconds()
	}
	s.steps++
	return a, err
}
