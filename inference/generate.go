package inference

import (
	"fmt"
	"slices"
	"strings"
	"time"
	"unicode/utf8"

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
	return m.Stream(prompt, maxTokens, o, nil)
}

// Stream emits text on the calling goroutine; callback errors return partial output.
func (m *Model) Stream(prompt string, maxTokens int, o lm.SamplingOptions, emit func(string) error) (Result, error) {
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
	var tokens []int32
	var text textStream
	var onToken func(int32) error
	if emit != nil {
		onToken = func(id int32) error {
			if slices.Contains(m.eos, id) {
				return nil
			}
			tokens = append(tokens, id)
			return text.update(m.decode(tokens), false, emit)
		}
	}
	r, err := m.generate(ids, maxTokens, o, onToken, m.eos)
	if len(r.Tokens) > 0 && slices.Contains(m.eos, r.Tokens[len(r.Tokens)-1]) {
		r.Tokens = r.Tokens[:len(r.Tokens)-1]
	}
	r.Text = m.decode(r.Tokens)
	if err == nil && emit != nil {
		err = text.update(r.Text, true, emit)
	}
	return r, err
}

type textStream struct{ emitted string }

func (s *textStream) update(decoded string, final bool, emit func(string) error) error {
	if !final {
		decoded = stableText(decoded)
	}
	if !strings.HasPrefix(decoded, s.emitted) {
		return fmt.Errorf("inference: decoded text changed after emission")
	}
	suffix := decoded[len(s.emitted):]
	if suffix == "" {
		return nil
	}
	if err := emit(suffix); err != nil {
		return fmt.Errorf("inference: emit text: %w", err)
	}
	s.emitted = decoded
	return nil
}

func stableText(text string) string {
	for i := 0; i < len(text); {
		if !utf8.FullRuneInString(text[i:]) {
			text = text[:i]
			break
		}
		_, n := utf8.DecodeRuneInString(text[i:])
		i += n
	}
	return strings.TrimRight(text, "\ufffd")
}

// GenerateTokens works for every supported architecture without a tokenizer.
// Returned tokens include any matching stop token. No text is decoded.
func (m *Model) GenerateTokens(prompt []int32, maxTokens int, stop ...int32) (Result, error) {
	return m.GenerateTokensWith(prompt, maxTokens, lm.SamplingOptions{}, stop...)
}

// GenerateTokensWith samples raw IDs, including a matching stop token.
func (m *Model) GenerateTokensWith(prompt []int32, maxTokens int, o lm.SamplingOptions, stop ...int32) (Result, error) {
	return m.generate(prompt, maxTokens, o, nil, stop)
}

func (m *Model) generate(prompt []int32, maxTokens int, o lm.SamplingOptions, emit func(int32) error, stop []int32) (r Result, err error) {
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
	r.Tokens, err = lm.Stream(timed, prompt, maxTokens, o, emit, stop...)
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
