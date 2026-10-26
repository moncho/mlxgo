package main

import (
	"fmt"

	"github.com/moncho/mlxgo/bpe"
	"github.com/moncho/mlxgo/internal/bankingbench"
	"github.com/moncho/mlxgo/lm"
	"github.com/moncho/mlxgo/qwen2"
)

func choicePredictor(w *qwen2.Weights, c qwen2.Config, tok *bpe.Tokenizer, a *qwen2.Adapters, labels []string, maxTokens int) (func(string) (qwen2.Result, error), error) {
	end, ok := tok.SpecialID("<|im_end|>")
	if !ok {
		return nil, fmt.Errorf("missing im_end token")
	}
	var sequences [][]int32
	for _, label := range labels {
		ids := tok.Encode(label)
		if tok.Decode(ids) != label || len(ids)+1 > maxTokens {
			return nil, fmt.Errorf("label cannot round-trip or fit generation cap: %s", label)
		}
		sequences = append(sequences, append(ids, end))
	}
	choices, err := lm.NewChoices(sequences)
	if err != nil {
		return nil, err
	}
	return func(text string) (qwen2.Result, error) {
		var s *qwen2.Session
		var err error
		if a == nil {
			s, err = qwen2.NewSession(w, c)
		} else {
			s, err = qwen2.NewSessionWithAdapters(w, c, a)
		}
		if err != nil {
			return qwen2.Result{}, err
		}
		defer s.Close()
		prompt := tok.Encode(bpe.ChatTemplate(bankingbench.Prompt(text, labels)))
		ids, err := choices.Greedy(s, prompt, maxTokens)
		if err != nil {
			return qwen2.Result{}, err
		}
		if len(ids) == 0 || ids[len(ids)-1] != end {
			return qwen2.Result{}, fmt.Errorf("constrained generation did not complete a label")
		}
		ids = ids[:len(ids)-1]
		return qwen2.Result{Text: tok.Decode(ids), Tokens: ids, PrefillTokens: len(prompt)}, nil
	}, nil
}
