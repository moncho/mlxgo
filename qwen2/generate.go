package qwen2

import (
	"fmt"
	"time"

	"github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/bpe"
)

type Result struct {
	Text                          string
	Tokens                        []int32
	PrefillTokens                 int
	PrefillSeconds, DecodeSeconds float64
}

func Generate(w *Weights, c Config, tok *bpe.Tokenizer, prompt string, maxTokens int) (Result, error) {
	return generate(w, c, tok, prompt, maxTokens, nil)
}

func generate(w *Weights, c Config, tok *bpe.Tokenizer, prompt string, maxTokens int, hook projectionHook) (r Result, err error) {
	if tok == nil || maxTokens <= 0 {
		return r, fmt.Errorf("qwen2: tokenizer and positive maxTokens required")
	}
	end, ok := tok.SpecialID("<|im_end|>")
	if !ok {
		return r, fmt.Errorf("qwen2: missing im_end token")
	}
	eot, ok := tok.SpecialID("<|endoftext|>")
	if !ok {
		return r, fmt.Errorf("qwen2: missing endoftext token")
	}
	ids := tok.Encode(bpe.ChatTemplate(prompt))
	r.PrefillTokens = len(ids)
	cache := NewKVCache(c.NumLayers)
	defer cache.Close()
	for step := 0; step < maxTokens; step++ {
		start := time.Now()
		var id int32
		err := mlx.Batch(func() error {
			logits, err := forward(w, c, ids, cache, hook, false)
			if err != nil {
				return err
			}
			defer logits.Close()
			next, err := mlx.ArgmaxAxis(logits, -1, false)
			if err != nil {
				return err
			}
			defer next.Close()
			if err := mlx.Eval(append(cache.Arrays(), next)...); err != nil {
				return err
			}
			data, err := next.UInt32Data()
			if err != nil {
				return err
			}
			id = int32(data[0])
			return nil
		})
		if err != nil {
			return r, fmt.Errorf("qwen2: generation step %d: %w", step, err)
		}
		if step == 0 {
			r.PrefillSeconds = time.Since(start).Seconds()
		} else {
			r.DecodeSeconds += time.Since(start).Seconds()
		}
		if id == end || id == eot {
			break
		}
		r.Tokens = append(r.Tokens, id)
		ids = []int32{id}
	}
	r.Text = tok.Decode(r.Tokens)
	return r, nil
}
