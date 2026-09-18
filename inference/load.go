// Package inference loads supported local model bundles behind a common session
// and generation API. It never downloads weights or executes repository code.
package inference

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"

	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/bpe"
	"github.com/moncho/mlxgo/deepseek"
	"github.com/moncho/mlxgo/lm"
	"github.com/moncho/mlxgo/qwen2"
)

var ErrUnsupported = errors.New("inference: unsupported model feature")
var ErrClosed = errors.New("inference: model or session is closed")
var ErrTextUnsupported = errors.New("inference: text encoding is not supported for this model; use token IDs")

// Info describes the adapter selected from config.json, not a support claim for
// all checkpoints in the architecture family. Inspect does not read the weights.
type Info struct {
	Architecture            string
	Format                  string
	VocabSize, MaxPositions int
	Text, Adapters          bool
}

// Options selects an optional mlxgo Qwen LoRA checkpoint. Its base hash and
// configuration must match the model being loaded.
type Options struct{ Adapters string }
type manifest struct {
	info     Info
	qwen     qwen2.Config
	deepseek deepseek.Config
}

func readManifest(dir string) (manifest, error) {
	var m manifest
	f, err := os.Open(filepath.Join(dir, "config.json"))
	if err != nil {
		return m, err
	}
	defer f.Close()
	const limit = 1 << 20
	data, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return m, err
	}
	if len(data) > limit {
		return m, fmt.Errorf("inference: config.json exceeds 1 MiB")
	}
	var h struct {
		ModelType          string          `json:"model_type"`
		Format             string          `json:"format"`
		Quantization       json.RawMessage `json:"quantization"`
		QuantizationConfig json.RawMessage `json:"quantization_config"`
		DType              string          `json:"dtype"`
		TorchDType         string          `json:"torch_dtype"`
	}
	if err = json.Unmarshal(data, &h); err != nil {
		return m, fmt.Errorf("inference: config.json: %w", err)
	}
	for _, q := range []json.RawMessage{h.Quantization, h.QuantizationConfig} {
		if len(q) > 0 && !bytes.Equal(bytes.TrimSpace(q), []byte("null")) {
			return m, fmt.Errorf("%w: quantized checkpoints", ErrUnsupported)
		}
	}
	switch {
	case h.Format == deepseek.Float32Format && h.ModelType == "":
		d := json.NewDecoder(bytes.NewReader(data))
		d.DisallowUnknownFields()
		if err = d.Decode(&m.deepseek); err != nil {
			return m, fmt.Errorf("%w: experimental DeepSeek config: %v", ErrUnsupported, err)
		}
		if err = m.deepseek.Validate(); err != nil {
			return m, fmt.Errorf("%w: %v", ErrUnsupported, err)
		}
		m.info = Info{Architecture: "deepseek_v41", Format: deepseek.Float32Format, VocabSize: m.deepseek.VocabSize, MaxPositions: m.deepseek.MaxSeq}
	case h.ModelType == "qwen2" && h.Format == "":
		for _, dtype := range []string{h.DType, h.TorchDType} {
			if dtype != "" && dtype != "bfloat16" {
				return m, fmt.Errorf("%w: Qwen requires bfloat16 weights, got %q", ErrUnsupported, dtype)
			}
		}
		if err = json.Unmarshal(data, &m.qwen); err != nil {
			return m, err
		}
		if err = m.qwen.Validate(); err != nil {
			return m, fmt.Errorf("%w: %v", ErrUnsupported, err)
		}
		m.info = Info{Architecture: "qwen2", Format: "huggingface-bfloat16", VocabSize: m.qwen.VocabSize, MaxPositions: m.qwen.MaxPositions, Text: true, Adapters: true}
	default:
		return m, fmt.Errorf("%w: model_type=%q format=%q; released DeepSeek checkpoints are not supported", ErrUnsupported, h.ModelType, h.Format)
	}
	if _, err = os.Stat(filepath.Join(dir, "model.safetensors.index.json")); err == nil {
		return m, fmt.Errorf("%w: sharded checkpoints; expected one model.safetensors", ErrUnsupported)
	} else if !errors.Is(err, os.ErrNotExist) {
		return m, err
	}
	return m, nil
}

// Inspect validates supported configuration features without creating MLX arrays.
// Text=true means a supported text adapter exists; Open also validates tokenizer
// files, special IDs and weights before returning a usable model.
func Inspect(dir string) (Info, error) {
	m, err := readManifest(dir)
	if err != nil {
		return Info{}, err
	}
	return m.info, nil
}

// Model owns weights, optional adapters and all sessions created from it. Close
// closes outstanding sessions before weights. Native state is confined to the
// MLX worker; independent sessions can be used concurrently.
type Model struct {
	info       Info
	newSession func() (lm.Session, error)
	release    func() error
	closed     bool
	sessions   map[*session]struct{}
	encode     func(string) []int32
	decode     func([]int32) string
	eos        []int32
}

// Open loads a local config.json + model.safetensors bundle. Qwen additionally
// requires tokenizer.json and uses the existing single-turn Qwen chat format.
// Experimental DeepSeek bundles support token IDs only. No remote code or
// downloaded Jinja template is executed. The caller selects the MLX device.
func Open(dir string, options Options) (*Model, error) {
	mf, err := readManifest(dir)
	if err != nil {
		return nil, err
	}
	if options.Adapters != "" && !mf.info.Adapters {
		return nil, fmt.Errorf("%w: adapters for %s", ErrUnsupported, mf.info.Architecture)
	}
	m := &Model{info: mf.info, sessions: make(map[*session]struct{})}
	if mf.info.Text {
		tok, err := bpe.Load(filepath.Join(dir, "tokenizer.json"))
		if err != nil {
			return nil, err
		}
		if err = tok.ValidateVocabSize(mf.info.VocabSize); err != nil {
			return nil, err
		}
		m.encode = func(prompt string) []int32 { return tok.Encode(bpe.ChatTemplate(prompt)) }
		m.decode = tok.Decode
		for _, name := range []string{"<|im_start|>", "<|im_end|>", "<|endoftext|>"} {
			id, ok := tok.SpecialID(name)
			if !ok {
				return nil, fmt.Errorf("inference: missing Qwen special token %s", name)
			}
			if name != "<|im_start|>" {
				m.eos = append(m.eos, id)
			}
		}
	}
	if mf.info.Architecture == "qwen2" {
		w, err := qwen2.Load(dir, mf.qwen)
		if err != nil {
			return nil, err
		}
		var adapters *qwen2.Adapters
		if options.Adapters != "" {
			hash, e := qwen2.CheckpointHash(filepath.Join(dir, "model.safetensors"))
			if e != nil {
				w.Close()
				return nil, e
			}
			adapters, e = qwen2.LoadAdapters(options.Adapters, mf.qwen, hash)
			if e != nil {
				w.Close()
				return nil, e
			}
		}
		m.newSession = func() (lm.Session, error) {
			if adapters != nil {
				return qwen2.NewSessionWithAdapters(w, mf.qwen, adapters)
			}
			return qwen2.NewSession(w, mf.qwen)
		}
		m.release = func() error {
			var err error
			if adapters != nil {
				err = adapters.Close()
			}
			return errors.Join(err, w.Close())
		}
	} else {
		f, err := mlx.LoadSafetensors(filepath.Join(dir, "model.safetensors"))
		if err != nil {
			return nil, err
		}
		defer f.Close()
		shapes, _ := mf.deepseek.ParameterShapes()
		params := make(map[string]mlx.Array, len(shapes))
		defer func() {
			for _, a := range params {
				a.Close()
			}
		}()
		for name := range shapes {
			a, e := f.Get(name)
			if e != nil {
				return nil, fmt.Errorf("inference: parameter %s: %w", name, e)
			}
			params[name] = a
		}
		w, err := deepseek.NewModel(mf.deepseek, params)
		if err != nil {
			return nil, err
		}
		m.newSession = func() (lm.Session, error) { return w.NewSession() }
		m.release = w.Close
	}
	return m, nil
}

// Info returns immutable capabilities selected when the model was opened.
func (m *Model) Info() Info {
	if m == nil {
		return Info{}
	}
	return m.info
}

// NewSession creates independent cache state. Callers own returned logits and
// should close the session when finished; Model.Close also closes live sessions.
func (m *Model) NewSession() (out lm.Session, err error) {
	err = mlx.Batch(func() error {
		if m == nil || m.closed || m.newSession == nil {
			return ErrClosed
		}
		inner, e := m.newSession()
		if e != nil {
			return e
		}
		s := &session{model: m, inner: inner}
		m.sessions[s] = struct{}{}
		out = s
		return nil
	})
	return out, err
}

// Close releases sessions, adapters and weights. Concurrent use either finishes
// its current step or receives ErrClosed. Close is idempotent.
func (m *Model) Close() error {
	if m == nil {
		return nil
	}
	return mlx.Batch(func() error {
		if m.closed {
			return nil
		}
		m.closed = true
		var err error
		for s := range m.sessions {
			err = errors.Join(err, s.close())
		}
		if m.release != nil {
			err = errors.Join(err, m.release())
		}
		return err
	})
}

type session struct {
	model  *Model
	inner  lm.Session
	closed bool
}

func (s *session) Step(tokens []int32) (out mlx.Array, err error) {
	err = mlx.Batch(func() error {
		if s.closed || s.model.closed {
			return ErrClosed
		}
		var e error
		out, e = s.inner.Step(tokens)
		return e
	})
	return out, err
}
func (s *session) Position() int {
	var n int
	_ = mlx.Batch(func() error { n = s.inner.Position(); return nil })
	return n
}
func (s *session) close() error {
	if s.closed {
		return nil
	}
	s.closed = true
	delete(s.model.sessions, s)
	return s.inner.Close()
}
func (s *session) Close() error { return mlx.Batch(s.close) }
