package qwen2

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"slices"
	"strconv"

	"github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/bpe"
)

// Adapters holds float32 LoRA factors for q/v projections in every layer.
// The frozen base weights are never replaced. One instance belongs to one caller.
type Adapters struct {
	Params     []mlx.Array
	Rank       int
	Alpha      float32
	BaseSHA256 string
	config     Config
	closed     bool
}

// CheckpointHash binds saved adapters to an exact base checkpoint, not just its shape.
func CheckpointHash(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func NewAdapters(c Config, rank int, alpha float32, seed int64, baseSHA256 string) (*Adapters, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	if rank <= 0 || rank > c.HiddenSize || alpha <= 0 || math.IsNaN(float64(alpha)) || math.IsInf(float64(alpha), 0) {
		return nil, fmt.Errorf("qwen2: invalid LoRA rank or alpha")
	}
	if raw, err := hex.DecodeString(baseSHA256); err != nil || len(raw) != sha256.Size {
		return nil, fmt.Errorf("qwen2: a base checkpoint SHA256 is required")
	}
	a := &Adapters{Rank: rank, Alpha: alpha, BaseSHA256: baseSHA256, config: c}
	rng := rand.New(rand.NewSource(seed))
	err := mlx.Batch(func() error {
		for _, shape := range a.shapes() {
			data := make([]float32, shape[0]*shape[1])
			if len(a.Params)%2 == 0 {
				for i := range data {
					data[i] = float32(rng.NormFloat64() / math.Sqrt(float64(c.HiddenSize)))
				}
			}
			x, err := mlx.NewFloat32(data, shape)
			if err != nil {
				return err
			}
			a.Params = append(a.Params, x)
		}
		return nil
	})
	if err != nil {
		_ = a.Close()
		return nil, err
	}
	return a, nil
}

func (a *Adapters) shapes() [][]int {
	var out [][]int
	for range a.config.NumLayers {
		out = append(out, []int{a.config.HiddenSize, a.Rank}, []int{a.Rank, a.config.HiddenSize}, []int{a.config.HiddenSize, a.Rank}, []int{a.Rank, a.config.NumKVHeads * a.config.HeadDim()})
	}
	return out
}

func (a *Adapters) Close() error {
	if a == nil || a.closed {
		return nil
	}
	a.closed = true
	return mlx.CloseArrays(a.Params)
}

func (a *Adapters) hook(params []mlx.Array) projectionHook {
	return func(layer int, kind string, x, base mlx.Array) (mlx.Array, error) {
		s := &scope{}
		defer s.close()
		i := layer * 4
		if kind == "v" {
			i += 2
		}
		x = s.add(mlx.AsType(x, mlx.Float32))
		low := s.add(mlx.Matmul(x, params[i]))
		delta := s.add(mlx.Matmul(low, params[i+1]))
		scale := s.add(mlx.NewScalarFloat32(a.Alpha / float32(a.Rank)))
		delta = s.add(mlx.Multiply(delta, scale))
		dtype, err := base.DType()
		if err != nil {
			return mlx.Array{}, err
		}
		base32 := s.add(mlx.AsType(base, mlx.Float32))
		out := s.add(mlx.Add(base32, delta))
		out = s.add(mlx.AsType(out, dtype))
		return s.take(out)
	}
}

func (a *Adapters) validate() error {
	if a == nil || a.closed || len(a.Params) != 4*a.config.NumLayers {
		return fmt.Errorf("qwen2: closed or invalid adapters")
	}
	return nil
}

func (a *Adapters) Forward(w *Weights, tokens []int32, cache *KVCache) (mlx.Array, error) {
	if err := a.validate(); err != nil {
		return mlx.Array{}, err
	}
	return forward(w, a.config, tokens, cache, a.hook(a.Params), false)
}

func (a *Adapters) Generate(w *Weights, tok *bpe.Tokenizer, prompt string, maxTokens int) (Result, error) {
	if err := a.validate(); err != nil {
		return Result{}, err
	}
	return generate(w, a.config, tok, prompt, maxTokens, a.hook(a.Params))
}

func (a *Adapters) Save(path string) error {
	if err := a.validate(); err != nil {
		return err
	}
	weights := map[string]mlx.Array{}
	for i, p := range a.Params {
		weights[fmt.Sprintf("lora.%d", i)] = p
	}
	cfg, err := json.Marshal(a.config)
	if err != nil {
		return err
	}
	return mlx.SaveSafetensors(path, weights, map[string]string{"format": "mlxgo-qwen2-lora-v1", "config": string(cfg), "rank": strconv.Itoa(a.Rank), "alpha": strconv.FormatFloat(float64(a.Alpha), 'g', -1, 32), "base_sha256": a.BaseSHA256})
}

func LoadAdapters(path string, c Config, baseSHA256 string) (*Adapters, error) {
	f, err := mlx.LoadSafetensors(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	get := func(key string) (string, error) {
		v, ok, err := f.Metadata(key)
		if err != nil {
			return "", err
		}
		if !ok {
			return "", fmt.Errorf("qwen2: missing adapter metadata %s", key)
		}
		return v, nil
	}
	format, err := get("format")
	if err != nil {
		return nil, err
	}
	if format != "mlxgo-qwen2-lora-v1" {
		return nil, fmt.Errorf("qwen2: unsupported adapter format")
	}
	hash, err := get("base_sha256")
	if err != nil {
		return nil, err
	}
	if hash != baseSHA256 {
		return nil, fmt.Errorf("qwen2: adapters belong to a different base checkpoint")
	}
	cfg, err := get("config")
	if err != nil {
		return nil, err
	}
	expected, _ := json.Marshal(c)
	if cfg != string(expected) {
		return nil, fmt.Errorf("qwen2: adapter config mismatch")
	}
	rankText, err := get("rank")
	if err != nil {
		return nil, err
	}
	rank, err := strconv.Atoi(rankText)
	if err != nil {
		return nil, err
	}
	alphaText, err := get("alpha")
	if err != nil {
		return nil, err
	}
	alpha, err := strconv.ParseFloat(alphaText, 32)
	if err != nil {
		return nil, err
	}
	a, err := NewAdapters(c, rank, float32(alpha), 42, baseSHA256)
	if err != nil {
		return nil, err
	}
	_ = mlx.CloseArrays(a.Params)
	a.Params = nil
	for i, shape := range a.shapes() {
		p, err := f.Get(fmt.Sprintf("lora.%d", i))
		if err != nil {
			_ = a.Close()
			return nil, err
		}
		a.Params = append(a.Params, p)
		dt, err := p.DType()
		if err != nil || dt != mlx.Float32 || !slices.Equal(p.Shape(), shape) {
			_ = a.Close()
			return nil, fmt.Errorf("qwen2: invalid adapter tensor %d", i)
		}
	}
	return a, nil
}
