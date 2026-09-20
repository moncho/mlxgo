package deepseek

import (
	"encoding/json"
	"math"
	"os"
	"slices"
	"testing"
)

type engramFixture struct {
	fixture
	SHA    map[string]string `json:"sha256"`
	Config EngramConfig      `json:"config"`
	Tokens [][]int32         `json:"tokens"`
	Cases  []struct {
		Mask   [][]bool `json:"mask"`
		Hashes []int32  `json:"hashes"`
	} `json:"cases"`
	Epsilon float32 `json:"epsilon"`
	Lookup  []int32 `json:"lookup_hashes"`
}

func readEngramFixture(t *testing.T) engramFixture {
	t.Helper()
	b, err := os.ReadFile("testdata/engram.json")
	if err != nil {
		t.Fatal(err)
	}
	var f engramFixture
	if err = json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestEngramHashReference(t *testing.T) {
	f := readEngramFixture(t)
	if f.Revision != ReferenceRevision || f.SHA["engram.py"] != "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897" || len(f.Cases) != 2 || len(f.Tensors) != 25 {
		t.Fatal("missing Engram reference provenance")
	}
	c := f.Config
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	if c.Multipliers[0][0] < (1<<53) || c.TokenMap[0] != c.TokenMap[1] || c.TokenMap[0] != c.TokenMap[3] || c.TokenMap[4] != c.TokenMap[5] || c.TokenMap[6] != c.TokenMap[7] || c.TokenMap[9] == c.TokenMap[10] {
		t.Fatal("fixture does not exercise compressed IDs and full-width hashing")
	}
	width := len(c.Layers) * c.columns()
	for _, test := range f.Cases {
		for row, tokens := range f.Tokens {
			want := test.Hashes[row*len(tokens)*width : (row+1)*len(tokens)*width]
			for _, chunks := range [][]int{{11}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {2, 1, 3, 2, 3}, {5, 6}} {
				h, err := NewEngramHasher(c)
				if err != nil {
					t.Fatal(err)
				}
				var got []int32
				start := 0
				for _, n := range chunks {
					var mask []bool
					if test.Mask != nil {
						mask = test.Mask[row][start : start+n]
					}
					x, err := h.Hash(tokens[start:start+n], mask)
					if err != nil {
						t.Fatal(err)
					}
					got = append(got, x...)
					start += n
					if len(h.history) > c.MaxNGram-1 || h.Position() != start {
						t.Fatal("unbounded or incorrect history")
					}
				}
				if !slices.Equal(got, want) {
					t.Fatalf("hash mismatch row=%d chunks=%v\ngot=%v\nwant=%v", row, chunks, got, want)
				}
				h.Reset()
				var mask []bool
				if test.Mask != nil {
					mask = test.Mask[row]
				}
				again, err := h.Hash(tokens, mask)
				if err != nil || !slices.Equal(again, want) {
					t.Fatal("reset mismatch", err)
				}
			}
		}
	}
}

func TestEngramHashValidation(t *testing.T) {
	c := readEngramFixture(t).Config
	for name, change := range map[string]func(*EngramConfig){
		"layers":             func(c *EngramConfig) { c.Layers[1] = c.Layers[0] },
		"empty":              func(c *EngramConfig) { c.Layers = nil },
		"ngram":              func(c *EngramConfig) { c.MaxNGram = 1 },
		"heads":              func(c *EngramConfig) { c.Heads = 0 },
		"head dimension":     func(c *EngramConfig) { c.HeadDim = 0 },
		"rows":               func(c *EngramConfig) { c.Rows[0] = 1 },
		"row count":          func(c *EngramConfig) { c.Rows = nil },
		"primes":             func(c *EngramConfig) { c.Primes[0][0] = 4 },
		"duplicate prime":    func(c *EngramConfig) { c.Primes[0][1] = c.Primes[0][0] },
		"prime count":        func(c *EngramConfig) { c.Primes[0] = nil },
		"overflow":           func(c *EngramConfig) { c.Multipliers[0][0] = math.MaxInt64 },
		"even multiplier":    func(c *EngramConfig) { c.Multipliers[0][0] = 2 },
		"missing multiplier": func(c *EngramConfig) { c.Multipliers[0] = nil },
		"pad":                func(c *EngramConfig) { c.PadID = -1 },
		"map bounds":         func(c *EngramConfig) { c.TokenMap[0] = -1 },
		"map density": func(c *EngramConfig) {
			for i := range c.TokenMap {
				c.TokenMap[i] = 0
			}
		},
	} {
		t.Run(name, func(t *testing.T) {
			bad := c.clone()
			change(&bad)
			if _, err := NewEngramHasher(bad); err == nil {
				t.Fatal("accepted bad config")
			}
		})
	}
	h, err := NewEngramHasher(c)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = h.Hash([]int32{1, 2}, nil); err != nil {
		t.Fatal(err)
	}
	before := slices.Clone(h.history)
	for _, input := range [][]int32{nil, {-1}, {int32(len(c.TokenMap))}} {
		if _, err = h.Hash(input, nil); err == nil {
			t.Fatal("accepted invalid tokens")
		}
	}
	if _, err = h.Hash([]int32{1}, []bool{}); err == nil {
		t.Fatal("accepted bad mask")
	}
	if !slices.Equal(h.history, before) || h.Position() != 2 {
		t.Fatal("bad input mutated history")
	}
	if _, err := (*EngramHasher)(nil).Hash([]int32{1}, nil); err == nil {
		t.Fatal("nil hasher")
	}
	if _, err := new(EngramHasher).Hash([]int32{1}, nil); err == nil {
		t.Fatal("zero hasher")
	}
	// Mutating the caller's nested metadata cannot change an existing hasher.
	a, err := NewEngramHasher(c)
	if err != nil {
		t.Fatal(err)
	}
	c.TokenMap[0] = c.TokenMap[2]
	c.Primes[0][0] = 2
	c.Multipliers[0][0] = 1
	want, _ := NewEngramHasher(readEngramFixture(t).Config)
	x, _ := a.Hash([]int32{0, 1, 2}, nil)
	y, _ := want.Hash([]int32{0, 1, 2}, nil)
	if !slices.Equal(x, y) {
		t.Fatal("hasher borrowed mutable metadata")
	}
}

func TestEngramModelConfig(t *testing.T) {
	f := readModelFixturePath(t, "testdata/model_engram.json")
	if f.Config.Engram == nil {
		t.Fatal("Engram disabled")
	}
	shapes, err := f.Config.ParameterShapes()
	if err != nil {
		t.Fatal(err)
	}
	if len(shapes) != len(f.Parameters) {
		t.Fatal("parameter inventory mismatch")
	}
	for name, shape := range shapes {
		if !slices.Equal(shape, f.Parameters[name].Shape) {
			t.Fatal("wrong parameter shape", name)
		}
	}
	for _, change := range []func(*Config){
		func(c *Config) { c.Engram.Layers[2] = c.Layers },
		func(c *Config) { c.Engram.TokenMap = c.Engram.TokenMap[:2] },
	} {
		c := f.Config.clone()
		change(&c)
		if c.Validate() == nil {
			t.Fatal("accepted incompatible Engram")
		}
	}
}
