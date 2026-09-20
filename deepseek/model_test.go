package deepseek

import (
	"encoding/json"
	"os"
	"testing"
)

type tensorFixture struct {
	Shape []int     `json:"shape"`
	Data  []float32 `json:"data"`
}
type modelCase struct {
	Prefill    int           `json:"prefill"`
	Yarn       bool          `json:"yarn"`
	Logits     tensorFixture `json:"logits"`
	ChunkError float64       `json:"reference_chunk_error"`
}
type modelFixture struct {
	Revision       string                   `json:"revision"`
	Config         Config                   `json:"config"`
	Parameters     map[string]tensorFixture `json:"parameters"`
	Second         map[string]tensorFixture `json:"second_parameters"`
	SecondLogits   tensorFixture            `json:"second_logits"`
	Tokens         []int32                  `json:"tokens"`
	Cases          []modelCase              `json:"cases"`
	UnpatchedError float64                  `json:"unpatched_reference_chunk_error"`
	Adjustments    []string                 `json:"reference_adjustments"`
}

func readModelFixture(t *testing.T) modelFixture {
	return readModelFixturePath(t, "testdata/model.json")
}
func readModelFixturePath(t *testing.T, path string) modelFixture {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var f modelFixture
	if err = json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}
func TestModelFixtureContract(t *testing.T) {
	for _, path := range []string{"testdata/model.json", "testdata/model_engram.json"} {
		t.Run(path, func(t *testing.T) { checkModelFixtureContract(t, readModelFixturePath(t, path)) })
	}
}
func checkModelFixtureContract(t *testing.T, f modelFixture) {
	t.Helper()
	if f.Revision != ReferenceRevision || len(f.Cases) != 14 || len(f.Adjustments) != 1 || f.UnpatchedError < .001 {
		t.Fatal("missing reference provenance or regression evidence")
	}
	shapes, err := f.Config.ParameterShapes()
	if err != nil {
		t.Fatal(err)
	}
	if len(shapes) != len(f.Parameters) {
		t.Fatal("parameter inventory mismatch")
	}
	for _, test := range f.Cases {
		if test.ChunkError > 2e-5 {
			t.Fatal("reference self-consistency failed")
		}
	}
}
func TestModelConfigValidation(t *testing.T) {
	f := readModelFixture(t)
	for name, change := range map[string]func(*Config){
		"format":              func(c *Config) { c.Format = "deepseek_v41" },
		"odd_rope":            func(c *Config) { c.RopeDim = 3 },
		"groups":              func(c *Config) { c.Groups = 3 },
		"missing_owner":       func(c *Config) { c.KVSources = []int{4} },
		"owner_without_index": func(c *Config) { c.IndexSources = []int{2, 4, 6} },
		"ratio_change":        func(c *Config) { c.Ratios[2] = 1 },
		"duplicate_sources":   func(c *Config) { c.KVSources = []int{1, 1, 4} },
		"out_of_bounds":       func(c *Config) { c.KVSources = []int{1, 100} },
		"candidate_capacity":  func(c *Config) { c.CandidateBlocks = 1 },
		"candidate_source":    func(c *Config) { c.CandidateSource = 5 },
		"candidate_reset":     func(c *Config) { c.KVSources = append(c.KVSources, 6) },
		"short_schedule":      func(c *Config) { c.Ratios = c.Ratios[:2] },
		"window":              func(c *Config) { c.Window = c.MaxSeq + 1 },
	} {
		t.Run(name, func(t *testing.T) {
			c := f.Config.clone()
			change(&c)
			if c.Validate() == nil {
				t.Fatal("accepted invalid configuration")
			}
		})
	}
}
