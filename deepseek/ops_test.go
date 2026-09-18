package deepseek

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

type fixture struct {
	Revision string `json:"revision"`
	Tensors  map[string]struct {
		Shape []int     `json:"shape"`
		Data  []float32 `json:"data"`
	} `json:"tensors"`
}

func readFixture(t *testing.T) fixture {
	t.Helper()
	data, err := os.ReadFile("testdata/reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var f fixture
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestFixtureIntegrity(t *testing.T) {
	f := readFixture(t)
	if f.Revision != ReferenceRevision || len(f.Tensors) != 52 {
		t.Fatal("unexpected reference provenance or tensor count")
	}
	for name, value := range f.Tensors {
		size := 1
		for _, dim := range value.Shape {
			if dim < 1 {
				t.Fatalf("%s: empty dimension", name)
			}
			size *= dim
		}
		if len(value.Data) != size {
			t.Fatalf("%s: shape/data mismatch", name)
		}
		for _, x := range value.Data {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				t.Fatalf("%s: nonfinite value", name)
			}
		}
	}
}

func TestRouterConfig(t *testing.T) {
	base := RouterConfig{TopK: 2, Score: "sqrtsoftplus", Temperature: .7, Scale: 1.5, Normalize: true}
	if err := base.validate(4); err != nil {
		t.Fatal(err)
	}
	for _, change := range []func(*RouterConfig){
		func(c *RouterConfig) { c.TopK = 0 }, func(c *RouterConfig) { c.TopK = 5 },
		func(c *RouterConfig) { c.Temperature = 0 }, func(c *RouterConfig) { c.Scale = float32(math.Inf(1)) },
		func(c *RouterConfig) { c.Temperature = float32(math.NaN()) }, func(c *RouterConfig) { c.Score = "unknown" },
	} {
		c := base
		change(&c)
		if c.validate(4) == nil {
			t.Fatalf("accepted invalid configuration: %+v", c)
		}
	}
}
