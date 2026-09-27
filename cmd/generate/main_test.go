package main

import (
	"slices"
	"testing"
)

func TestParse(t *testing.T) {
	o, err := parse([]string{"-model", "bundle", "-device", "cpu", "-tokens", "1, 5,2,9,4", "-max-tokens", "4"})
	if err != nil {
		t.Fatal(err)
	}
	if o.dir != "bundle" || o.device != "cpu" || o.maxTokens != 4 || !slices.Equal(o.tokens, []int32{1, 5, 2, 9, 4}) {
		t.Fatalf("%+v", o)
	}
	for _, args := range [][]string{
		{"-temperature", "-1"}, {"-temperature", "NaN"}, {"-temperature", "+Inf"},
		{"-temperature", "1e100"}, {"-top-p", "0.9"}, {"-temperature", "1", "-top-p", "1.00000001"},
		{"-top-p", "NaN"}, {"-seed", "-1"}, {"-top-p", "0"},
		{"-tokens", ""}, {"-tokens", "-1"}, {"-tokens", "1,"}, {"-tokens", "2147483648"},
		{"-tokens", "1", "-prompt", "hello"}, {"-device", "metal"}, {"-max-tokens", "0"}, {"extra"},
	} {
		if _, err := parse(args); err == nil {
			t.Fatalf("accepted %v", args)
		}
	}
	o, err = parse([]string{"-temperature", "0.8", "-top-p", "0.95", "-seed", "42"})
	if err != nil || o.sampling.Temperature != .8 || o.sampling.TopP != .95 || o.sampling.Seed != 42 {
		t.Fatalf("sampling flags: %+v %v", o, err)
	}
}
