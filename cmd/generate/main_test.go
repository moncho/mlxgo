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
		{"-tokens", ""}, {"-tokens", "-1"}, {"-tokens", "1,"}, {"-tokens", "2147483648"},
		{"-tokens", "1", "-prompt", "hello"}, {"-device", "metal"}, {"-max-tokens", "0"}, {"extra"},
	} {
		if _, err := parse(args); err == nil {
			t.Fatalf("accepted %v", args)
		}
	}
}
