package lm

import (
	"math"
	"testing"
)

func TestSamplingValidation(t *testing.T) {
	s := &errorSession{}
	for _, o := range []SamplingOptions{
		{Temperature: -1}, {Temperature: float32(math.NaN())}, {Temperature: float32(math.Inf(1))},
		{TopP: -1}, {TopP: 1.1}, {TopP: float32(math.NaN())}, {TopP: float32(math.Inf(1))},
	} {
		if _, err := Sample(s, []int32{0}, 1, o); err == nil {
			t.Fatalf("accepted %+v", o)
		}
	}
	if s.calls != 0 {
		t.Fatal("invalid options touched session")
	}
	if got, err := Sample(s, []int32{0}, 0, SamplingOptions{Temperature: 1}); err != nil || len(got) != 0 {
		t.Fatalf("empty sample: %v %v", got, err)
	}
}
