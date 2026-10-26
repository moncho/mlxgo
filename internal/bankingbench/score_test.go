package bankingbench

import (
	"math"
	"testing"
)

func TestStrictScore(t *testing.T) {
	p := []Prediction{
		{ID: "1", Expected: "a", Output: " a\n", Seconds: 1},
		{ID: "2", Expected: "a", Output: "b", Seconds: 2},
		{ID: "3", Expected: "b", Output: "b", Seconds: 3},
		{ID: "4", Expected: "b", Output: "```b```", Seconds: 4, TokenLimit: true},
	}
	s, err := Score(p, []string{"a", "b"})
	if err != nil || s.Correct != 2 || s.Invalid != 1 || s.Count != 4 || s.Accuracy != 0.5 || s.Seconds != 10 || s.TokenLimit != 1 {
		t.Fatalf("summary=%+v err=%v", s, err)
	}
	if math.Abs(s.MacroF1-7.0/12) > 1e-12 || s.PerClass["a"].Precision != 1 || s.PerClass["b"].Precision != 0.5 {
		t.Fatalf("wrong macro metrics: %+v", s)
	}
	for _, output := range []string{"A", "a because", `{"intent":"a"}`, "unknown", ""} {
		s, err := Score([]Prediction{{ID: "1", Expected: "a", Output: output}}, []string{"a"})
		if err != nil || s.Invalid != 1 || s.Correct != 0 {
			t.Fatalf("accepted %q", output)
		}
	}
}

func TestScoreRejectsBadReferences(t *testing.T) {
	for _, p := range [][]Prediction{
		nil,
		{{ID: "1", Expected: "x"}},
		{{ID: "1", Expected: "a"}, {ID: "1", Expected: "a"}},
		{{ID: "1", Expected: "a", Seconds: -1}},
	} {
		if _, err := Score(p, []string{"a"}); err == nil {
			t.Fatal("accepted bad predictions", p)
		}
	}
}
