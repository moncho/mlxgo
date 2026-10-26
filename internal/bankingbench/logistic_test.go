package bankingbench

import (
	"math"
	"reflect"
	"testing"
)

func TestLogisticGradient(t *testing.T) {
	x := [][]feature{{{0, .6}, {1, .8}}, {{1, 1}}}
	y := []int{0, 1}
	w := []float64{.1, -.2, .3, .4, .5, -.6}
	g := make([]float64, len(w))
	logisticObjective(w, g, x, y, 2, 2, 4)
	for i := range w {
		old, epsilon := w[i], 1e-6
		w[i] = old + epsilon
		plus := logisticObjective(w, make([]float64, len(w)), x, y, 2, 2, 4)
		w[i] = old - epsilon
		minus := logisticObjective(w, make([]float64, len(w)), x, y, 2, 2, 4)
		w[i] = old
		if math.Abs((plus-minus)/(2*epsilon)-g[i]) > 1e-7 {
			t.Fatalf("gradient[%d]", i)
		}
	}
}

func TestLogisticFitAndTrainingOnlyVocabulary(t *testing.T) {
	train := []Record{{Text: "red apple red", Completion: "a"}, {Text: "red apple", Completion: "a"}, {Text: "blue sea", Completion: "b"}, {Text: "blue sea blue", Completion: "b"}}
	m, err := FitLogistic(train, []string{"a", "b"}, 4)
	if err != nil {
		t.Fatal(err)
	}
	if m.Predict("red apple") != "a" || m.Predict("blue sea") != "b" {
		t.Fatal("failed separable fixture")
	}
	before := len(m.vocab)
	if math.Abs(m.idf[m.vocab["red"]]-(1+math.Log(5.0/3))) > 1e-12 {
		t.Fatal("IDF must count documents, not repeated tokens")
	}
	if _, ok := m.vocab["red apple"]; !ok {
		t.Fatal("missing bigram feature")
	}
	if len(m.vector("unseen validation vocabulary")) != 0 || len(m.vocab) != before {
		t.Fatal("evaluation changed vocabulary")
	}
	x := m.vector("red red apple")
	norm := 0.0
	for _, f := range x {
		norm += f.value * f.value
	}
	if math.Abs(norm-1) > 1e-12 {
		t.Fatal("TF-IDF not normalized")
	}
	second, err := FitLogistic(train, []string{"a", "b"}, 4)
	if err != nil || !reflect.DeepEqual(m.weights, second.weights) {
		t.Fatal("nondeterministic training", err)
	}
	if _, err := FitLogistic(train, []string{"a", "b"}, 0); err == nil {
		t.Fatal("accepted invalid C")
	}
	if _, err := FitLogistic(train, []string{"a", "b", "c"}, 4); err == nil {
		t.Fatal("accepted absent class")
	}
}
