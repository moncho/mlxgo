package bankingbench

import (
	"fmt"
	"math"
	"slices"

	"gonum.org/v1/gonum/optimize"
)

type feature struct {
	index int
	value float64
}

// Logistic uses training-only word unigram/bigram TF-IDF (min document count 2),
// sublinear TF, smoothed IDF and L2-normalized rows. Biases are unregularized.
type Logistic struct {
	labels       []string
	vocab        map[string]int
	idf, weights []float64
	Info         LogisticInfo
}

type LogisticInfo struct {
	C           float64 `json:"c"`
	Features    int     `json:"features"`
	Iterations  int     `json:"iterations"`
	Status      string  `json:"status"`
	Objective   float64 `json:"objective"`
	GradientMax float64 `json:"gradient_max"`
}

func terms(text string) map[string]int {
	w := words(text)
	t := map[string]int{}
	for i, word := range w {
		t[word]++
		if i > 0 {
			t[w[i-1]+" "+word]++
		}
	}
	return t
}

func (m *Logistic) vector(text string) []feature {
	var x []feature
	var norm float64
	for term, count := range terms(text) {
		if j, ok := m.vocab[term]; ok {
			v := (1 + math.Log(float64(count))) * m.idf[j]
			x = append(x, feature{j, v})
		}
	}
	slices.SortFunc(x, func(a, b feature) int { return a.index - b.index })
	for _, f := range x {
		norm += f.value * f.value
	}
	if norm > 0 {
		for i := range x {
			x[i].value /= math.Sqrt(norm)
		}
	}
	return x
}

// logisticObjective returns mean cross-entropy plus ||W||^2/(2*C*N).
// Feature-major weights keep all classes for a sparse feature contiguous.
func logisticObjective(w, grad []float64, x [][]feature, y []int, classes, features int, c float64) float64 {
	clear(grad)
	loss, scale := 0.0, 1/float64(len(x))
	scores := make([]float64, classes)
	bias := features * classes
	for row, v := range x {
		copy(scores, w[bias:])
		for _, f := range v {
			for k := range scores {
				scores[k] += w[f.index*classes+k] * f.value
			}
		}
		largest := slices.Max(scores)
		truth := scores[y[row]]
		sum := 0.0
		for k := range scores {
			scores[k] = math.Exp(scores[k] - largest)
			sum += scores[k]
		}
		loss += (largest + math.Log(sum) - truth) * scale
		for k := range scores {
			scores[k] /= sum
			if k == y[row] {
				scores[k]--
			}
			scores[k] *= scale
			grad[bias+k] += scores[k]
		}
		for _, f := range v {
			for k, p := range scores {
				grad[f.index*classes+k] += p * f.value
			}
		}
	}
	for j, v := range w[:bias] {
		loss += v * v * scale / (2 * c)
		grad[j] += v * scale / c
	}
	return loss
}

func FitLogistic(train []Record, labels []string, c float64) (*Logistic, error) {
	if len(train) == 0 || len(labels) < 2 || c <= 0 || math.IsNaN(c) || math.IsInf(c, 0) {
		return nil, fmt.Errorf("invalid logistic training data, labels or C")
	}
	m := &Logistic{labels: slices.Clone(labels), vocab: map[string]int{}}
	df := map[string]int{}
	y, counts := make([]int, len(train)), make([]int, len(labels))
	for i, r := range train {
		y[i] = slices.Index(labels, r.Completion)
		if y[i] < 0 {
			return nil, fmt.Errorf("unknown logistic training label")
		}
		counts[y[i]]++
		for term := range terms(r.Text) {
			df[term]++
		}
	}
	if slices.Min(counts) == 0 {
		return nil, fmt.Errorf("missing logistic training class")
	}
	var vocabulary []string
	for term, n := range df {
		if n >= 2 {
			vocabulary = append(vocabulary, term)
		}
	}
	slices.Sort(vocabulary)
	for j, term := range vocabulary {
		m.vocab[term] = j
		m.idf = append(m.idf, 1+math.Log(float64(1+len(train))/float64(1+df[term])))
	}
	x := make([][]feature, len(train))
	for i, r := range train {
		x[i] = m.vector(r.Text)
	}
	dim := (len(m.idf) + 1) * len(labels)
	cache, gradient := make([]float64, dim), make([]float64, dim)
	valid, loss := false, 0.0
	evaluate := func(w []float64) {
		if !valid || !slices.Equal(w, cache) {
			loss = logisticObjective(w, gradient, x, y, len(labels), len(m.idf), c)
			copy(cache, w)
			valid = true
		}
	}
	problem := optimize.Problem{
		Func: func(w []float64) float64 { evaluate(w); return loss },
		Grad: func(g, w []float64) { evaluate(w); copy(g, gradient) },
	}
	result, err := optimize.Minimize(problem, make([]float64, dim), &optimize.Settings{MajorIterations: 300, GradientThreshold: 1e-6, Concurrent: 1}, &optimize.LBFGS{})
	if err != nil {
		return nil, fmt.Errorf("logistic optimization: %w", err)
	}
	if result.Status.Early() {
		return nil, fmt.Errorf("logistic optimization did not converge: %s", result.Status)
	}
	m.weights = slices.Clone(result.X)
	maxGrad := 0.0
	for _, v := range result.Gradient {
		maxGrad = math.Max(maxGrad, math.Abs(v))
	}
	m.Info = LogisticInfo{c, len(m.idf), result.MajorIterations, result.Status.String(), result.F, maxGrad}
	return m, nil
}

func (m *Logistic) Predict(text string) string {
	k := len(m.labels)
	scores := slices.Clone(m.weights[len(m.idf)*k:])
	for _, f := range m.vector(text) {
		for j := range scores {
			scores[j] += m.weights[f.index*k+j] * f.value
		}
	}
	return m.labels[slices.Index(scores, slices.Max(scores))]
}
