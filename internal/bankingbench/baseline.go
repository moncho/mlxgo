package bankingbench

import (
	"fmt"
	"math"
	"slices"
	"strings"
	"unicode"
)

// NaiveBayes is a multinomial word-unigram baseline with Laplace smoothing
// (alpha=1), empirical class priors, and no vocabulary from validation or test.
type NaiveBayes struct {
	labels []string
	vocab  map[string]bool
	words  []map[string]int
	totals []int
	priors []float64
}

func words(text string) []string {
	return strings.FieldsFunc(strings.ToLower(text), func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsNumber(r) })
}

func Fit(train []Record, labels []string) (*NaiveBayes, error) {
	if len(train) == 0 || len(labels) == 0 {
		return nil, fmt.Errorf("empty training data or labels")
	}
	n := &NaiveBayes{labels: slices.Clone(labels), vocab: map[string]bool{}, words: make([]map[string]int, len(labels)), totals: make([]int, len(labels)), priors: make([]float64, len(labels))}
	counts := make([]int, len(labels))
	for i := range n.words {
		n.words[i] = map[string]int{}
	}
	for _, r := range train {
		i := slices.Index(labels, r.Completion)
		if i < 0 {
			return nil, fmt.Errorf("unknown training label %s", r.Completion)
		}
		counts[i]++
		for _, word := range words(r.Text) {
			n.vocab[word] = true
			n.words[i][word]++
			n.totals[i]++
		}
	}
	for i, count := range counts {
		if count == 0 {
			return nil, fmt.Errorf("no training examples for %s", labels[i])
		}
		n.priors[i] = math.Log(float64(count) / float64(len(train)))
	}
	return n, nil
}

func (n *NaiveBayes) Predict(text string) string {
	best, score := 0, math.Inf(-1)
	tokens := words(text)
	for i := range n.labels {
		value := n.priors[i]
		for _, word := range tokens {
			if n.vocab[word] {
				value += math.Log(float64(n.words[i][word]+1) / float64(n.totals[i]+len(n.vocab)))
			}
		}
		if value > score {
			best, score = i, value
		}
	}
	return n.labels[best]
}
