package bankingbench

import (
	"fmt"
	"math"
	"slices"
	"strings"
)

type Prediction struct {
	ID         string  `json:"id"`
	Expected   string  `json:"expected"`
	Output     string  `json:"output"`
	Seconds    float64 `json:"seconds"`
	Tokens     int     `json:"tokens"`
	TokenLimit bool    `json:"token_limit"`
}

type ClassScore struct {
	Count     int     `json:"count"`
	Correct   int     `json:"correct"`
	Predicted int     `json:"predicted"`
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	F1        float64 `json:"f1"`
}

type Summary struct {
	Count      int                   `json:"count"`
	Correct    int                   `json:"correct"`
	Invalid    int                   `json:"invalid"`
	TokenLimit int                   `json:"token_limit"`
	Accuracy   float64               `json:"accuracy"`
	MacroF1    float64               `json:"macro_f1"`
	Seconds    float64               `json:"seconds"`
	PerClass   map[string]ClassScore `json:"per_class"`
}

func Score(predictions []Prediction, labels []string) (Summary, error) {
	s := Summary{PerClass: map[string]ClassScore{}}
	seen := map[string]bool{}
	for _, p := range predictions {
		if p.Seconds < 0 || math.IsNaN(p.Seconds) || math.IsInf(p.Seconds, 0) || p.Tokens < 0 {
			return s, fmt.Errorf("invalid prediction timing or token count: %s", p.ID)
		}
		if p.ID == "" || seen[p.ID] || !slices.Contains(labels, p.Expected) {
			return s, fmt.Errorf("duplicate ID or invalid reference: %s", p.ID)
		}
		seen[p.ID] = true
		s.Count++
		s.Seconds += p.Seconds
		if p.TokenLimit {
			s.TokenLimit++
		}
		out := strings.TrimSpace(p.Output)
		c := s.PerClass[p.Expected]
		c.Count++
		if out == p.Expected {
			c.Correct++
			s.Correct++
		}
		s.PerClass[p.Expected] = c
		if slices.Contains(labels, out) {
			c := s.PerClass[out]
			c.Predicted++
			s.PerClass[out] = c
		} else {
			s.Invalid++
		}
	}
	if s.Count == 0 {
		return s, fmt.Errorf("no predictions")
	}
	for _, label := range labels {
		c := s.PerClass[label]
		if c.Count == 0 {
			return s, fmt.Errorf("no references for %s", label)
		}
		c.Recall = float64(c.Correct) / float64(c.Count)
		if c.Predicted > 0 {
			c.Precision = float64(c.Correct) / float64(c.Predicted)
		}
		if c.Precision+c.Recall > 0 {
			c.F1 = 2 * c.Precision * c.Recall / (c.Precision + c.Recall)
		}
		s.MacroF1 += c.F1 / float64(len(labels))
		s.PerClass[label] = c
	}
	s.Accuracy = float64(s.Correct) / float64(s.Count)
	return s, nil
}
