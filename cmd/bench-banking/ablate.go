package main

import (
	"encoding/json"
	"fmt"
	"os"
	"reflect"

	"github.com/moncho/mlxgo/internal/bankingbench"
)

func decodingMode(r report) string {
	if r.Decoding != "" {
		return r.Decoding
	}
	if classical(r.System) {
		return "none"
	}
	return "greedy"
}

func ablate(paths []string, m bankingbench.Manifest, data map[string][]bankingbench.Record) error {
	if len(paths) < 2 {
		return fmt.Errorf("ablate requires at least two reports")
	}
	var model *report
	var split string
	seen := map[string]bool{}
	var reports []report
	for _, path := range paths {
		b, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		var r report
		if err := json.Unmarshal(b, &r); err != nil {
			return err
		}
		if err := validateReport(r, m, data); err != nil {
			return err
		}
		key := r.System + "/" + decodingMode(r) + "/" + r.Identity.Adapters + "/" + r.TrainingSHA
		if r.Identity.Logistic != nil {
			key += fmt.Sprint("/", r.Identity.Logistic.C)
		}
		if seen[key] || split != "" && split != r.Split {
			return fmt.Errorf("duplicate variants or different splits")
		}
		if r.System != "base" && (r.TrainingSHA == "" || r.TrainingN <= 0) {
			return fmt.Errorf("ablation needs declared training data for every trained model")
		}
		seen[key], split = true, r.Split
		if !classical(r.System) {
			if model != nil && (r.Identity.Model != model.Identity.Model || r.Identity.Tokenizer != model.Identity.Tokenizer || !reflect.DeepEqual(r.Identity.Config, model.Identity.Config) || r.MaxTokens != model.MaxTokens) {
				return fmt.Errorf("ablation changed model, tokenizer, configuration, or token cap")
			}
			copy := r
			model = &copy
		}
		reports = append(reports, r)
	}
	for _, r := range reports {
		printSummary(r)
	}
	return nil
}

func classical(system string) bool { return system == "nb" || system == "tfidf" }
