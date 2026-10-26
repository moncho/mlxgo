// Package bankingbench implements a reproducible BANKING77 intent experiment.
package bankingbench

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"time"
)

const Version = "banking77-v1"
const ExpandedVersion = "banking77-v2"
const Revision = "57ec275d8078af65b7731c2a98be812d844a6d6b"
const Source = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/" + Revision + "/"

type Record struct {
	ID         string `json:"id"`
	Text       string `json:"text"`
	Prompt     string `json:"prompt"`
	Completion string `json:"completion"`
}

type Counts struct {
	Train int `json:"train"`
	Valid int `json:"valid"`
	Test  int `json:"test"`
}

type Manifest struct {
	Version       string            `json:"version"`
	Revision      string            `json:"revision"`
	License       string            `json:"license"`
	Seed          int               `json:"seed"`
	PerClass      Counts            `json:"per_class"`
	Labels        []string          `json:"labels"`
	Files         map[string]string `json:"sha256"`
	ExcludedTrain int               `json:"excluded_train_duplicates_or_test_overlap"`
	TrainCounts   map[string]int    `json:"train_counts,omitempty"`
}

func Hash(data []byte) string {
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:])
}

func Prompt(text string, labels []string) string {
	return "Classify the banking customer message. Return exactly one intent from the list, with no explanation or extra text.\nIntents: " + strings.Join(labels, ", ") + "\n\nCustomer message: " + text + "\nIntent:"
}

func normalized(text string) string { return strings.Join(strings.Fields(strings.ToLower(text)), " ") }

func ParseCSV(data []byte, split string, labels []string) ([]Record, error) {
	rows, err := csv.NewReader(bytes.NewReader(data)).ReadAll()
	if err != nil {
		return nil, err
	}
	if len(rows) < 2 || !slices.Equal(rows[0], []string{"text", "category"}) {
		return nil, fmt.Errorf("unexpected BANKING77 CSV header or empty file")
	}
	var records []Record
	for i, row := range rows[1:] {
		if len(row) != 2 || strings.TrimSpace(row[0]) == "" || !slices.Contains(labels, row[1]) {
			return nil, fmt.Errorf("%s row %d: invalid text or category", split, i+2)
		}
		records = append(records, Record{ID: fmt.Sprintf("%s-%05d", split, i+1), Text: row[0], Prompt: Prompt(row[0], labels), Completion: row[1]})
	}
	return records, nil
}

// Split uses only source-training rows for train/validation. Whole-source test
// overlaps are excluded from training before any sampling, regardless of labels.
func Split(train, test []Record, labels []string, counts Counts, seed int) (map[string][]Record, int, error) {
	if counts.Train <= 0 || counts.Valid <= 0 || counts.Test <= 0 {
		return nil, 0, fmt.Errorf("per-class counts must be positive")
	}
	clean, excluded, err := cleanTraining(train, test)
	if err != nil {
		return nil, 0, err
	}
	result := map[string][]Record{"train": {}, "valid": {}, "test": {}}
	for _, label := range labels {
		a, b := orderedRows(clean, label, seed), orderedRows(test, label, seed)
		if counts.Train > len(a) || counts.Valid > len(a)-counts.Train || counts.Test > len(b) {
			return nil, 0, fmt.Errorf("not enough records for %s", label)
		}
		result["train"] = append(result["train"], a[:counts.Train]...)
		result["valid"] = append(result["valid"], a[counts.Train:counts.Train+counts.Valid]...)
		result["test"] = append(result["test"], b[:counts.Test]...)
	}
	return result, excluded, nil
}

func cleanTraining(train, test []Record) ([]Record, int, error) {
	testTexts := map[string]bool{}
	for _, r := range test {
		testTexts[normalized(r.Text)] = true
	}
	seen := map[string]string{}
	clean := make([]Record, 0, len(train))
	excluded := 0
	for _, r := range train {
		key := normalized(r.Text)
		if testTexts[key] {
			excluded++
			continue
		}
		if label, exists := seen[key]; exists {
			if label != r.Completion {
				return nil, 0, fmt.Errorf("conflicting labels for duplicate training text")
			}
			excluded++
			continue
		}
		seen[key] = r.Completion
		clean = append(clean, r)
	}
	return clean, excluded, nil
}

func orderedRows(rows []Record, label string, seed int) []Record {
	var selected []Record
	for _, r := range rows {
		if r.Completion == label {
			selected = append(selected, r)
		}
	}
	sort.Slice(selected, func(i, j int) bool {
		key := func(r Record) string { return Hash([]byte(fmt.Sprintf("%d/%s", seed, r.ID))) }
		return key(selected[i]) < key(selected[j])
	})
	return selected
}

// Prepare refuses existing output directories. No partial dataset is usable
// until its manifest has been written; retry in a fresh directory on failure.
func Prepare(dir string, counts Counts, seed int) error {
	if counts.Train <= 0 || counts.Valid <= 0 || counts.Test <= 0 {
		return fmt.Errorf("per-class counts must be positive")
	}
	if err := os.MkdirAll(filepath.Dir(dir), 0o755); err != nil {
		return err
	}
	if err := os.Mkdir(dir, 0o755); err != nil {
		return err
	}
	client := &http.Client{Timeout: time.Minute}
	m := Manifest{Version: Version, Revision: Revision, License: "CC-BY-4.0; PolyAI; Casanueva et al. (2020)", Seed: seed, PerClass: counts, Files: map[string]string{}}
	raw := map[string][]byte{}
	for _, name := range []string{"categories.json", "train.csv", "test.csv", "LICENSE"} {
		path := "banking_data/" + name
		if name == "LICENSE" {
			path = name
		}
		response, err := client.Get(Source + path)
		if err != nil {
			return err
		}
		data, err := io.ReadAll(io.LimitReader(response.Body, (16<<20)+1))
		response.Body.Close()
		if err != nil {
			return err
		}
		if response.StatusCode != http.StatusOK || len(data) > 16<<20 {
			return fmt.Errorf("download %s: status %d or oversized response", name, response.StatusCode)
		}
		raw[name] = data
		m.Files[name] = Hash(data)
		if err := os.WriteFile(filepath.Join(dir, name), data, 0o644); err != nil {
			return err
		}
	}
	if err := json.Unmarshal(raw["categories.json"], &m.Labels); err != nil {
		return err
	}
	if len(m.Labels) != 77 {
		return fmt.Errorf("expected 77 categories")
	}
	slices.Sort(m.Labels)
	if len(slices.Compact(slices.Clone(m.Labels))) != 77 {
		return fmt.Errorf("duplicate categories")
	}
	train, err := ParseCSV(raw["train.csv"], "train", m.Labels)
	if err != nil {
		return err
	}
	test, err := ParseCSV(raw["test.csv"], "test", m.Labels)
	if err != nil {
		return err
	}
	if len(train) != 10003 || len(test) != 3080 {
		return fmt.Errorf("unexpected upstream split counts")
	}
	data, excluded, err := Split(train, test, m.Labels, counts, seed)
	if err != nil {
		return err
	}
	m.ExcludedTrain = excluded
	for _, split := range []string{"train", "valid", "test"} {
		var buf bytes.Buffer
		enc := json.NewEncoder(&buf)
		for _, r := range data[split] {
			if err := enc.Encode(r); err != nil {
				return err
			}
		}
		name := split + ".jsonl"
		m.Files[name] = Hash(buf.Bytes())
		if err := os.WriteFile(filepath.Join(dir, name), buf.Bytes(), 0o644); err != nil {
			return err
		}
	}
	b, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "manifest.json"), append(b, '\n'), 0o644)
}

func Load(dir string) (Manifest, map[string][]Record, error) {
	var m Manifest
	b, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return m, nil, err
	}
	if err := json.Unmarshal(b, &m); err != nil {
		return m, nil, err
	}
	if (m.Version != Version && m.Version != ExpandedVersion && m.Version != FullVersion) || m.Revision != Revision || len(m.Labels) != 77 || !slices.IsSorted(m.Labels) || len(slices.Compact(slices.Clone(m.Labels))) != 77 {
		return m, nil, fmt.Errorf("invalid dataset manifest")
	}
	raw := map[string][]byte{}
	for _, name := range []string{"categories.json", "train.csv", "test.csv", "LICENSE", "train.jsonl", "valid.jsonl", "test.jsonl"} {
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			return m, nil, err
		}
		if Hash(data) != m.Files[name] {
			return m, nil, fmt.Errorf("dataset hash mismatch: %s", name)
		}
		raw[name] = data
	}
	var labels []string
	if err := json.Unmarshal(raw["categories.json"], &labels); err != nil {
		return m, nil, err
	}
	slices.Sort(labels)
	if !slices.Equal(labels, m.Labels) {
		return m, nil, fmt.Errorf("manifest labels differ from source categories")
	}
	train, err := ParseCSV(raw["train.csv"], "train", labels)
	if err != nil {
		return m, nil, err
	}
	test, err := ParseCSV(raw["test.csv"], "test", labels)
	if err != nil {
		return m, nil, err
	}
	var expected map[string][]Record
	var excluded int
	if m.Version == Version {
		expected, excluded, err = Split(train, test, labels, m.PerClass, m.Seed)
	}
	if m.Version == ExpandedVersion || m.Version == FullVersion {
		parentBytes, readErr := os.ReadFile(filepath.Join(dir, "parent-manifest.json"))
		if readErr != nil {
			return m, nil, readErr
		}
		if Hash(parentBytes) != m.Files["parent-manifest.json"] {
			return m, nil, fmt.Errorf("parent manifest hash mismatch")
		}
		var parent Manifest
		if err := json.Unmarshal(parentBytes, &parent); err != nil {
			return m, nil, err
		}
		if err := verifyParent(parent, m, raw, train, test); err != nil {
			return m, nil, err
		}
		if m.Version == ExpandedVersion {
			expected, excluded, err = ExpandedSplit(train, test, labels, parent.PerClass, m.PerClass, m.Seed)
		}
		if m.Version == FullVersion {
			raw["parent-manifest.json"] = parentBytes
			previousBytes, readErr := os.ReadFile(filepath.Join(dir, "previous-manifest.json"))
			if readErr != nil {
				return m, nil, readErr
			}
			if Hash(previousBytes) != m.Files["previous-manifest.json"] {
				return m, nil, fmt.Errorf("previous manifest hash mismatch")
			}
			var previous Manifest
			if err := json.Unmarshal(previousBytes, &previous); err != nil {
				return m, nil, err
			}
			if err := verifyExpanded(previous, parent, m, raw, train, test); err != nil {
				return m, nil, err
			}
			expected, excluded, err = FullSplit(train, test, labels, parent.PerClass, previous.PerClass, m.PerClass, m.Seed)
		}
	}
	if err != nil {
		return m, nil, err
	}
	if excluded != m.ExcludedTrain {
		return m, nil, fmt.Errorf("incorrect excluded-training count")
	}
	result := map[string][]Record{}
	seenIDs, seenText := map[string]bool{}, map[string]string{}
	for i, split := range []string{"train", "valid", "test"} {
		data, err := os.ReadFile(filepath.Join(dir, split+".jsonl"))
		if err != nil {
			return m, nil, err
		}
		scanner := bufio.NewScanner(bytes.NewReader(data))
		scanner.Buffer(make([]byte, 4096), 1<<20)
		counts := map[string]int{}
		for scanner.Scan() {
			var r Record
			if err := json.Unmarshal(scanner.Bytes(), &r); err != nil {
				return m, nil, err
			}
			key := normalized(r.Text)
			previous := seenText[key]
			if r.ID == "" || key == "" || seenIDs[r.ID] || previous != "" && previous != split || !slices.Contains(m.Labels, r.Completion) || r.Prompt != Prompt(r.Text, m.Labels) {
				return m, nil, fmt.Errorf("invalid/overlapping record %s in %s", r.ID, split)
			}
			seenIDs[r.ID], seenText[key] = true, split
			counts[r.Completion]++
			result[split] = append(result[split], r)
		}
		if err := scanner.Err(); err != nil {
			return m, nil, err
		}
		want := []int{m.PerClass.Train, m.PerClass.Valid, m.PerClass.Test}[i]
		for _, label := range m.Labels {
			classWant := want
			if (m.Version == ExpandedVersion || m.Version == FullVersion) && split == "train" {
				classWant = m.TrainCounts[label]
				if m.Version == ExpandedVersion && classWant > want {
					return m, nil, fmt.Errorf("training count exceeds declared cap")
				}
			}
			if classWant <= 0 || counts[label] != classWant {
				return m, nil, fmt.Errorf("incorrect class count in %s", split)
			}
		}
		if !slices.Equal(result[split], expected[split]) {
			return m, nil, fmt.Errorf("%s differs from the declared deterministic source selection", split)
		}
	}
	return m, result, nil
}
