package bankingbench

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
)

func encodeRecords(records []Record) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	for _, record := range records {
		if err := enc.Encode(record); err != nil {
			return nil, err
		}
	}
	return buf.Bytes(), nil
}

// ExpandedSplit preserves the parent's validation rows and excludes every old
// test text (case/whitespace normalized) from the fresh test selection. Training
// remains isolated from the entire official test source, not just the sample.
func ExpandedSplit(train, test []Record, labels []string, parent, counts Counts, seed int) (map[string][]Record, int, error) {
	if counts.Train <= parent.Train || counts.Valid != parent.Valid || counts.Test <= 0 {
		return nil, 0, fmt.Errorf("expansion requires more training rows and unchanged validation count")
	}
	old, excluded, err := Split(train, test, labels, parent, seed)
	if err != nil {
		return nil, 0, err
	}
	reserved := map[string]bool{}
	for _, split := range []string{"valid", "test"} {
		for _, r := range old[split] {
			reserved[normalized(r.Text)] = true
		}
	}
	clean, _, err := cleanTraining(train, test)
	if err != nil {
		return nil, 0, err
	}
	var eligibleTrain, eligibleTest []Record
	for _, r := range clean {
		if !reserved[normalized(r.Text)] {
			eligibleTrain = append(eligibleTrain, r)
		}
	}
	for _, r := range test {
		if !reserved[normalized(r.Text)] {
			eligibleTest = append(eligibleTest, r)
		}
	}
	result := map[string][]Record{"train": {}, "valid": old["valid"], "test": {}}
	for _, label := range labels {
		a, b := orderedRows(eligibleTrain, label, seed), orderedRows(eligibleTest, label, seed)
		if len(a) <= parent.Train || len(b) < counts.Test {
			return nil, 0, fmt.Errorf("not enough additional records for %s", label)
		}
		result["train"] = append(result["train"], a[:min(counts.Train, len(a))]...)
		result["test"] = append(result["test"], b[:counts.Test]...)
	}
	return result, excluded, nil
}

func verifyParent(parent, m Manifest, raw map[string][]byte, train, test []Record) error {
	if parent.Version != Version || parent.Revision != m.Revision || parent.Seed != m.Seed || !slices.Equal(parent.Labels, m.Labels) {
		return fmt.Errorf("incompatible parent manifest")
	}
	for _, name := range []string{"categories.json", "train.csv", "test.csv", "LICENSE"} {
		if parent.Files[name] != Hash(raw[name]) {
			return fmt.Errorf("changed parent source %s", name)
		}
	}
	old, excluded, err := Split(train, test, m.Labels, parent.PerClass, m.Seed)
	if err != nil {
		return err
	}
	if excluded != parent.ExcludedTrain {
		return fmt.Errorf("changed parent exclusion count")
	}
	for _, split := range []string{"train", "valid", "test"} {
		b, err := encodeRecords(old[split])
		if err != nil {
			return err
		}
		if Hash(b) != parent.Files[split+".jsonl"] {
			return fmt.Errorf("parent %s selection does not reproduce", split)
		}
	}
	return nil
}

// Expand copies a verified v1 source locally; it never mutates or redownloads it.
func Expand(parentDir, dir string, trainPerClass, testPerClass int) error {
	parent, _, err := Load(parentDir)
	if err != nil {
		return err
	}
	if parent.Version != Version {
		return fmt.Errorf("expand requires the original v1 pilot")
	}
	m := parent
	m.Version = ExpandedVersion
	m.PerClass = Counts{trainPerClass, parent.PerClass.Valid, testPerClass}
	m.Files = map[string]string{}
	raw := map[string][]byte{}
	for _, name := range []string{"categories.json", "train.csv", "test.csv", "LICENSE", "manifest.json"} {
		b, err := os.ReadFile(filepath.Join(parentDir, name))
		if err != nil {
			return err
		}
		if name == "manifest.json" {
			name = "parent-manifest.json"
		}
		raw[name] = b
	}
	train, err := ParseCSV(raw["train.csv"], "train", m.Labels)
	if err != nil {
		return err
	}
	test, err := ParseCSV(raw["test.csv"], "test", m.Labels)
	if err != nil {
		return err
	}
	data, excluded, err := ExpandedSplit(train, test, m.Labels, parent.PerClass, m.PerClass, m.Seed)
	if err != nil {
		return err
	}
	m.ExcludedTrain = excluded
	m.TrainCounts = map[string]int{}
	for _, r := range data["train"] {
		m.TrainCounts[r.Completion]++
	}
	for _, split := range []string{"train", "valid", "test"} {
		raw[split+".jsonl"], err = encodeRecords(data[split])
		if err != nil {
			return err
		}
	}
	if err := os.MkdirAll(filepath.Dir(dir), 0o755); err != nil {
		return err
	}
	if err := os.Mkdir(dir, 0o755); err != nil {
		return err
	}
	for name, b := range raw {
		if err := os.WriteFile(filepath.Join(dir, name), b, 0o644); err != nil {
			return err
		}
		m.Files[name] = Hash(b)
	}
	b, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "manifest.json"), append(b, '\n'), 0o644)
}
