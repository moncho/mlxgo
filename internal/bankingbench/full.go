package bankingbench

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
)

const FullVersion = "banking77-v3"

// FullSplit preserves pilot validation and excludes both inspected test sets.
// A zero training cap means every eligible source-training record.
func FullSplit(train, test []Record, labels []string, pilot, expanded, counts Counts, seed int) (map[string][]Record, int, error) {
	if counts.Train != 0 || counts.Valid != pilot.Valid || counts.Test <= 0 {
		return nil, 0, fmt.Errorf("full split requires zero training cap, frozen validation, and positive test count")
	}
	first, _, err := Split(train, test, labels, pilot, seed)
	if err != nil {
		return nil, 0, err
	}
	second, _, err := ExpandedSplit(train, test, labels, pilot, expanded, seed)
	if err != nil {
		return nil, 0, err
	}
	reserved := map[string]bool{}
	for _, rows := range [][]Record{first["valid"], first["test"], second["test"]} {
		for _, r := range rows {
			reserved[normalized(r.Text)] = true
		}
	}
	clean, excluded, err := cleanTraining(train, test)
	if err != nil {
		return nil, 0, err
	}
	result := map[string][]Record{"train": {}, "valid": first["valid"], "test": {}}
	for _, label := range labels {
		for _, r := range orderedRows(clean, label, seed) {
			if !reserved[normalized(r.Text)] {
				result["train"] = append(result["train"], r)
			}
		}
		var fresh []Record
		for _, r := range orderedRows(test, label, seed) {
			if !reserved[normalized(r.Text)] {
				fresh = append(fresh, r)
			}
		}
		if len(fresh) < counts.Test {
			return nil, 0, fmt.Errorf("insufficient untouched test rows for %s", label)
		}
		result["test"] = append(result["test"], fresh[:counts.Test]...)
	}
	return result, excluded, nil
}

func verifyExpanded(previous, pilot, m Manifest, raw map[string][]byte, train, test []Record) error {
	if previous.Version != ExpandedVersion || previous.Revision != m.Revision || previous.Seed != m.Seed || !slices.Equal(previous.Labels, m.Labels) {
		return fmt.Errorf("incompatible expanded manifest")
	}
	if err := verifyParent(pilot, previous, raw, train, test); err != nil {
		return err
	}
	for _, name := range []string{"categories.json", "train.csv", "test.csv", "LICENSE", "parent-manifest.json"} {
		if previous.Files[name] != Hash(raw[name]) {
			return fmt.Errorf("changed expanded source %s", name)
		}
	}
	data, excluded, err := ExpandedSplit(train, test, m.Labels, pilot.PerClass, previous.PerClass, m.Seed)
	if err != nil {
		return err
	}
	if excluded != previous.ExcludedTrain {
		return fmt.Errorf("changed expanded exclusion count")
	}
	for _, split := range []string{"train", "valid", "test"} {
		b, err := encodeRecords(data[split])
		if err != nil {
			return err
		}
		if Hash(b) != previous.Files[split+".jsonl"] {
			return fmt.Errorf("expanded %s selection does not reproduce", split)
		}
	}
	return nil
}

// Full copies a verified v2 source into a new directory; old artifacts stay intact.
func Full(parentDir, dir string, testPerClass int) error {
	previous, _, err := Load(parentDir)
	if err != nil {
		return err
	}
	if previous.Version != ExpandedVersion {
		return fmt.Errorf("full requires the v2 expanded dataset")
	}
	raw := map[string][]byte{}
	for _, name := range []string{"categories.json", "train.csv", "test.csv", "LICENSE", "parent-manifest.json", "manifest.json"} {
		b, err := os.ReadFile(filepath.Join(parentDir, name))
		if err != nil {
			return err
		}
		if name == "manifest.json" {
			name = "previous-manifest.json"
		}
		raw[name] = b
	}
	var pilot Manifest
	if err := json.Unmarshal(raw["parent-manifest.json"], &pilot); err != nil {
		return err
	}
	train, err := ParseCSV(raw["train.csv"], "train", previous.Labels)
	if err != nil {
		return err
	}
	test, err := ParseCSV(raw["test.csv"], "test", previous.Labels)
	if err != nil {
		return err
	}
	m := previous
	m.Version, m.PerClass = FullVersion, Counts{0, previous.PerClass.Valid, testPerClass}
	m.Files, m.TrainCounts = map[string]string{}, map[string]int{}
	data, excluded, err := FullSplit(train, test, m.Labels, pilot.PerClass, previous.PerClass, m.PerClass, m.Seed)
	if err != nil {
		return err
	}
	m.ExcludedTrain = excluded
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
