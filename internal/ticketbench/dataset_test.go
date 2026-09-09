package ticketbench

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestDatasetDeterministicAndDisjoint(t *testing.T) {
	dir := t.TempDir()
	if err := Prepare(dir); err != nil {
		t.Fatal(err)
	}
	m, data, err := LoadVerified(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(m.Counts, map[string]int{"train": 192, "valid": 48, "test": 96, "retention": 8}) {
		t.Fatalf("unexpected counts %v", m.Counts)
	}
	for split, records := range data {
		if split == "retention" {
			continue
		}
		counts := map[string]int{}
		for _, record := range records {
			f, err := parseFields(record.Completion)
			if err != nil {
				t.Fatal(err)
			}
			counts[f.Queue+"/"+f.Priority]++
		}
		if len(counts) != 12 {
			t.Fatal("missing label combinations")
		}
		for _, n := range counts {
			if n != len(records)/12 {
				t.Fatal("unbalanced labels")
			}
		}
	}
	if err := Prepare(dir); err != nil {
		t.Fatal(err)
	}
	m2, data2, err := LoadVerified(dir)
	if err != nil || !reflect.DeepEqual(m, m2) || !reflect.DeepEqual(data, data2) {
		t.Fatal("nondeterministic preparation", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "test.jsonl"), []byte("changed\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := LoadVerified(dir); err == nil {
		t.Fatal("accepted altered data")
	}
}

func TestRejectSplitLeakageWithUpdatedHash(t *testing.T) {
	for _, field := range []string{"id", "family", "prompt"} {
		t.Run(field, func(t *testing.T) {
			dir := t.TempDir()
			if err := Prepare(dir); err != nil {
				t.Fatal(err)
			}
			manifest, records, err := LoadVerified(dir)
			if err != nil {
				t.Fatal(err)
			}
			r := &records["test"][0]
			switch field {
			case "id":
				r.ID = records["train"][0].ID
			case "family":
				r.Family = records["train"][0].Family
			case "prompt":
				r.Prompt = records["train"][0].Prompt
			}
			var lines []string
			for _, record := range records["test"] {
				data, err := json.Marshal(record)
				if err != nil {
					t.Fatal(err)
				}
				lines = append(lines, string(data))
			}
			data := []byte(strings.Join(lines, "\n") + "\n")
			hash := sha256.Sum256(data)
			manifest.SHA256["test.jsonl"] = hex.EncodeToString(hash[:])
			if err := os.WriteFile(filepath.Join(dir, "test.jsonl"), data, 0o644); err != nil {
				t.Fatal(err)
			}
			meta, err := json.Marshal(manifest)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, "manifest.json"), meta, 0o644); err != nil {
				t.Fatal(err)
			}
			if _, _, err := LoadVerified(dir); err == nil {
				t.Fatal("accepted split leakage")
			}
		})
	}
}

func TestCommittedDatasetMatchesGenerator(t *testing.T) {
	_, data, err := LoadVerified(filepath.Join("..", "..", "benchmarks", "tickets"))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(data, Dataset()) {
		t.Fatal("committed data differs from generator")
	}
}

func TestRejectManifestCountAndVersion(t *testing.T) {
	for _, change := range []func(*Manifest){
		func(m *Manifest) { m.Version = "unknown" },
		func(m *Manifest) { m.Counts["test"]++ },
	} {
		dir := t.TempDir()
		if err := Prepare(dir); err != nil {
			t.Fatal(err)
		}
		m, _, err := LoadVerified(dir)
		if err != nil {
			t.Fatal(err)
		}
		change(&m)
		data, err := json.Marshal(m)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "manifest.json"), data, 0o644); err != nil {
			t.Fatal(err)
		}
		if _, _, err := LoadVerified(dir); err == nil {
			t.Fatal("accepted invalid manifest")
		}
	}
}
