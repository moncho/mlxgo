package bankingbench

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"
)

func rows(split string, labels []string, count int) []Record {
	var records []Record
	for _, label := range labels {
		for i := range count {
			text := fmt.Sprintf("%s %s message %d", split, label, i)
			records = append(records, Record{ID: fmt.Sprintf("%s-%s-%d", split, label, i), Text: text, Prompt: Prompt(text, labels), Completion: label})
		}
	}
	return records
}

func TestSplitDeterministicAndDisjoint(t *testing.T) {
	labels := []string{"a", "b"}
	train, test := rows("train", labels, 5), rows("test", labels, 3)
	train = append(train, Record{ID: "leak", Text: "  TEST A MESSAGE 0  ", Completion: "a"}, train[0])
	counts := Counts{2, 2, 2}
	a, excluded, err := Split(train, test, labels, counts, 42)
	if err != nil || excluded != 2 {
		t.Fatalf("split: excluded=%d err=%v", excluded, err)
	}
	b, _, err := Split(train, test, labels, counts, 42)
	if err != nil || !reflect.DeepEqual(a, b) {
		t.Fatal("non-deterministic split", err)
	}
	seen := map[string]bool{}
	for _, split := range []string{"train", "valid", "test"} {
		if len(a[split]) != 4 {
			t.Fatalf("wrong count for %s", split)
		}
		for _, r := range a[split] {
			key := normalized(r.Text)
			if seen[key] || r.ID == "leak" {
				t.Fatal("leak or duplicate", r)
			}
			seen[key] = true
		}
	}
	if _, _, err := Split(train, test, labels, Counts{4, 4, 2}, 42); err == nil {
		t.Fatal("accepted insufficient data")
	}
	if _, _, err := Split(train, test, labels, Counts{0, 1, 1}, 42); err == nil {
		t.Fatal("accepted zero-sized split")
	}
}

func TestSplitConflictingLabels(t *testing.T) {
	train := []Record{{ID: "a", Text: "same text", Completion: "a"}, {ID: "b", Text: "Same Text", Completion: "b"}}
	if _, _, err := Split(train, nil, []string{"a", "b"}, Counts{1, 1, 1}, 42); err == nil {
		t.Fatal("accepted conflicting duplicate labels")
	}
}

func TestCSVParsing(t *testing.T) {
	r, err := ParseCSV([]byte("text,category\n\"hello, card\",card\n"), "train", []string{"card"})
	if err != nil || len(r) != 1 || r[0].Text != "hello, card" || r[0].ID != "train-00001" || r[0].Completion != "card" {
		t.Fatalf("records=%v err=%v", r, err)
	}
	for _, input := range []string{"wrong,header\nx,card\n", "text,category\nx,unknown\n", "text,category\n,card\n", "text,category\nx,card,extra\n"} {
		if _, err := ParseCSV([]byte(input), "train", []string{"card"}); err == nil {
			t.Fatalf("accepted invalid CSV %q", input)
		}
	}
}

func fixture(t *testing.T) (string, Manifest, map[string][]Record) {
	t.Helper()
	dir := t.TempDir()
	labels := make([]string, 77)
	for i := range labels {
		labels[i] = fmt.Sprintf("label_%02d", i)
	}
	m := Manifest{Version: Version, Revision: Revision, Seed: 42, PerClass: Counts{1, 1, 1}, Labels: labels, Files: map[string]string{}}
	raw := map[string][]Record{}
	write := func(name string, b []byte) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(dir, name), b, 0o644); err != nil {
			t.Fatal(err)
		}
		m.Files[name] = Hash(b)
	}
	b, _ := json.Marshal(labels)
	write("categories.json", b)
	write("LICENSE", []byte("test fixture, no external data"))
	for _, split := range []string{"train", "test"} {
		var buf bytes.Buffer
		w := csv.NewWriter(&buf)
		w.Write([]string{"text", "category"})
		for _, r := range rows(split, labels, 3) {
			w.Write([]string{r.Text, r.Completion})
		}
		w.Flush()
		write(split+".csv", buf.Bytes())
		var err error
		raw[split], err = ParseCSV(buf.Bytes(), split, labels)
		if err != nil {
			t.Fatal(err)
		}
	}
	data, _, err := Split(raw["train"], raw["test"], labels, m.PerClass, m.Seed)
	if err != nil {
		t.Fatal(err)
	}
	for _, split := range []string{"train", "valid", "test"} {
		var buf bytes.Buffer
		for _, r := range data[split] {
			if err := json.NewEncoder(&buf).Encode(r); err != nil {
				t.Fatal(err)
			}
		}
		write(split+".jsonl", buf.Bytes())
	}
	saveManifest(t, dir, m)
	return dir, m, data
}

func saveManifest(t *testing.T, dir string, m Manifest) {
	t.Helper()
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), b, 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestLoadIntegrity(t *testing.T) {
	dir, m, want := fixture(t)
	gotM, got, err := Load(dir)
	if err != nil || !reflect.DeepEqual(m, gotM) || !reflect.DeepEqual(got, want) {
		t.Fatalf("load: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "train.jsonl"), []byte("corrupt"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Load(dir); err == nil {
		t.Fatal("accepted hash mismatch")
	}
}

func TestLoadRejectsLeakWithUpdatedHash(t *testing.T) {
	dir, m, data := fixture(t)
	r := &data["valid"][0]
	r.Text, r.Prompt = data["train"][0].Text, data["train"][0].Prompt
	var buf bytes.Buffer
	for _, r := range data["valid"] {
		json.NewEncoder(&buf).Encode(r)
	}
	if err := os.WriteFile(filepath.Join(dir, "valid.jsonl"), buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	m.Files["valid.jsonl"] = Hash(buf.Bytes())
	saveManifest(t, dir, m)
	if _, _, err := Load(dir); err == nil {
		t.Fatal("accepted cross-split overlap despite matching hash")
	}
}

func TestPrepareDoesNotOverwrite(t *testing.T) {
	dir := t.TempDir()
	if err := Prepare(dir, Counts{1, 1, 1}, 42); err == nil {
		t.Fatal("accepted existing directory")
	}
	entries, err := os.ReadDir(dir)
	if err != nil || len(entries) != 0 {
		t.Fatal("prepare touched existing directory", entries, err)
	}
}

func TestLoadRejectsChangedSelection(t *testing.T) {
	dir, m, _ := fixture(t)
	m.Seed++
	saveManifest(t, dir, m)
	if _, _, err := Load(dir); err == nil {
		t.Fatal("accepted a selection inconsistent with the split seed")
	}
}

func TestFitUsesOnlyTrainingWords(t *testing.T) {
	train := []Record{{Text: "card card arrive", Completion: "card"}, {Text: "transfer money money", Completion: "transfer"}}
	n, err := Fit(train, []string{"card", "transfer"})
	if err != nil {
		t.Fatal(err)
	}
	if n.Predict("My CARD!") != "card" || n.Predict("transfer money") != "transfer" {
		t.Fatal("baseline failed simple classification")
	}
	before := len(n.vocab)
	if n.Predict("new unseen words") != "card" || len(n.vocab) != before {
		t.Fatal("prediction changed training vocabulary or tie behavior")
	}
	if _, err := Fit(train, []string{"card"}); err == nil {
		t.Fatal("accepted unknown label")
	}
	if !slices.Equal(words("CARD's 123"), []string{"card", "s", "123"}) {
		t.Fatal("unexpected tokenizer")
	}
}
