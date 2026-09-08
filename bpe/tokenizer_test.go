package bpe

import (
	"encoding/json"
	"os"
	"slices"
	"sync"
	"testing"
)

func TestHuggingFaceParity(t *testing.T) {
	tok, err := Load("testdata/tokenizer.json")
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("testdata/parity.json")
	if err != nil {
		t.Fatal(err)
	}
	var entries []struct {
		Text, Normalized string
		Pieces           []string
		IDs              []int32
	}
	if err := json.Unmarshal(data, &entries); err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		t.Run(e.Text, func(t *testing.T) {
			parts := pretokenize(e.Normalized)
			for i := range parts {
				parts[i] = tok.byteEncode(parts[i])
			}
			if !slices.Equal(parts, e.Pieces) {
				t.Fatalf("pretokenization: got %q want %q", parts, e.Pieces)
			}
			ids := tok.Encode(e.Text)
			if !slices.Equal(ids, e.IDs) {
				t.Fatalf("IDs: got %v want %v", ids, e.IDs)
			}
			if got := tok.Decode(ids); got != e.Normalized {
				t.Fatalf("round-trip: got %q want %q", got, e.Normalized)
			}
		})
	}
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			for _, e := range entries {
				if !slices.Equal(tok.Encode(e.Text), e.IDs) {
					t.Error("concurrent encoding differs")
				}
			}
		})
	}
	wg.Wait()
}
