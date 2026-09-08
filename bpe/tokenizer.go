// Package bpe implements the Qwen2 byte-level BPE tokenizer and chat template.
package bpe

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

const splitPattern = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
const spaces = `\t\n\v\f\r \x{85}\x{A0}\x{1680}\x{2000}-\x{200A}\x{2028}\x{2029}\x{202F}\x{205F}\x{3000}`

var splitAlternatives = []*regexp.Regexp{
	regexp.MustCompile(`^(?i:'s|'t|'re|'ve|'m|'ll|'d)`),
	regexp.MustCompile(`^[^\r\n\p{L}\p{N}]?\p{L}+`),
	regexp.MustCompile(`^\p{N}`),
	regexp.MustCompile(`^ ?[^` + spaces + `\p{L}\p{N}]+[\r\n]*`),
	regexp.MustCompile(`^[` + spaces + `]*[\r\n]+`),
}

type addedToken struct {
	ID         int32  `json:"id"`
	Content    string `json:"content"`
	Special    bool   `json:"special"`
	SingleWord bool   `json:"single_word"`
	LStrip     bool   `json:"lstrip"`
	RStrip     bool   `json:"rstrip"`
	Normalized bool   `json:"normalized"`
}

type tokenizerFile struct {
	Model struct {
		Type         string            `json:"type"`
		Vocab        map[string]int32  `json:"vocab"`
		Merges       []json.RawMessage `json:"merges"`
		Dropout      *float64          `json:"dropout"`
		ByteFallback bool              `json:"byte_fallback"`
	} `json:"model"`
	Added      []addedToken `json:"added_tokens"`
	Normalizer *struct {
		Type string `json:"type"`
	} `json:"normalizer"`
	PreTokenizer struct {
		Type  string `json:"type"`
		Parts []struct {
			Type    string `json:"type"`
			Pattern struct {
				Regex string `json:"Regex"`
			} `json:"pattern"`
			Behavior       string `json:"behavior"`
			Invert         bool   `json:"invert"`
			AddPrefixSpace bool   `json:"add_prefix_space"`
			UseRegex       bool   `json:"use_regex"`
		} `json:"pretokenizers"`
	} `json:"pre_tokenizer"`
}

// Tokenizer is safe for concurrent Encode and Decode calls. Its cache is bounded.
type Tokenizer struct {
	vocab   map[string]int32
	reverse map[int32]string
	ranks   map[[2]string]int
	added   []addedToken
	nfc     bool
	bytes   [256]rune
	unbytes map[rune]byte
	mu      sync.Mutex
	cache   map[string][]int32
}

func Load(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var f tokenizerFile
	if err := json.Unmarshal(data, &f); err != nil {
		return nil, err
	}
	p := f.PreTokenizer
	if f.Model.Type != "BPE" || f.Model.Dropout != nil || f.Model.ByteFallback || len(f.Model.Vocab) == 0 {
		return nil, fmt.Errorf("bpe: only deterministic byte-level BPE is supported")
	}
	if f.Normalizer != nil && f.Normalizer.Type != "NFC" {
		return nil, fmt.Errorf("bpe: unsupported normalizer %q", f.Normalizer.Type)
	}
	if p.Type != "Sequence" || len(p.Parts) != 2 || p.Parts[0].Type != "Split" || p.Parts[0].Pattern.Regex != splitPattern || p.Parts[0].Behavior != "Isolated" || p.Parts[0].Invert || p.Parts[1].Type != "ByteLevel" || p.Parts[1].AddPrefixSpace || p.Parts[1].UseRegex {
		return nil, fmt.Errorf("bpe: unsupported pre-tokenizer; expected Qwen2 split and ByteLevel")
	}
	t := &Tokenizer{vocab: f.Model.Vocab, reverse: map[int32]string{}, ranks: map[[2]string]int{}, added: f.Added, nfc: f.Normalizer != nil, unbytes: map[rune]byte{}, cache: map[string][]int32{}}
	n := 256
	for b := 0; b < 256; b++ {
		r := rune(b)
		if !(b >= 33 && b <= 126 || b >= 161 && b <= 172 || b >= 174) {
			r = rune(n)
			n++
		}
		t.bytes[b], t.unbytes[r] = r, byte(b)
		if _, ok := t.vocab[string(r)]; !ok {
			return nil, fmt.Errorf("bpe: missing byte token %d", b)
		}
	}
	for token, id := range t.vocab {
		if id < 0 {
			return nil, fmt.Errorf("bpe: negative token ID")
		}
		if _, exists := t.reverse[id]; exists {
			return nil, fmt.Errorf("bpe: duplicate token ID %d", id)
		}
		t.reverse[id] = token
	}
	for rank, raw := range f.Model.Merges {
		var pair []string
		if err := json.Unmarshal(raw, &pair); err != nil {
			var joined string
			if err := json.Unmarshal(raw, &joined); err != nil {
				return nil, err
			}
			pair = strings.Split(joined, " ")
		}
		if len(pair) != 2 {
			return nil, fmt.Errorf("bpe: malformed merge %d", rank)
		}
		if _, ok := t.vocab[pair[0]+pair[1]]; !ok {
			return nil, fmt.Errorf("bpe: missing merged token %d", rank)
		}
		t.ranks[[2]string{pair[0], pair[1]}] = rank
	}
	for _, a := range t.added {
		if a.Content == "" || a.ID < 0 || a.SingleWord || a.LStrip || a.RStrip || a.Normalized {
			return nil, fmt.Errorf("bpe: unsupported added token %q", a.Content)
		}
		if old, ok := t.reverse[a.ID]; ok && old != a.Content {
			return nil, fmt.Errorf("bpe: conflicting added token ID %d", a.ID)
		}
		t.reverse[a.ID] = a.Content
	}
	sort.Slice(t.added, func(i, j int) bool { return len(t.added[i].Content) > len(t.added[j].Content) })
	return t, nil
}

func (t *Tokenizer) SpecialID(content string) (int32, bool) {
	for _, a := range t.added {
		if a.Special && a.Content == content {
			return a.ID, true
		}
	}
	return 0, false
}

// Encode recognizes literal added tokens before normalizing ordinary text.
func (t *Tokenizer) Encode(s string) []int32 {
	ids := make([]int32, 0)
	for s != "" {
		at, found := len(s), -1
		for i, a := range t.added {
			if pos := strings.Index(s, a.Content); pos >= 0 && (pos < at || pos == at && found < 0) {
				at, found = pos, i
			}
		}
		span := s[:at]
		if t.nfc {
			span = norm.NFC.String(span)
		}
		for _, part := range pretokenize(span) {
			ids = append(ids, t.encodePart(part)...)
		}
		if found < 0 {
			break
		}
		ids = append(ids, t.added[found].ID)
		s = s[at+len(t.added[found].Content):]
	}
	return ids
}

func pretokenize(s string) []string {
	var out []string
	for s != "" {
		n := 0
		for _, pattern := range splitAlternatives {
			if loc := pattern.FindStringIndex(s); loc != nil {
				n = loc[1]
				break
			}
		}
		if n == 0 {
			// Implement \s+(?!\S), including regex backtracking by one rune.
			last := 0
			for i, r := range s {
				if !unicode.IsSpace(r) {
					break
				}
				last = i
				n = i + utf8.RuneLen(r)
			}
			if n < len(s) && last > 0 {
				n = last
			}
			if n == 0 {
				_, n = utf8.DecodeRuneInString(s)
			}
		}
		out = append(out, s[:n])
		s = s[n:]
	}
	return out
}

func (t *Tokenizer) byteEncode(s string) string {
	var b strings.Builder
	for i := range len(s) {
		b.WriteRune(t.bytes[s[i]])
	}
	return b.String()
}

func (t *Tokenizer) encodePart(s string) []int32 {
	t.mu.Lock()
	cached, ok := t.cache[s]
	t.mu.Unlock()
	if ok {
		return cached
	}
	parts := make([]string, len(s))
	for i := range parts {
		parts[i] = string(t.bytes[s[i]])
	}
	for len(parts) > 1 {
		best, at := int(^uint(0)>>1), -1
		for i := 0; i+1 < len(parts); i++ {
			if rank, ok := t.ranks[[2]string{parts[i], parts[i+1]}]; ok && rank < best {
				best, at = rank, i
			}
		}
		if at < 0 {
			break
		}
		parts[at] += parts[at+1]
		parts = append(parts[:at+1], parts[at+2:]...)
	}
	ids := make([]int32, len(parts))
	for i, p := range parts {
		ids[i] = t.vocab[p]
	}
	t.mu.Lock()
	if len(t.cache) < 8192 {
		t.cache[s] = ids
	}
	t.mu.Unlock()
	return ids
}

// Decode preserves added tokens literally and replaces unknown IDs with U+FFFD.
func (t *Tokenizer) Decode(ids []int32) string {
	var out strings.Builder
	for _, id := range ids {
		token, ok := t.reverse[id]
		if !ok {
			out.WriteRune(utf8.RuneError)
			continue
		}
		special := false
		for _, a := range t.added {
			if a.ID == id {
				out.WriteString(a.Content)
				special = true
				break
			}
		}
		if special {
			continue
		}
		for _, r := range token {
			out.WriteByte(t.unbytes[r])
		}
	}
	return out.String()
}

func ChatTemplate(prompt string) string {
	return "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
}
