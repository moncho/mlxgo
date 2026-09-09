package ticketbench

import (
	"encoding/json"
	"fmt"
	"io"
	"slices"
	"strings"
)

type Score struct {
	ValidJSON bool `json:"valid_json"`
	Schema    bool `json:"schema_valid"`
	Exact     bool `json:"exact"`
	TicketID  bool `json:"ticket_id"`
	Queue     bool `json:"queue"`
	Priority  bool `json:"priority"`
}

// ScoreExtraction accepts key reordering and JSON whitespace, but not fences,
// trailing text, extra fields, duplicate keys, nulls or coerced value types.
func ScoreExtraction(output, expected string) (Score, error) {
	want, err := parseFields(expected)
	if err != nil {
		return Score{}, fmt.Errorf("invalid reference: %w", err)
	}
	s := Score{ValidJSON: json.Valid([]byte(output))}
	got, err := parseFields(output)
	if err != nil {
		return s, nil
	}
	s.Schema = true
	s.TicketID, s.Queue, s.Priority = got.TicketID == want.TicketID, got.Queue == want.Queue, got.Priority == want.Priority
	s.Exact = s.TicketID && s.Queue && s.Priority
	return s, nil
}

func parseFields(text string) (Fields, error) {
	d := json.NewDecoder(strings.NewReader(text))
	token, err := d.Token()
	if err != nil || token != json.Delim('{') {
		return Fields{}, fmt.Errorf("expected object")
	}
	fields := map[string]string{}
	for d.More() {
		key, err := d.Token()
		if err != nil {
			return Fields{}, err
		}
		name, ok := key.(string)
		if !ok {
			return Fields{}, fmt.Errorf("expected string key")
		}
		if _, exists := fields[name]; exists {
			return Fields{}, fmt.Errorf("duplicate key")
		}
		var value any
		if err := d.Decode(&value); err != nil {
			return Fields{}, err
		}
		str, ok := value.(string)
		if !ok {
			return Fields{}, fmt.Errorf("expected string value")
		}
		fields[name] = str
	}
	if _, err := d.Token(); err != nil {
		return Fields{}, err
	}
	if _, err := d.Token(); err != io.EOF {
		return Fields{}, fmt.Errorf("trailing content")
	}
	f := Fields{fields["ticket_id"], fields["queue"], fields["priority"]}
	if len(fields) != 3 || f.TicketID == "" || !oneOf(f.Queue, "billing", "access", "bug", "feature") || !oneOf(f.Priority, "high", "normal", "low") {
		return Fields{}, fmt.Errorf("schema mismatch")
	}
	return f, nil
}

func oneOf(value string, options ...string) bool {
	for _, option := range options {
		if value == option {
			return true
		}
	}
	return false
}

func ScoreRetention(output, expected string) bool {
	if strings.HasPrefix(expected, "[") {
		var got, want []int
		return json.Unmarshal([]byte(expected), &want) == nil && json.Unmarshal([]byte(output), &got) == nil && slices.Equal(got, want)
	}
	return strings.TrimSpace(output) == expected
}
