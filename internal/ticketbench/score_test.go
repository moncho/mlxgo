package ticketbench

import "testing"

func TestScoreExtraction(t *testing.T) {
	const want = `{"ticket_id":"TK-1","queue":"bug","priority":"high"}`
	for _, tc := range []struct {
		name, text           string
		valid, schema, exact bool
	}{
		{"exact", want, true, true, true},
		{"reordered", " {\n\"priority\": \"high\", \"queue\": \"bug\", \"ticket_id\": \"TK-1\" } ", true, true, true},
		{"wrong", `{"ticket_id":"TK-1","queue":"access","priority":"high"}`, true, true, false},
		{"fenced", "```json\n" + want + "\n```", false, false, false},
		{"trailing", want + " thanks", false, false, false},
		{"second object", want + want, false, false, false},
		{"extra", `{"ticket_id":"TK-1","queue":"bug","priority":"high","reason":"error"}`, true, false, false},
		{"duplicate", `{"ticket_id":"TK-1","queue":"bug","priority":"high","priority":"high"}`, true, false, false},
		{"missing", `{"ticket_id":"TK-1","queue":"bug"}`, true, false, false},
		{"null", `{"ticket_id":null,"queue":"bug","priority":"high"}`, true, false, false},
		{"number", `{"ticket_id":1,"queue":"bug","priority":"high"}`, true, false, false},
		{"enum", `{"ticket_id":"TK-1","queue":"bugs","priority":"high"}`, true, false, false},
		{"array", "[" + want + "]", true, false, false},
		{"empty", "", false, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, err := ScoreExtraction(tc.text, want)
			if err != nil || s.ValidJSON != tc.valid || s.Schema != tc.schema || s.Exact != tc.exact {
				t.Fatalf("got %+v err=%v", s, err)
			}
			if tc.name == "wrong" && (!s.TicketID || s.Queue || !s.Priority) {
				t.Fatalf("incorrect field scores: %+v", s)
			}
		})
	}
	if _, err := ScoreExtraction(want, `{}`); err == nil {
		t.Fatal("accepted invalid reference")
	}
}

func TestScoreRetention(t *testing.T) {
	if !ScoreRetention(" 15\n", "15") || ScoreRetention("The answer is 15", "15") {
		t.Fatal("text scoring mismatch")
	}
	if !ScoreRetention("[1, 2, 3]", "[1,2,3]") || ScoreRetention("[1,2,3,4]", "[1,2,3]") {
		t.Fatal("array scoring mismatch")
	}
}
