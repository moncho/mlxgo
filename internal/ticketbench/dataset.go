// Package ticketbench defines a deterministic synthetic extraction benchmark.
package ticketbench

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

const Version = "tickets-v1"

const Instruction = `Extract the ticket as JSON with exactly three string fields: "ticket_id", "queue", "priority". Copy the ticket ID exactly. Queue: billing for invoices or payments; access for sign-in or permissions; bug for software malfunctions; feature for new functionality. Priority: high if everyone is blocked; normal if only one person is blocked; low if nobody is blocked. Return only the JSON object, without markdown.

`

type Fields struct {
	TicketID string `json:"ticket_id"`
	Queue    string `json:"queue"`
	Priority string `json:"priority"`
}

type Record struct {
	ID         string `json:"id"`
	Family     string `json:"family"`
	Prompt     string `json:"prompt"`
	Completion string `json:"completion"`
}

type Manifest struct {
	Version string            `json:"version"`
	Origin  string            `json:"origin"`
	SHA256  map[string]string `json:"sha256"`
	Counts  map[string]int    `json:"counts"`
}

// Dataset uses disjoint rendering families and issue phrasings per split.
// The schema and label policy are deliberately shared: those define the task.
func Dataset() map[string][]Record {
	queues := []string{"billing", "access", "bug", "feature"}
	priorities := []string{"high", "normal", "low"}
	issues := [][]string{
		{"The invoice includes a duplicate charge.", "The sign-in page refuses valid passwords.", "The export function crashes instead of downloading a file.", "We need a new calendar integration."},
		{"A payment was collected twice.", "The account permissions prevent access to the workspace.", "The search results disappear due to a software error.", "Please add an option to schedule reports."},
		{"Our subscription invoice has the wrong amount.", "The login verification code never arrives.", "The editor fails to save changes because of a software malfunction.", "Could you build a dark mode setting?"},
	}
	impact := [][]string{
		{"Everyone is blocked from working.", "Only one person is blocked; everyone else can work.", "Nobody is blocked from working."},
		{"All users are blocked by this issue.", "A single user is blocked, but the rest can continue.", "No users are blocked; work continues normally."},
		{"This blocks every person on the team.", "This blocks just one person, not the rest of the team.", "This does not block anyone on the team."},
	}
	families := [][]string{
		{"Ticket %s\nIssue: %s\nImpact: %s", "Reference: %s. %s %s", "Support request %s\n%s\n%s", "ID=%s\nCustomer reports: %s\nCurrent impact: %s"},
		{"Incoming case %s: %s The reported scope is: %s"},
		{"Helpdesk note [%s]\nProblem description: %s\nWho is affected? %s", "Please triage case %s. Customer message: %s Operational status: %s"},
	}
	result := map[string][]Record{}
	for split, name := range []string{"train", "valid", "test"} {
		serial := 0
		for f, format := range families[split] {
			for q, queue := range queues {
				for p, priority := range priorities {
					for variant := range 4 {
						serial++
						ticket := fmt.Sprintf("TK-%d", (split+1)*10000+serial*7)
						answer, _ := json.Marshal(Fields{ticket, queue, priority})
						// Irrelevant metadata tests that routing follows issue and impact,
						// not customer tier or communication channel.
						noise := []string{"Channel: email.", "Customer plan: standard.", "Channel: web form.", "Customer plan: premium."}[variant]
						result[name] = append(result[name], Record{
							ID: ticket, Family: fmt.Sprintf("%s-%d", name, f),
							Prompt:     Instruction + fmt.Sprintf(format, ticket, issues[split][q], impact[split][p]) + "\n" + noise,
							Completion: string(answer),
						})
					}
				}
			}
		}
	}
	for i, pair := range [][2]string{
		{"What is 8 plus 7? Reply with only the number.", "15"},
		{"What is 9 minus 4? Reply with only the number.", "5"},
		{"Write the word HELLO in lowercase. Reply with only that word.", "hello"},
		{"What is the opposite of cold? Reply with one lowercase word.", "hot"},
		{"Sort these numbers from smallest to largest: 3, 1, 2. Reply only with a JSON array.", "[1,2,3]"},
		{"How many days are in one week? Reply with only the number.", "7"},
		{"Repeat exactly this word and nothing else: maple", "maple"},
		{"Translate the Spanish word 'gracias' into English. Reply with only the two lowercase words.", "thank you"},
	} {
		result["retention"] = append(result["retention"], Record{ID: fmt.Sprintf("retention-%d", i), Family: "retention", Prompt: pair[0], Completion: pair[1]})
	}
	return result
}

func Prepare(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	m := Manifest{Version: Version, Origin: "Synthetic, handwritten templates; no customer data. Not a real-world capability benchmark.", SHA256: map[string]string{}, Counts: map[string]int{}}
	for name, records := range Dataset() {
		var data []byte
		for _, record := range records {
			line, err := json.Marshal(record)
			if err != nil {
				return err
			}
			data = append(append(data, line...), '\n')
		}
		filename := name + ".jsonl"
		hash := sha256.Sum256(data)
		m.SHA256[filename] = hex.EncodeToString(hash[:])
		m.Counts[name] = len(records)
		if err := os.WriteFile(filepath.Join(dir, filename), data, 0o644); err != nil {
			return err
		}
	}
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "manifest.json"), append(data, '\n'), 0o644)
}
