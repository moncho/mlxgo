package ticketbench

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// LoadVerified checks every split's bytes, disjoint IDs, prompts, and families.
// It fails closed on edits so a report cannot silently refer to different data.
func LoadVerified(dir string) (Manifest, map[string][]Record, error) {
	var manifest Manifest
	data, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return manifest, nil, err
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		return manifest, nil, err
	}
	if manifest.Version != Version {
		return manifest, nil, fmt.Errorf("unsupported benchmark version %q", manifest.Version)
	}
	all := map[string][]Record{}
	ids, prompts, families := map[string]bool{}, map[string]bool{}, map[string]string{}
	for _, split := range []string{"train", "valid", "test", "retention"} {
		name := split + ".jsonl"
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			return manifest, nil, err
		}
		hash := sha256.Sum256(data)
		if hex.EncodeToString(hash[:]) != manifest.SHA256[name] {
			return manifest, nil, fmt.Errorf("%s: dataset hash mismatch", name)
		}
		scanner := bufio.NewScanner(bytes.NewReader(data))
		scanner.Buffer(make([]byte, 4096), 4<<20)
		for scanner.Scan() {
			var record Record
			if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
				return manifest, nil, err
			}
			if record.ID == "" || record.Family == "" || record.Prompt == "" || record.Completion == "" || ids[record.ID] || prompts[record.Prompt] {
				return manifest, nil, fmt.Errorf("%s: empty or duplicate record %q", split, record.ID)
			}
			if previous, found := families[record.Family]; found && previous != split {
				return manifest, nil, fmt.Errorf("family %q leaks across splits", record.Family)
			}
			if split != "retention" {
				f, err := parseFields(record.Completion)
				if err != nil || f.TicketID != record.ID {
					return manifest, nil, fmt.Errorf("%s: invalid reference %q", split, record.ID)
				}
			}
			ids[record.ID], prompts[record.Prompt], families[record.Family] = true, true, split
			all[split] = append(all[split], record)
		}
		if err := scanner.Err(); err != nil {
			return manifest, nil, err
		}
		if len(all[split]) == 0 || len(all[split]) != manifest.Counts[split] {
			return manifest, nil, fmt.Errorf("%s: dataset count mismatch", split)
		}
	}
	return manifest, all, nil
}
