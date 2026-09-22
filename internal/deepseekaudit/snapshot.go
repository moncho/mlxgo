package deepseekaudit

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
)

func ReadSnapshot(r io.Reader) (Snapshot, error) {
	var s Snapshot
	z, err := gzip.NewReader(r)
	if err != nil {
		return s, err
	}
	defer z.Close()
	d := json.NewDecoder(io.LimitReader(z, 128<<20))
	d.DisallowUnknownFields()
	if err = d.Decode(&s); err != nil {
		return s, err
	}
	var extra any
	if err = d.Decode(&extra); err != io.EOF {
		return s, fmt.Errorf("audit: trailing or oversized snapshot data: %v", err)
	}
	return s, nil
}

func WriteSnapshot(w io.Writer, s Snapshot) error {
	z := gzip.NewWriter(w)
	if err := json.NewEncoder(z).Encode(s); err != nil {
		z.Close()
		return err
	}
	return z.Close()
}
