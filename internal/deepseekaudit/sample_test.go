package deepseekaudit

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func sampleHeader(t *testing.T) Header {
	t.Helper()
	f, err := os.Open("../../deepseek/testdata/released-checkpoint.metadata.json.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	s, err := ReadSnapshot(f)
	if err != nil {
		t.Fatal(err)
	}
	return s.Headers[sampleShard]
}

func TestSamplePlan(t *testing.T) {
	h := sampleHeader(t)
	plan, err := samplePlan(h)
	if err != nil || len(plan) != 6 {
		t.Fatal(plan, err)
	}
	var total int64
	for _, p := range plan {
		total += p.Bytes
		if p.Offsets[0] < int64(len(h.Prefix)) || p.Offsets[1] > h.Size || p.Bytes != p.Offsets[1]-p.Offsets[0] {
			t.Fatal(p)
		}
	}
	if total != 42478080 || plan[0].DType != "F8_E4M3" || plan[2].DType != "I8" || plan[1].DType != "F8_E8M0" {
		t.Fatal(total, plan)
	}
	h.Prefix[10] ^= 1
	if _, err := samplePlan(h); err == nil {
		t.Fatal("accepted changed header")
	}
}

func TestDownloadSample(t *testing.T) {
	h := sampleHeader(t)
	plan, err := samplePlan(h)
	if err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"ok", "changed etag", "changed size", "ignored range", "short body", "canceled"} {
		t.Run(mode, func(t *testing.T) {
			var received int64
			client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				if err := r.Context().Err(); err != nil {
					return nil, err
				}
				var start, end int64
				if _, err := fmt.Sscanf(r.Header.Get("Range"), "bytes=%d-%d", &start, &end); err != nil {
					t.Fatal("missing Range", err)
				}
				if r.Header.Get("Accept-Encoding") != "identity" {
					t.Fatal("missing identity encoding")
				}
				payload := start >= int64(len(h.Prefix))
				allowed := !payload && ((start == 0 && end == 7) || (start == 8 && end == int64(len(h.Prefix))-1))
				for _, p := range plan {
					allowed = allowed || (start >= p.Offsets[0] && end < p.Offsets[1])
				}
				if !allowed || end-start+1 > sampleChunkBytes {
					t.Fatalf("unexpected request %d-%d", start, end)
				}
				data := make([]byte, end-start+1)
				if !payload {
					copy(data, h.Prefix[start:end+1])
				} else {
					received += int64(len(data))
				}
				size, tag := h.Size, h.ETag
				status := 206
				if payload {
					switch mode {
					case "changed etag":
						tag = `"changed"`
					case "changed size":
						size++
					case "ignored range":
						return &http.Response{StatusCode: 200, Header: http.Header{}, Body: forbiddenBody{t}}, nil
					case "short body":
						data = data[:len(data)-1]
					}
				}
				header := http.Header{}
				header.Set("ETag", tag)
				header.Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, end, size))
				return &http.Response{StatusCode: status, Header: header, ContentLength: int64(len(data)), Body: io.NopCloser(bytes.NewReader(data))}, nil
			})}
			out := filepath.Join(t.TempDir(), "sample")
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if mode == "canceled" {
				cancel()
			}
			err := downloadSample(ctx, client, "https://example.test/shard", out, nil)
			if mode != "ok" {
				if err == nil {
					t.Fatal("accepted bad response")
				}
				if _, err := os.Stat(out); !os.IsNotExist(err) {
					t.Fatal("failed download not cleaned up", err)
				}
				return
			}
			if err != nil || received != samplePayloadBytes {
				t.Fatal(err, received)
			}
			b, err := os.ReadFile(filepath.Join(out, "manifest.json"))
			if err != nil {
				t.Fatal(err)
			}
			var m sampleManifest
			if err := json.Unmarshal(b, &m); err != nil {
				t.Fatal(err)
			}
			if m.Revision != Revision || m.PayloadBytes != samplePayloadBytes || len(m.Tensors) != 6 {
				t.Fatal(m)
			}
			for _, p := range m.Tensors {
				b, err := os.ReadFile(filepath.Join(out, p.File))
				if err != nil || int64(len(b)) != p.Bytes || digest(b) != p.SHA256 {
					t.Fatal(p, err)
				}
			}
			if err := downloadSample(ctx, client, "https://example.test/shard", out, nil); err == nil || !strings.Contains(err.Error(), "file exists") {
				t.Fatal("did not preserve existing output", err)
			}
			if _, err := os.Stat(filepath.Join(out, "manifest.json")); err != nil {
				t.Fatal("removed existing output", err)
			}
		})
	}
}
