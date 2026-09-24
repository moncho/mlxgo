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
	return sourceHeader(t, sampleShard)
}

func sourceHeader(t *testing.T, shard string) Header {
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
	return s.Headers[shard]
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

func TestExpertPlanAndReuse(t *testing.T) {
	h := sampleHeader(t)
	plan, err := selectedPlan(h, expertNames, expertPayloadBytes)
	if err != nil || len(plan) != 6 {
		t.Fatal(plan, err)
	}
	var total int64
	for _, p := range plan {
		total += p.Bytes
	}
	if total != 18800640 || plan[2].Shape[0] != 5120 || plan[2].Shape[1] != 1152 {
		t.Fatal(plan)
	}
	dir := t.TempDir()
	m := sampleManifest{Repository: Repository, Revision: Revision, Shard: sampleShard, ETag: h.ETag, HeaderSHA256: sampleHeaderSHA256, Tensors: append([]sampleTensor(nil), plan[:2]...)}
	for i := range m.Tensors {
		p := &m.Tensors[i]
		b := make([]byte, p.Bytes)
		p.SHA256 = digest(b)
		if err := os.WriteFile(filepath.Join(dir, p.File), b, 0600); err != nil {
			t.Fatal(err)
		}
	}
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
	var fetched int64
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		var start, end int64
		if _, err := fmt.Sscanf(r.Header.Get("Range"), "bytes=%d-%d", &start, &end); err != nil {
			t.Fatal(err)
		}
		data := make([]byte, end-start+1)
		if start < int64(len(h.Prefix)) {
			copy(data, h.Prefix[start:end+1])
		} else {
			allowed := false
			for _, p := range plan[2:] {
				allowed = allowed || (start >= p.Offsets[0] && end < p.Offsets[1])
			}
			if !allowed {
				t.Fatal("downloaded a reused or unselected range")
			}
			fetched += int64(len(data))
		}
		header := http.Header{}
		header.Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, end, h.Size))
		header.Set("ETag", h.ETag)
		return &http.Response{StatusCode: 206, Header: header, ContentLength: int64(len(data)), Body: io.NopCloser(bytes.NewReader(data))}, nil
	})}
	out := filepath.Join(t.TempDir(), "expert")
	if err := downloadSelection(context.Background(), client, "https://example.test/shard", out, dir, expertNames, expertPayloadBytes, nil); err != nil {
		t.Fatal(err)
	}
	if fetched != 12533760 {
		t.Fatal("wrong incremental payload", fetched)
	}
	for _, p := range plan {
		b, err := os.ReadFile(filepath.Join(out, p.File))
		if err != nil || int64(len(b)) != p.Bytes {
			t.Fatal(p.Name, err)
		}
	}
	// A corrupt cached tensor is an error, not silently replaced or trusted.
	if err := os.WriteFile(filepath.Join(dir, plan[0].File), []byte{1}, 0600); err != nil {
		t.Fatal(err)
	}
	badOut := filepath.Join(t.TempDir(), "failed")
	if err := downloadSelection(context.Background(), client, "https://example.test/shard", badOut, dir, expertNames, expertPayloadBytes, nil); err == nil {
		t.Fatal("accepted corrupt reuse")
	}
	if _, err := os.Stat(badOut); !os.IsNotExist(err) {
		t.Fatal("partial output left behind", err)
	}
	if b, err := os.ReadFile(filepath.Join(dir, plan[0].File)); err != nil || !bytes.Equal(b, []byte{1}) {
		t.Fatal("source modified")
	}
}

func TestReuseRejectsMetadata(t *testing.T) {
	h := sampleHeader(t)
	for _, kind := range []string{"revision", "etag", "duplicate", "oversize"} {
		t.Run(kind, func(t *testing.T) {
			m := sampleManifest{Repository: Repository, Revision: Revision, Shard: sampleShard, ETag: h.ETag, HeaderSHA256: sampleHeaderSHA256}
			switch kind {
			case "revision":
				m.Revision = "different"
			case "etag":
				m.ETag = `"changed"`
			case "duplicate":
				m.Tensors = []sampleTensor{{Name: "a"}, {Name: "a"}}
			}
			b, err := json.Marshal(m)
			if err != nil {
				t.Fatal(err)
			}
			if kind == "oversize" {
				b = make([]byte, (64<<10)+1)
			}
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "manifest.json"), b, 0600); err != nil {
				t.Fatal(err)
			}
			if _, err := layer0Source.readReuseManifest(dir, h); err == nil {
				t.Fatal("accepted invalid manifest")
			}
		})
	}
	_, err := copyReusedSample(io.Discard, t.TempDir(), sampleManifest{Tensors: []sampleTensor{{Name: "w", File: "../outside"}}}, sampleTensor{Name: "w", File: "w.bin"})
	if err == nil {
		t.Fatal("accepted path mismatch")
	}
}

func TestAttentionSelection(t *testing.T) {
	h := sampleHeader(t)
	plan, err := selectedPlan(h, attentionNames, attentionPayloadBytes)
	if err != nil || len(plan) != 14 {
		t.Fatal(plan, err)
	}
	var total, reused int64
	seen := map[string]bool{}
	for _, p := range plan {
		if seen[p.Name] {
			t.Fatal("duplicate attention tensor")
		}
		seen[p.Name] = true
		total += p.Bytes
		if strings.Contains(p.Name, ".wkv.") || strings.Contains(p.Name, ".wo_a.") {
			reused += p.Bytes
		}
	}
	if total != 126753280 || total-reused != 90542080 || total+int64(len(h.Prefix)) > 128<<20 {
		t.Fatal(total, reused)
	}
	if !seen["layers.0.attn_norm.weight"] || !seen["layers.0.attn.attn_sink"] {
		t.Fatal("missing attention norm/sink")
	}
}

func TestExplicitSampleBudget(t *testing.T) {
	client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		h := http.Header{}
		h.Set("Content-Range", "bytes 0-0/100")
		return &http.Response{StatusCode: 206, Header: h, ContentLength: 1, Body: io.NopCloser(strings.NewReader("x"))}, nil
	})}
	for _, tc := range []struct {
		used, budget int64
		ok           bool
	}{
		{maxDownload, 0, false}, {maxDownload, 128 << 20, true},
		{(128 << 20) - 1, 128 << 20, true}, {128 << 20, 128 << 20, false},
		{maxSampleDownload - 1, maxSampleDownload, true},
		{maxSampleDownload, maxSampleDownload, false},
		{0, maxSampleDownload + 1, false}, {0, -1, false},
	} {
		f := fetcher{ctx: context.Background(), client: client, bytes: tc.used, budget: tc.budget}
		_, _, _, err := f.get("https://example.test/shard", 0, 0)
		if (err == nil) != tc.ok {
			t.Fatalf("budget case %+v: %v", tc, err)
		}
	}
}

func TestCompressedAttentionPlan(t *testing.T) {
	h := sourceHeader(t, layer2Source.shard)
	plan, err := layer2Source.plan(h, compressedAttentionNames, compressedAttentionPayloadBytes)
	if err != nil || len(plan) != 22 {
		t.Fatal(plan, err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(h.Prefix[8:], &raw); err != nil {
		t.Fatal(err)
	}
	for _, p := range plan {
		if p.Offsets[0] < int64(len(h.Prefix)) || p.Offsets[1] > h.Size || p.Bytes != p.Offsets[1]-p.Offsets[0] {
			t.Fatal(p)
		}
		delete(raw, p.Name)
	}
	for name := range raw {
		if strings.HasPrefix(name, "layers.2.attn.") || name == "layers.2.attn_norm.weight" {
			t.Fatal("missing", name)
		}
	}
	if compressedAttentionPayloadBytes+int64(len(h.Prefix)) > maxSampleDownload {
		t.Fatal("budget too small")
	}
	if _, err := layer0Source.plan(h, compressedAttentionNames, compressedAttentionPayloadBytes); err == nil {
		t.Fatal("accepted wrong shard")
	}
	if _, err := layer2Source.plan(h, compressedAttentionNames, compressedAttentionPayloadBytes-1); err == nil {
		t.Fatal("accepted wrong size")
	}
	h.Prefix[10] ^= 1
	if _, err := layer2Source.plan(h, compressedAttentionNames, compressedAttentionPayloadBytes); err == nil {
		t.Fatal("accepted changed header")
	}
}

func TestDownloadCompressedAttentionSample(t *testing.T) {
	testDownloadAttentionSource(t, layer2Source, compressedAttentionNames, compressedAttentionPayloadBytes)
}

func TestDownloadConsumerAttentionSample(t *testing.T) {
	testDownloadAttentionSource(t, layer3Source, consumerAttentionNames(), attentionPayloadBytes)
}

func testDownloadAttentionSource(t *testing.T, source sampleSource, names []string, payloadBytes int64) {
	t.Helper()
	h := sourceHeader(t, source.shard)
	plan, err := source.plan(h, names, payloadBytes)
	if err != nil {
		t.Fatal(err)
	}
	var fetched int64
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		var start, end int64
		if _, err := fmt.Sscanf(r.Header.Get("Range"), "bytes=%d-%d", &start, &end); err != nil {
			t.Fatal(err)
		}
		payload := start >= int64(len(h.Prefix))
		allowed := !payload && ((start == 0 && end == 7) || (start == 8 && end == int64(len(h.Prefix))-1))
		for _, p := range plan {
			allowed = allowed || (start >= p.Offsets[0] && end < p.Offsets[1])
		}
		if !allowed || end-start+1 > sampleChunkBytes {
			t.Fatalf("unexpected range %d-%d", start, end)
		}
		data := make([]byte, end-start+1)
		if payload {
			fetched += int64(len(data))
		} else {
			copy(data, h.Prefix[start:end+1])
		}
		header := http.Header{}
		header.Set("ETag", h.ETag)
		header.Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, end, h.Size))
		return &http.Response{StatusCode: 206, Header: header, ContentLength: int64(len(data)), Body: io.NopCloser(bytes.NewReader(data))}, nil
	})}
	out := filepath.Join(t.TempDir(), "sample")
	if err := source.downloadSelection(context.Background(), client, "https://example.test/shard", out, "", names, payloadBytes, nil); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join(out, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	var m sampleManifest
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	if fetched != payloadBytes || m.PayloadBytes != fetched || m.Shard != source.shard || m.HeaderSHA256 != source.headerSHA || m.ETag != h.ETag || m.Revision != Revision || m.Repository != Repository || len(m.Tensors) != len(plan) {
		t.Fatal("incorrect provenance or byte count", m, fetched)
	}
	for _, p := range m.Tensors {
		b, err := os.ReadFile(filepath.Join(out, p.File))
		if err != nil || int64(len(b)) != p.Bytes || digest(b) != p.SHA256 {
			t.Fatal(p, err)
		}
	}
	if _, err := layer0Source.readReuseManifest(out, h); err == nil {
		t.Fatal("accepted cross-shard reuse")
	}
}

func TestConsumerAttentionPlan(t *testing.T) {
	h := sourceHeader(t, layer3Source.shard)
	plan, err := layer3Source.plan(h, consumerAttentionNames(), attentionPayloadBytes)
	if err != nil || len(plan) != 14 {
		t.Fatal(plan, err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(h.Prefix[8:], &raw); err != nil {
		t.Fatal(err)
	}
	for _, p := range plan {
		if p.Offsets[0] < int64(len(h.Prefix)) || p.Offsets[1] > h.Size || p.Bytes != p.Offsets[1]-p.Offsets[0] {
			t.Fatal(p)
		}
		delete(raw, p.Name)
	}
	for name := range raw {
		if strings.HasPrefix(name, "layers.3.attn.") || name == "layers.3.attn_norm.weight" {
			t.Fatal("missing", name)
		}
	}
	if _, err := layer3Source.plan(sourceHeader(t, layer0Source.shard), consumerAttentionNames(), attentionPayloadBytes); err == nil {
		t.Fatal("accepted wrong shard with same file size")
	}
	h.Prefix[10] ^= 1
	if _, err := layer3Source.plan(h, consumerAttentionNames(), attentionPayloadBytes); err == nil {
		t.Fatal("accepted changed header")
	}
}
