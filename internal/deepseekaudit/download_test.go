package deepseekaudit

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

func TestFetchOnlyHeaders(t *testing.T) {
	h := []byte(`{"a":{"dtype":"F8_E4M3","shape":[32],"data_offsets":[0,32]}}`)
	p := make([]byte, 8)
	binary.LittleEndian.PutUint64(p, uint64(len(h)))
	p = append(p, h...)
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		var start, end int
		if _, err := fmt.Sscanf(r.Header.Get("Range"), "bytes=%d-%d", &start, &end); err != nil || start < 0 || end >= len(p) {
			t.Error("request included payload", r.Header)
			w.WriteHeader(400)
			return
		}
		if r.Header.Get("Accept-Encoding") != "identity" {
			t.Error("encoding not constrained")
		}
		w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, end, len(p)+32))
		w.Header().Set("ETag", `"stable"`)
		w.WriteHeader(206)
		_, _ = w.Write(p[start : end+1])
	}))
	defer server.Close()
	f := fetcher{ctx: context.Background(), client: server.Client()}
	got, err := f.header(server.URL)
	if err != nil || !bytes.Equal(got.Prefix, p) || requests != 2 || f.bytes != int64(len(p)) {
		t.Fatal(got, requests, f.bytes, err)
	}
}

func TestRangeFailures(t *testing.T) {
	for _, tc := range []struct {
		name, cr, encoding, body string
		status                   int
	}{
		{"ignored", "", "", "01234567", 200},
		{"wrong start", "bytes 1-8/100", "", "01234567", 206},
		{"wrong end", "bytes 0-8/100", "", "01234567", 206},
		{"unknown total", "bytes 0-7/*", "", "01234567", 206},
		{"total too small", "bytes 0-7/7", "", "01234567", 206},
		{"short", "bytes 0-7/100", "", "123", 206},
		{"long", "bytes 0-7/100", "", "012345678", 206},
		{"compressed", "bytes 0-7/100", "gzip", "01234567", 206},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Range", tc.cr)
				w.Header().Set("Content-Encoding", tc.encoding)
				w.WriteHeader(tc.status)
				io.WriteString(w, tc.body)
			}))
			defer s.Close()
			f := fetcher{ctx: context.Background(), client: s.Client()}
			if _, _, _, err := f.get(s.URL, 0, 7); err == nil {
				t.Fatal("accepted invalid response")
			}
		})
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return fn(r) }

type forbiddenBody struct{ t *testing.T }

func (b forbiddenBody) Read([]byte) (int, error) {
	b.t.Error("read body after server ignored Range")
	return 0, io.EOF
}
func (b forbiddenBody) Close() error { return nil }

func TestIgnoringRangeNeverReadsBody(t *testing.T) {
	f := fetcher{ctx: context.Background(), client: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Header: http.Header{}, Body: forbiddenBody{t}}, nil
	})}}
	if _, _, _, err := f.get("https://example.test/shard", 0, 7); err == nil {
		t.Fatal("accepted full response")
	}
}

func TestChangedShardAndInvalidLength(t *testing.T) {
	for _, mode := range []string{"size", "etag", "length", "missing etag", "weak etag"} {
		t.Run(mode, func(t *testing.T) {
			s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				length := make([]byte, 8)
				binary.LittleEndian.PutUint64(length, 2)
				if mode == "length" {
					binary.LittleEndian.PutUint64(length, maxDocument+1)
				}
				w.Header().Set("ETag", `"one"`)
				if mode == "missing etag" {
					w.Header().Del("ETag")
				}
				if mode == "weak etag" {
					w.Header().Set("ETag", `W/"one"`)
				}
				w.Header().Set("Content-Range", "bytes 0-7/10")
				b := length
				if r.Header.Get("Range") == "bytes=8-9" {
					b = []byte("{}")
					w.Header().Set("Content-Range", "bytes 8-9/10")
					if mode == "size" {
						w.Header().Set("Content-Range", "bytes 8-9/11")
					}
					if mode == "etag" {
						w.Header().Set("ETag", `"two"`)
					}
				}
				w.WriteHeader(206)
				w.Write(b)
			}))
			defer s.Close()
			f := fetcher{ctx: context.Background(), client: s.Client()}
			if _, err := f.header(s.URL); err == nil {
				t.Fatal("accepted changed/oversized shard")
			}
		})
	}
}

func TestBudgetAndCancellation(t *testing.T) {
	f := fetcher{ctx: context.Background(), client: http.DefaultClient, bytes: maxDownload}
	if _, _, _, err := f.get("http://invalid", 0, 7); err == nil || !strings.Contains(err.Error(), "budget") {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	f = fetcher{ctx: ctx, client: http.DefaultClient}
	if _, _, _, err := f.get("http://invalid", 0, 7); err == nil {
		t.Fatal("ignored cancellation")
	}
}

func TestPinnedDownloadWithMetadataServer(t *testing.T) {
	file, err := os.Open("../../deepseek/testdata/released-checkpoint.metadata.json.gz")
	if err != nil {
		t.Fatal(err)
	}
	s, err := ReadSnapshot(file)
	file.Close()
	if err != nil {
		t.Fatal(err)
	}
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		name := strings.TrimPrefix(r.URL.Path, "/")
		switch name {
		case "config.json":
			w.Write(s.Config)
			return
		case "model.safetensors.index.json":
			w.Write(s.Index)
			return
		case "inference/convert.py":
			w.Write(s.Convert)
			return
		}
		h, ok := s.Headers[name]
		var start, end int
		if _, err := fmt.Sscanf(r.Header.Get("Range"), "bytes=%d-%d", &start, &end); err != nil || !ok || start < 0 || end < start || end >= len(h.Prefix) {
			t.Error("download requested missing metadata or tensor payload", name, r.Header.Get("Range"))
			w.WriteHeader(400)
			return
		}
		w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, end, h.Size))
		w.Header().Set("ETag", h.ETag)
		w.WriteHeader(206)
		w.Write(h.Prefix[start : end+1])
	}))
	defer server.Close()
	got, err := download(context.Background(), server.Client(), server.URL+"/", nil)
	if err != nil {
		t.Fatal(err)
	}
	if requests != 99 || len(got.Headers) != 48 {
		t.Fatal("unexpected download request count", requests)
	}
	for name, h := range s.Headers {
		if x := got.Headers[name]; x.Size != h.Size || x.ETag != h.ETag || !bytes.Equal(x.Prefix, h.Prefix) {
			t.Fatal("changed metadata", name)
		}
	}
}
