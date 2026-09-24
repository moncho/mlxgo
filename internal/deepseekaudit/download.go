// Package deepseekaudit audits metadata for one pinned DeepSeek release and
// optionally downloads bounded raw validation samples. It does not load models
// or advertise compatibility with the native runtime.
package deepseekaudit

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/moncho/mlxgo/checkpoint"
)

const Repository = "deepseek-ai/DeepSeek-V4.1-Flash"
const Revision = "df42c109f1defefcbfcedbe7d905718a12266e40"
const ConfigSHA256 = "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879"
const IndexSHA256 = "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8"
const ConvertSHA256 = "035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76"
const maxDocument = 16 << 20
const maxDownload = 64 << 20

type Header struct {
	Size   int64  `json:"file_bytes"`
	ETag   string `json:"etag"`
	Prefix []byte `json:"prefix"`
}

// Snapshot contains metadata only. Prefix ends exactly at the payload boundary.
type Snapshot struct {
	Revision string            `json:"revision"`
	Config   []byte            `json:"config"`
	Index    []byte            `json:"index"`
	Convert  []byte            `json:"conversion_source"`
	Headers  map[string]Header `json:"headers"`
}

func digest(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func verifySources(s Snapshot) error {
	if s.Revision != Revision || digest(s.Config) != ConfigSHA256 || digest(s.Index) != IndexSHA256 || digest(s.Convert) != ConvertSHA256 {
		return fmt.Errorf("audit: pinned revision/config/index/conversion checksum mismatch")
	}
	return nil
}

// Download fetches pinned config/index/source and exact header byte ranges.
// A server ignoring Range is rejected before reading its response body.
func Download(ctx context.Context, progress func(string)) (Snapshot, error) {
	client := &http.Client{Timeout: 45 * time.Second, CheckRedirect: func(req *http.Request, via []*http.Request) error {
		if len(via) >= 5 || req.URL.Scheme != "https" {
			return fmt.Errorf("audit: unsafe/excessive redirect")
		}
		return nil
	}}
	return download(ctx, client, "https://huggingface.co/"+Repository+"/resolve/"+Revision+"/", progress)
}

func download(ctx context.Context, client *http.Client, base string, progress func(string)) (Snapshot, error) {
	s := Snapshot{Revision: Revision, Headers: map[string]Header{}}
	f := fetcher{ctx: ctx, client: client}
	convert, _, _, err := f.get(base+"inference/convert.py", -1, -1)
	if err != nil {
		return s, err
	}
	s.Convert = convert
	config, _, _, err := f.get(base+"config.json", -1, -1)
	if err != nil {
		return s, err
	}
	index, _, _, err := f.get(base+"model.safetensors.index.json", -1, -1)
	if err != nil {
		return s, err
	}
	s.Config, s.Index = config, index
	if err = verifySources(s); err != nil {
		return s, err
	}
	i, err := checkpoint.ParseIndex(index)
	if err != nil {
		return s, err
	}
	files := map[string]bool{}
	for _, file := range i.WeightMap {
		files[file] = true
	}
	names := make([]string, 0, len(files))
	for file := range files {
		names = append(names, file)
	}
	sort.Strings(names)
	for _, file := range names {
		if progress != nil {
			progress(file)
		}
		h, err := f.header(base + file)
		if err != nil {
			return s, fmt.Errorf("audit: %s: %w", file, err)
		}
		s.Headers[file] = h
	}
	return s, nil
}

type fetcher struct {
	ctx    context.Context
	client *http.Client
	bytes  int64
	budget int64 // Zero retains the metadata-only default; samples cap at 144 MiB.
}

func (f *fetcher) get(url string, start, end int64) ([]byte, int64, string, error) {
	limit := int64(maxDocument)
	if start >= 0 {
		if end < start || end-start+1 > maxDocument {
			return nil, 0, "", fmt.Errorf("audit: invalid range")
		}
		limit = end - start + 1
		url += "?mlxgo_metadata_range=" + strconv.FormatInt(start, 10) + "-" + strconv.FormatInt(end, 10)
	}
	budget := f.budget
	if budget == 0 {
		budget = maxDownload
	}
	if budget < 0 || budget > maxSampleDownload || f.bytes+limit > budget {
		return nil, 0, "", fmt.Errorf("audit: metadata download budget exceeded")
	}
	req, err := http.NewRequestWithContext(f.ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, 0, "", err
	}
	req.Header.Set("Accept-Encoding", "identity")
	if start >= 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", start, end))
	}
	res, err := f.client.Do(req)
	if err != nil {
		return nil, 0, "", err
	}
	defer res.Body.Close()
	var size int64
	if res.Header.Get("Content-Encoding") != "" && res.Header.Get("Content-Encoding") != "identity" {
		return nil, 0, "", fmt.Errorf("audit: unexpected content encoding")
	}
	if start >= 0 {
		if res.StatusCode != http.StatusPartialContent {
			return nil, 0, "", fmt.Errorf("audit: server did not honor Range (HTTP %d)", res.StatusCode)
		}
		prefix := fmt.Sprintf("bytes %d-%d/", start, end)
		cr := res.Header.Get("Content-Range")
		if !strings.HasPrefix(cr, prefix) {
			return nil, 0, "", fmt.Errorf("audit: wrong Content-Range %q", cr)
		}
		size, err = strconv.ParseInt(strings.TrimPrefix(cr, prefix), 10, 64)
		if err != nil || size <= end {
			return nil, 0, "", fmt.Errorf("audit: invalid range total")
		}
	} else if res.StatusCode != http.StatusOK {
		return nil, 0, "", fmt.Errorf("audit: HTTP %d", res.StatusCode)
	}
	if res.ContentLength > limit {
		return nil, 0, "", fmt.Errorf("audit: response exceeds metadata limit")
	}
	b, err := io.ReadAll(io.LimitReader(res.Body, limit+1))
	f.bytes += int64(len(b))
	if err != nil {
		return nil, 0, "", err
	}
	if int64(len(b)) > limit || (start >= 0 && int64(len(b)) != limit) {
		return nil, 0, "", fmt.Errorf("audit: incorrect response length")
	}
	return b, size, res.Header.Get("ETag"), nil
}

func (f *fetcher) header(url string) (Header, error) {
	var h Header
	length, size, tag, err := f.get(url, 0, 7)
	if err != nil {
		return h, err
	}
	if len(tag) < 2 || tag[0] != '"' || tag[len(tag)-1] != '"' {
		return h, fmt.Errorf("audit: shard requires a strong ETag")
	}
	n := binary.LittleEndian.Uint64(length)
	if n < 2 || n > maxDocument || n > uint64(size-8) {
		return h, fmt.Errorf("audit: invalid header size %d", n)
	}
	b, size2, tag2, err := f.get(url, 8, 7+int64(n))
	if err != nil {
		return h, err
	}
	if size != size2 || tag != tag2 {
		return h, fmt.Errorf("audit: shard changed between range requests")
	}
	h = Header{Size: size, ETag: tag, Prefix: append(length, b...)}
	if _, err = checkpoint.ReadMetadata(bytes.NewReader(h.Prefix), size); err != nil {
		return h, err
	}
	return h, nil
}
