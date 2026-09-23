package deepseekaudit

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

const sampleShard = "model-00003-of-00048.safetensors"
const sampleHeaderSHA256 = "ff66dd94d7eb6ef5cc1457b2ac13b422c14e9891914c786af995edaa4f10a614"
const sampleFileBytes int64 = 7389759032
const samplePayloadBytes int64 = 42478080
const sampleChunkBytes int64 = 4 << 20

var sampleNames = [...]string{
	"layers.0.attn.wkv.weight", "layers.0.attn.wkv.scale",
	"layers.0.ffn.experts.0.w1.weight", "layers.0.ffn.experts.0.w1.scale",
	"layers.0.attn.wo_a.weight", "layers.0.attn.wo_a.scale",
}

type sampleTensor struct {
	Name    string   `json:"name"`
	File    string   `json:"file"`
	DType   string   `json:"dtype"`
	Shape   []int    `json:"shape"`
	Offsets [2]int64 `json:"source_byte_offsets"` // Absolute, end-exclusive.
	Bytes   int64    `json:"bytes"`
	SHA256  string   `json:"sha256"`
}

type sampleManifest struct {
	Repository   string         `json:"repository"`
	Revision     string         `json:"revision"`
	Shard        string         `json:"shard"`
	ETag         string         `json:"etag"`
	HeaderSHA256 string         `json:"header_sha256"`
	PayloadBytes int64          `json:"payload_bytes"`
	Tensors      []sampleTensor `json:"tensors"`
}

func samplePlan(h Header) ([]sampleTensor, error) {
	if h.Size != sampleFileBytes || digest(h.Prefix) != sampleHeaderSHA256 {
		return nil, fmt.Errorf("sample: pinned shard header checksum/size mismatch")
	}
	var raw map[string]struct {
		DType   string   `json:"dtype"`
		Shape   []int    `json:"shape"`
		Offsets [2]int64 `json:"data_offsets"`
	}
	if err := json.Unmarshal(h.Prefix[8:], &raw); err != nil {
		return nil, err
	}
	var plan []sampleTensor
	var total int64
	for _, name := range sampleNames {
		t, ok := raw[name]
		if !ok {
			return nil, fmt.Errorf("sample: missing tensor %s", name)
		}
		n := t.Offsets[1] - t.Offsets[0]
		total += n
		base := int64(len(h.Prefix))
		plan = append(plan, sampleTensor{Name: name, File: name + ".bin", DType: t.DType,
			Shape: t.Shape, Offsets: [2]int64{base + t.Offsets[0], base + t.Offsets[1]}, Bytes: n})
	}
	if total != samplePayloadBytes {
		return nil, fmt.Errorf("sample: unexpected payload size %d", total)
	}
	return plan, nil
}

// DownloadSample downloads exactly three pinned weight/scale pairs as raw binary
// files plus a manifest. out must not exist; its parent must exist. It does not
// create a loadable checkpoint or decode weights. Failed downloads are removed.
func DownloadSample(ctx context.Context, out string, progress func(string)) error {
	client := &http.Client{Timeout: 2 * time.Minute, CheckRedirect: func(req *http.Request, via []*http.Request) error {
		if len(via) >= 5 || req.URL.Scheme != "https" {
			return fmt.Errorf("sample: unsafe/excessive redirect")
		}
		return nil
	}}
	url := "https://huggingface.co/" + Repository + "/resolve/" + Revision + "/" + sampleShard
	return downloadSample(ctx, client, url, out, progress)
}

func downloadSample(ctx context.Context, client *http.Client, url, out string, progress func(string)) (err error) {
	if err = os.Mkdir(out, 0700); err != nil {
		return err
	}
	var created []string
	defer func() {
		if err != nil {
			for _, path := range created {
				err = errors.Join(err, os.Remove(path))
			}
			err = errors.Join(err, os.Remove(out))
		}
	}()
	f := fetcher{ctx: ctx, client: client}
	h, err := f.header(url)
	if err != nil {
		return err
	}
	plan, err := samplePlan(h)
	if err != nil {
		return err
	}
	m := sampleManifest{Repository: Repository, Revision: Revision, Shard: sampleShard,
		ETag: h.ETag, HeaderSHA256: sampleHeaderSHA256, PayloadBytes: samplePayloadBytes, Tensors: plan}
	for i := range m.Tensors {
		t := &m.Tensors[i]
		if progress != nil {
			progress(fmt.Sprintf("Fetching %s (%d bytes)", t.Name, t.Bytes))
		}
		path := filepath.Join(out, t.File)
		file, e := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
		if e != nil {
			return e
		}
		created = append(created, path)
		hash := sha256.New()
		e = f.copySample(io.MultiWriter(file, hash), url, h, t.Offsets)
		if e = errors.Join(e, file.Close()); e != nil {
			return e
		}
		t.SHA256 = hex.EncodeToString(hash.Sum(nil))
	}
	path := filepath.Join(out, "manifest.json")
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	created = append(created, path)
	enc := json.NewEncoder(file)
	enc.SetIndent("", "  ")
	return errors.Join(enc.Encode(m), file.Close())
}

func (f *fetcher) copySample(w io.Writer, url string, h Header, offsets [2]int64) error {
	for start := offsets[0]; start < offsets[1]; {
		end := min(start+sampleChunkBytes, offsets[1]) - 1
		b, size, tag, err := f.get(url, start, end)
		if err != nil {
			return err
		}
		if size != h.Size || tag != h.ETag {
			return fmt.Errorf("sample: shard changed during payload download")
		}
		if n, err := w.Write(b); err != nil {
			return err
		} else if n != len(b) {
			return io.ErrShortWrite
		}
		start = end + 1
	}
	return nil
}
