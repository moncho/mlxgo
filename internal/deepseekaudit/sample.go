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
	"slices"
	"time"
)

const sampleShard = "model-00003-of-00048.safetensors"
const sampleHeaderSHA256 = "ff66dd94d7eb6ef5cc1457b2ac13b422c14e9891914c786af995edaa4f10a614"
const sampleFileBytes int64 = 7389759032
const samplePayloadBytes int64 = 42478080
const sampleChunkBytes int64 = 4 << 20
const expertPayloadBytes int64 = 18800640
const attentionPayloadBytes int64 = 126753280

var attentionNames = []string{
	"layers.0.attn.wkv.weight", "layers.0.attn.wkv.scale",
	"layers.0.attn.wo_a.weight", "layers.0.attn.wo_a.scale",
	"layers.0.attn.wq_a.weight", "layers.0.attn.wq_a.scale",
	"layers.0.attn.wq_b.weight", "layers.0.attn.wq_b.scale",
	"layers.0.attn.wo_b.weight", "layers.0.attn.wo_b.scale",
	"layers.0.attn.q_norm.weight", "layers.0.attn.kv_norm.weight",
	"layers.0.attn.attn_sink", "layers.0.attn_norm.weight",
}

var expertNames = []string{
	"layers.0.ffn.experts.0.w1.weight", "layers.0.ffn.experts.0.w1.scale",
	"layers.0.ffn.experts.0.w2.weight", "layers.0.ffn.experts.0.w2.scale",
	"layers.0.ffn.experts.0.w3.weight", "layers.0.ffn.experts.0.w3.scale",
}

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
	return selectedPlan(h, sampleNames[:], samplePayloadBytes)
}

func selectedPlan(h Header, names []string, expectedBytes int64) ([]sampleTensor, error) {
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
	for _, name := range names {
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
	if total != expectedBytes {
		return nil, fmt.Errorf("sample: unexpected payload size %d", total)
	}
	return plan, nil
}

// DownloadSample downloads exactly three pinned weight/scale pairs as raw binary
// files plus a manifest. out must not exist; its parent must exist. It does not
// create a loadable checkpoint or decode weights. Failed downloads are removed.
func DownloadSample(ctx context.Context, out string, progress func(string)) error {
	return downloadSelected(ctx, out, "", sampleNames[:], samplePayloadBytes, progress)
}

// DownloadExpertSample fetches all three weight/scale pairs for layer 0 expert 0.
// reuse optionally points to a previous sample directory: matching files are
// copied after checking layout and SHA-256, never modified or blindly trusted.
func DownloadExpertSample(ctx context.Context, out, reuse string, progress func(string)) error {
	return downloadSelected(ctx, out, reuse, expertNames, expertPayloadBytes, progress)
}

// DownloadAttentionSample fetches the complete layer-0 sliding-window attention
// and input norm. The optional reuse directory is verified as for expert samples.
func DownloadAttentionSample(ctx context.Context, out, reuse string, progress func(string)) error {
	return downloadSelected(ctx, out, reuse, attentionNames, attentionPayloadBytes, progress)
}

func downloadSelected(ctx context.Context, out, reuse string, names []string, expectedBytes int64, progress func(string)) error {
	client := &http.Client{Timeout: 2 * time.Minute, CheckRedirect: func(req *http.Request, via []*http.Request) error {
		if len(via) >= 5 || req.URL.Scheme != "https" {
			return fmt.Errorf("sample: unsafe/excessive redirect")
		}
		return nil
	}}
	url := "https://huggingface.co/" + Repository + "/resolve/" + Revision + "/" + sampleShard
	return downloadSelection(ctx, client, url, out, reuse, names, expectedBytes, progress)
}

func downloadSample(ctx context.Context, client *http.Client, url, out string, progress func(string)) error {
	return downloadSelection(ctx, client, url, out, "", sampleNames[:], samplePayloadBytes, progress)
}

func downloadSelection(ctx context.Context, client *http.Client, url, out, reuse string, names []string, expectedBytes int64, progress func(string)) (err error) {
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
	plan, err := selectedPlan(h, names, expectedBytes)
	if err != nil {
		return err
	}
	f.budget = int64(len(h.Prefix)) + expectedBytes
	if f.budget > 128<<20 {
		return fmt.Errorf("sample: selection exceeds download budget")
	}
	m := sampleManifest{Repository: Repository, Revision: Revision, Shard: sampleShard,
		ETag: h.ETag, HeaderSHA256: sampleHeaderSHA256, PayloadBytes: expectedBytes, Tensors: plan}
	var previous sampleManifest
	if reuse != "" {
		previous, err = readReuseManifest(reuse, h)
		if err != nil {
			return err
		}
	}
	for i := range m.Tensors {
		t := &m.Tensors[i]
		if err := ctx.Err(); err != nil {
			return err
		}
		path := filepath.Join(out, t.File)
		file, e := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
		if e != nil {
			return e
		}
		created = append(created, path)
		hash := sha256.New()
		writer := io.MultiWriter(file, hash)
		var reused bool
		reused, e = copyReusedSample(writer, reuse, previous, *t)
		if e == nil && !reused {
			if progress != nil {
				progress(fmt.Sprintf("Fetching %s (%d bytes)", t.Name, t.Bytes))
			}
			e = f.copySample(writer, url, h, t.Offsets)
		} else if e == nil && progress != nil {
			progress("Reused " + t.Name)
		}
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

func readReuseManifest(dir string, h Header) (sampleManifest, error) {
	var m sampleManifest
	f, err := os.Open(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return m, err
	}
	defer f.Close()
	b, err := io.ReadAll(io.LimitReader(f, (64<<10)+1))
	if err != nil || len(b) > 64<<10 {
		return m, fmt.Errorf("sample: invalid reuse manifest: %v", err)
	}
	if err := json.Unmarshal(b, &m); err != nil {
		return m, err
	}
	if m.Repository != Repository || m.Revision != Revision || m.Shard != sampleShard || m.HeaderSHA256 != sampleHeaderSHA256 || m.ETag != h.ETag {
		return m, fmt.Errorf("sample: reuse provenance mismatch")
	}
	seen := map[string]bool{}
	for _, t := range m.Tensors {
		if seen[t.Name] {
			return m, fmt.Errorf("sample: duplicate reuse entry")
		}
		seen[t.Name] = true
	}
	return m, nil
}

func copyReusedSample(w io.Writer, dir string, m sampleManifest, want sampleTensor) (bool, error) {
	for _, t := range m.Tensors {
		if t.Name != want.Name {
			continue
		}
		if t.File != want.File || t.DType != want.DType || !slices.Equal(t.Shape, want.Shape) || t.Bytes != want.Bytes || t.Offsets != want.Offsets {
			return false, fmt.Errorf("sample: reuse layout mismatch for %s", t.Name)
		}
		f, err := os.Open(filepath.Join(dir, want.File))
		if err != nil {
			return false, err
		}
		defer f.Close()
		h := sha256.New()
		n, err := io.CopyN(io.MultiWriter(w, h), f, want.Bytes)
		var extra [1]byte
		_, end := io.ReadFull(f, extra[:])
		if err != nil || n != want.Bytes || end != io.EOF || hex.EncodeToString(h.Sum(nil)) != t.SHA256 {
			return false, fmt.Errorf("sample: reuse length/checksum mismatch for %s", t.Name)
		}
		return true, nil
	}
	return false, nil
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
