// Package checkpoint reads local single-file or indexed safetensors bundles.
// Tensor decoding stays in MLX; Go validates metadata and shard routing first.
package checkpoint

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"
)

const indexName = "model.safetensors.index.json"
const singleName = "model.safetensors"
const maxJSONBytes = 16 << 20

type layout struct {
	dir       string
	sharded   bool
	files     []string
	weightMap map[string]string
	totalSize *int64
}

// object rejects duplicate JSON keys, including differently escaped spellings.
// Encoding/json's normal map decoding silently keeps the last duplicate.
func object(data []byte) (map[string]json.RawMessage, error) {
	if !utf8.Valid(data) {
		return nil, fmt.Errorf("invalid UTF-8 JSON")
	}
	d := json.NewDecoder(bytes.NewReader(data))
	t, err := d.Token()
	if err != nil {
		return nil, err
	}
	if t != json.Delim('{') {
		return nil, fmt.Errorf("expected JSON object")
	}
	out := map[string]json.RawMessage{}
	for d.More() {
		t, err := d.Token()
		if err != nil {
			return nil, err
		}
		key, ok := t.(string)
		if !ok {
			return nil, fmt.Errorf("expected object key")
		}
		if _, exists := out[key]; exists {
			return nil, fmt.Errorf("duplicate JSON key %q", key)
		}
		var value json.RawMessage
		if err = d.Decode(&value); err != nil {
			return nil, err
		}
		out[key] = value
	}
	if _, err = d.Token(); err != nil {
		return nil, err
	}
	if _, err = d.Token(); err != io.EOF {
		return nil, fmt.Errorf("trailing JSON data")
	}
	return out, nil
}

func readJSON(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("checkpoint: %s is not a regular file", path)
	}
	b, err := io.ReadAll(io.LimitReader(f, maxJSONBytes+1))
	if err != nil {
		return nil, err
	}
	if len(b) > maxJSONBytes {
		return nil, fmt.Errorf("checkpoint: JSON exceeds %d bytes", maxJSONBytes)
	}
	return b, nil
}

func exists(path string) (bool, error) {
	_, err := os.Lstat(path)
	if os.IsNotExist(err) {
		return false, nil
	}
	return err == nil, err
}

func discover(dir string) (layout, error) {
	var l layout
	abs, err := filepath.Abs(dir)
	if err != nil {
		return l, err
	}
	l.dir, err = filepath.EvalSymlinks(abs)
	if err != nil {
		return l, err
	}
	info, err := os.Stat(l.dir)
	if err != nil {
		return l, err
	}
	if !info.IsDir() {
		return l, fmt.Errorf("checkpoint: expected model directory")
	}
	index, err := exists(filepath.Join(l.dir, indexName))
	if err != nil {
		return l, err
	}
	single, err := exists(filepath.Join(l.dir, singleName))
	if err != nil {
		return l, err
	}
	if index && single {
		return l, fmt.Errorf("checkpoint: ambiguous bundle contains both %s and %s", singleName, indexName)
	}
	if !index {
		l.files = []string{singleName}
		return l, nil
	}
	l.sharded = true
	path, err := l.localFile(indexName)
	if err != nil {
		return l, err
	}
	b, err := readJSON(path)
	if err != nil {
		return l, err
	}
	root, err := object(b)
	if err != nil {
		return l, fmt.Errorf("checkpoint: index: %w", err)
	}
	weights, err := object(root["weight_map"])
	if err != nil {
		return l, fmt.Errorf("checkpoint: weight_map: %w", err)
	}
	if len(weights) == 0 {
		return l, fmt.Errorf("checkpoint: empty weight_map")
	}
	l.weightMap = make(map[string]string, len(weights))
	files := map[string]bool{}
	for name, raw := range weights {
		var file string
		if name == "" || name == "__metadata__" || strings.ContainsRune(name, 0) {
			return l, fmt.Errorf("checkpoint: invalid tensor name %q", name)
		}
		if err = json.Unmarshal(raw, &file); err != nil || file == "" || strings.ContainsAny(file, "/\\:\x00") || !strings.HasSuffix(file, ".safetensors") {
			return l, fmt.Errorf("checkpoint: tensor %q has invalid shard filename", name)
		}
		l.weightMap[name] = file
		files[file] = true
	}
	for file := range files {
		l.files = append(l.files, file)
	}
	sort.Strings(l.files)
	if raw, ok := root["metadata"]; ok {
		metadata, e := object(raw)
		if e != nil {
			return l, fmt.Errorf("checkpoint: index metadata: %w", e)
		}
		if raw, ok = metadata["total_size"]; ok {
			var n int64
			if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) || json.Unmarshal(raw, &n) != nil || n < 0 {
				return l, fmt.Errorf("checkpoint: invalid total_size")
			}
			l.totalSize = &n
		}
	}
	return l, nil
}

// Shard names are basenames, never arbitrary paths. Symlinks may resolve within
// the chosen bundle, but never outside it. Bundles must remain immutable during
// inspection/loading/use; this is not a sandbox for an actively changing tree.
func (l layout) localFile(name string) (string, error) {
	path, err := filepath.EvalSymlinks(filepath.Join(l.dir, name))
	if err != nil {
		return "", err
	}
	rel, err := filepath.Rel(l.dir, path)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("checkpoint: %s resolves outside model directory", name)
	}
	info, err := os.Stat(path)
	if err != nil {
		return "", err
	}
	if !info.Mode().IsRegular() {
		return "", fmt.Errorf("checkpoint: %s is not a regular file", name)
	}
	return path, nil
}

// SourceFiles returns absolute input paths, including the index when present.
// It validates index syntax and shard names but does not read tensor headers.
// A single-file path is returned even if absent, for output-overwrite checks.
func SourceFiles(dir string) ([]string, error) {
	l, err := discover(dir)
	if err != nil {
		return nil, err
	}
	files := make([]string, 0, len(l.files)+1)
	if l.sharded {
		files = append(files, filepath.Join(l.dir, indexName))
	}
	for _, name := range l.files {
		files = append(files, filepath.Join(l.dir, name))
	}
	return files, nil
}
