package checkpoint

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"sort"
	"strings"
)

var ErrUnsupportedDType = errors.New("checkpoint: unsupported storage dtype")

// Tensor describes on-disk metadata; Shape and returned slices are copies.
type Tensor struct {
	File  string
	DType string
	Shape []int
	Bytes int64
}

// Manifest is a validated snapshot of local index and safetensors headers.
// Inspection reads no tensor payload and requires no native MLX runtime.
type Manifest struct {
	layout  layout
	tensors map[string]Tensor
	paths   map[string]string
	stats   map[string]os.FileInfo
}

func (m *Manifest) Sharded() bool { return m != nil && m.layout.sharded }
func (m *Manifest) Names() []string {
	if m == nil {
		return nil
	}
	names := make([]string, 0, len(m.tensors))
	for name := range m.tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}
func (m *Manifest) Tensor(name string) (Tensor, bool) {
	if m == nil {
		return Tensor{}, false
	}
	t, ok := m.tensors[name]
	t.Shape = slices.Clone(t.Shape)
	return t, ok
}

// Inspect validates routing, duplicate names, shapes, byte counts, contiguous
// offsets and file sizes before any array allocation. FP8/FP4 and unknown dtypes
// are rejected; sharding does not add quantization support.
func Inspect(dir string) (*Manifest, error) {
	l, err := discover(dir)
	if err != nil {
		return nil, err
	}
	m := &Manifest{layout: l, tensors: map[string]Tensor{}, paths: map[string]string{}, stats: map[string]os.FileInfo{}}
	var total int64
	for _, file := range l.files {
		path, err := l.localFile(file)
		if err != nil {
			return nil, fmt.Errorf("checkpoint: shard %s: %w", file, err)
		}
		tensors, info, err := readHeader(path)
		if err != nil {
			return nil, fmt.Errorf("checkpoint: shard %s: %w", file, err)
		}
		m.paths[file] = path
		m.stats[file] = info
		for name, t := range tensors {
			if strings.HasPrefix(t.DType, "F8_") {
				return nil, fmt.Errorf("%w %q for %s", ErrUnsupportedDType, t.DType, name)
			}
			if _, ok := m.tensors[name]; ok {
				return nil, fmt.Errorf("checkpoint: tensor %q occurs in multiple shards", name)
			}
			if l.sharded && l.weightMap[name] != file {
				return nil, fmt.Errorf("checkpoint: tensor %q in %s disagrees with weight_map", name, file)
			}
			t.File = file
			m.tensors[name] = t
			if t.Bytes > math.MaxInt64-total {
				return nil, fmt.Errorf("checkpoint: total tensor size overflow")
			}
			total += t.Bytes
		}
	}
	if l.sharded {
		for name, file := range l.weightMap {
			if _, ok := m.tensors[name]; !ok {
				return nil, fmt.Errorf("checkpoint: indexed tensor %q missing from %s", name, file)
			}
		}
		if l.totalSize != nil && *l.totalSize != total {
			return nil, fmt.Errorf("checkpoint: total_size=%d, actual tensor bytes=%d", *l.totalSize, total)
		}
	}
	return m, nil
}

func readHeader(path string) (map[string]Tensor, os.FileInfo, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, nil, err
	}
	if !info.Mode().IsRegular() || info.Size() < 8 {
		return nil, nil, fmt.Errorf("truncated or nonregular safetensors file")
	}
	tensors, err := ReadMetadata(f, info.Size())
	return tensors, info, err
}

// ReadMetadata reads only the safetensors length prefix and JSON header from r.
// fileSize is the full file size, including payload; offsets are checked against
// it without reading payload bytes. Unlike Inspect, metadata inspection accepts
// F8_E4M3 and F8_E8M0 storage. This does not add native FP8 decoding support.
func ReadMetadata(r io.Reader, fileSize int64) (map[string]Tensor, error) {
	if fileSize < 8 {
		return nil, fmt.Errorf("truncated safetensors file")
	}
	var size uint64
	if err := binary.Read(r, binary.LittleEndian, &size); err != nil {
		return nil, err
	}
	if size < 2 || size > maxJSONBytes || size > uint64(fileSize-8) {
		return nil, fmt.Errorf("invalid/oversize safetensors header length %d", size)
	}
	b := make([]byte, int(size))
	if _, err := io.ReadFull(r, b); err != nil {
		return nil, err
	}
	if b[0] != '{' {
		return nil, fmt.Errorf("safetensors header must start with an object")
	}
	root, err := object(b)
	if err != nil {
		return nil, err
	}
	tensors := make(map[string]Tensor, len(root))
	var spans [][2]int64
	for name, raw := range root {
		// MLX's serializer emits null for the optional empty metadata map.
		if name == "__metadata__" && bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
			continue
		}
		fields, err := object(raw)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		if name == "__metadata__" {
			for _, v := range fields {
				var s string
				if bytes.Equal(bytes.TrimSpace(v), []byte("null")) || json.Unmarshal(v, &s) != nil {
					return nil, fmt.Errorf("metadata values must be strings")
				}
			}
			continue
		}
		if name == "" || strings.ContainsRune(name, 0) {
			return nil, fmt.Errorf("invalid tensor name %q", name)
		}
		var t Tensor
		shape, shapeErr := integerArray(fields["shape"])
		offsets, offsetErr := integerArray(fields["data_offsets"])
		if json.Unmarshal(fields["dtype"], &t.DType) != nil || shapeErr != nil || offsetErr != nil || len(offsets) != 2 {
			return nil, fmt.Errorf("%s: malformed tensor metadata", name)
		}
		width := map[string]int64{"BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2, "F16": 2, "BF16": 2, "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8, "F8_E4M3": 1, "F8_E8M0": 1}[t.DType]
		if width == 0 {
			return nil, fmt.Errorf("%w %q for %s", ErrUnsupportedDType, t.DType, name)
		}
		elements := int64(1)
		t.Shape = make([]int, len(shape))
		for i, d := range shape {
			if d < 0 || d > math.MaxInt32 {
				return nil, fmt.Errorf("%s: invalid dimension", name)
			}
			t.Shape[i] = int(d)
			if d == 0 {
				elements = 0
			}
		}
		if elements != 0 {
			for _, d := range t.Shape {
				if int64(d) > math.MaxInt64/width/elements {
					return nil, fmt.Errorf("%s: tensor byte count overflow", name)
				}
				elements *= int64(d)
			}
		}
		t.Bytes = elements * width
		if offsets[0] < 0 || offsets[1] < offsets[0] || offsets[1] > fileSize-8-int64(size) || offsets[1]-offsets[0] != t.Bytes {
			return nil, fmt.Errorf("%s: invalid offsets/byte count", name)
		}
		spans = append(spans, [2]int64{offsets[0], offsets[1]})
		tensors[name] = t
	}
	sort.Slice(spans, func(i, j int) bool {
		if spans[i][0] == spans[j][0] {
			return spans[i][1] < spans[j][1]
		}
		return spans[i][0] < spans[j][0]
	})
	var end int64
	for _, span := range spans {
		if span[0] != end {
			return nil, fmt.Errorf("overlapping tensors or gaps in data")
		}
		end = span[1]
	}
	if end != fileSize-8-int64(size) {
		return nil, fmt.Errorf("unindexed trailing tensor data")
	}
	return tensors, nil
}

func integerArray(raw json.RawMessage) ([]int64, error) {
	// Pointers distinguish JSON null from a legitimate zero dimension/offset.
	var values []*int64
	if err := json.Unmarshal(raw, &values); err != nil || values == nil {
		return nil, fmt.Errorf("expected integer array")
	}
	out := make([]int64, len(values))
	for i, v := range values {
		if v == nil {
			return nil, fmt.Errorf("null integer")
		}
		out[i] = *v
	}
	return out, nil
}

func (m *Manifest) unchanged(file string) error {
	path, err := m.layout.localFile(file)
	if err != nil {
		return err
	}
	info, err := os.Stat(path)
	if err != nil {
		return err
	}
	old := m.stats[file]
	if path != m.paths[file] || !os.SameFile(old, info) || info.Size() != old.Size() || !info.ModTime().Equal(old.ModTime()) {
		return fmt.Errorf("checkpoint: %s changed after inspection", file)
	}
	return nil
}
