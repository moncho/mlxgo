package checkpoint

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func write(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
}
func safeFile(t *testing.T, path, header string, data []byte) {
	t.Helper()
	header += strings.Repeat(" ", (8-len(header)%8)%8)
	b := make([]byte, 8+len(header)+len(data))
	binary.LittleEndian.PutUint64(b, uint64(len(header)))
	copy(b[8:], header)
	copy(b[8+len(header):], data)
	write(t, path, b)
}
func floats(values ...float32) []byte {
	b := make([]byte, 4*len(values))
	for i, v := range values {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return b
}
func twoShards(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	safeFile(t, filepath.Join(dir, "one.safetensors"), `{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}`, floats(1, 2))
	safeFile(t, filepath.Join(dir, "two.safetensors"), `{"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`, floats(3))
	write(t, filepath.Join(dir, indexName), []byte(`{"metadata":{"total_size":12},"weight_map":{"a":"one.safetensors","b":"two.safetensors"}}`))
	return dir
}

func TestInspectAndHash(t *testing.T) {
	dir := twoShards(t)
	m, err := Inspect(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !m.Sharded() || !slices.Equal(m.Names(), []string{"a", "b"}) {
		t.Fatal("wrong manifest")
	}
	a, ok := m.Tensor("a")
	if !ok || a.File != "one.safetensors" || a.DType != "F32" || a.Bytes != 8 || !slices.Equal(a.Shape, []int{2}) {
		t.Fatalf("%+v", a)
	}
	a.Shape[0] = 99
	again, _ := m.Tensor("a")
	if again.Shape[0] != 2 {
		t.Fatal("mutable manifest")
	}
	before, err := Hash(dir)
	if err != nil {
		t.Fatal(err)
	}
	const golden = "8eea0e135e5e0d42b38bcc60fab78fdd63f4ed08bee458aedc408ef2a5475a73"
	if before != golden {
		t.Fatalf("versioned checkpoint identity changed: %s", before)
	}
	write(t, filepath.Join(dir, indexName), []byte("{\n \"weight_map\":{\"b\":\"two.safetensors\",\"a\":\"one.safetensors\"},\"metadata\":{\"total_size\":12}}\n"))
	after, err := Hash(dir)
	if err != nil || before != after {
		t.Fatal("JSON formatting changed identity", err)
	}
	files, err := SourceFiles(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 3 || filepath.Base(files[0]) != indexName {
		t.Fatal(files)
	}
	safeFile(t, filepath.Join(dir, "two.safetensors"), `{"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`, floats(4))
	after, err = Hash(dir)
	if err != nil || before == after {
		t.Fatal("shard bytes did not change identity", err)
	}
	if err = m.unchanged("two.safetensors"); err == nil {
		t.Fatal("missed modified shard")
	}
	if err = os.Rename(filepath.Join(dir, "two.safetensors"), filepath.Join(dir, "renamed.safetensors")); err != nil {
		t.Fatal(err)
	}
	write(t, filepath.Join(dir, indexName), []byte(`{"weight_map":{"a":"one.safetensors","b":"renamed.safetensors"}}`))
	renamed, err := Hash(dir)
	if err != nil || renamed == after {
		t.Fatal("shard rename did not change identity", err)
	}

	single := t.TempDir()
	path := filepath.Join(single, singleName)
	safeFile(t, path, `{"scalar":{"dtype":"BF16","shape":[],"data_offsets":[0,2]},"empty":{"dtype":"F32","shape":[0,5],"data_offsets":[2,2]},"__metadata__":{"format":"pt"}}`, []byte{0, 0})
	m, err = Inspect(single)
	if err != nil {
		t.Fatal(err)
	}
	if m.Sharded() {
		t.Fatal("single marked sharded")
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	want := sha256.Sum256(b)
	hash, err := Hash(single)
	if err != nil || hash != hex.EncodeToString(want[:]) {
		t.Fatal("single-file hash compatibility", hash, err)
	}
}

func TestNullOptionalMetadata(t *testing.T) {
	dir := t.TempDir()
	safeFile(t, filepath.Join(dir, singleName), `{"__metadata__":null,"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`, floats(1))
	if _, err := Inspect(dir); err != nil {
		t.Fatal("rejected MLX empty metadata", err)
	}
}

func TestInvalidIndexes(t *testing.T) {
	for name, index := range map[string]string{
		"empty": "{}", "null": "null", "array": "[]", "trailing": `{"weight_map":{"a":"one.safetensors"}} {}`,
		"duplicate root":   `{"weight_map":{},"weight_map":{"a":"one.safetensors"}}`,
		"duplicate tensor": `{"weight_map":{"a":"one.safetensors","\u0061":"two.safetensors"}}`,
		"empty map":        `{"weight_map":{}}`, "missing shard": `{"weight_map":{"a":"absent.safetensors"}}`,
		"wrong routing":    `{"weight_map":{"a":"two.safetensors","b":"one.safetensors"}}`,
		"missing tensor":   `{"weight_map":{"a":"one.safetensors","missing":"one.safetensors"}}`,
		"unindexed tensor": `{"weight_map":{"b":"one.safetensors"}}`,
		"metadata null":    `{"metadata":null,"weight_map":{"a":"one.safetensors"}}`,
		"size mismatch":    `{"metadata":{"total_size":8},"weight_map":{"a":"one.safetensors","b":"two.safetensors"}}`,
		"negative size":    `{"metadata":{"total_size":-1},"weight_map":{"a":"one.safetensors"}}`,
		"fractional size":  `{"metadata":{"total_size":8.0},"weight_map":{"a":"one.safetensors"}}`,
		"null size":        `{"metadata":{"total_size":null},"weight_map":{"a":"one.safetensors"}}`,
	} {
		t.Run(name, func(t *testing.T) {
			dir := twoShards(t)
			write(t, filepath.Join(dir, indexName), []byte(index))
			if _, err := Inspect(dir); err == nil {
				t.Fatal("accepted invalid index")
			}
		})
	}
	for _, file := range []string{"../one.safetensors", "/tmp/one.safetensors", "nested/one.safetensors", "..\\one.safetensors", "https://host/one.safetensors", "C:one.safetensors", "one.safetensors\x00", "one.bin", "", "."} {
		t.Run(file, func(t *testing.T) {
			dir := twoShards(t)
			b, _ := json.Marshal(map[string]any{"weight_map": map[string]string{"a": file}})
			write(t, filepath.Join(dir, indexName), b)
			if _, err := Inspect(dir); err == nil {
				t.Fatal("accepted unsafe shard name")
			}
		})
	}
	t.Run("ambiguous", func(t *testing.T) {
		dir := twoShards(t)
		write(t, filepath.Join(dir, singleName), nil)
		if _, err := Inspect(dir); err == nil {
			t.Fatal("accepted ambiguous layout")
		}
	})
	t.Run("duplicate across shards", func(t *testing.T) {
		dir := twoShards(t)
		safeFile(t, filepath.Join(dir, "two.safetensors"), `{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`, floats(3))
		if _, err := Inspect(dir); err == nil {
			t.Fatal("accepted duplicate")
		}
	})
	t.Run("oversize index", func(t *testing.T) {
		dir := twoShards(t)
		write(t, filepath.Join(dir, indexName), []byte(strings.Repeat(" ", maxJSONBytes+1)))
		if _, err := Inspect(dir); err == nil {
			t.Fatal("accepted oversize")
		}
	})
}

func TestPathBoundaries(t *testing.T) {
	for _, name := range []string{indexName, "one.safetensors"} {
		t.Run(name, func(t *testing.T) {
			dir := twoShards(t)
			external := filepath.Join(t.TempDir(), "external")
			b, err := os.ReadFile(filepath.Join(dir, name))
			if err != nil {
				t.Fatal(err)
			}
			write(t, external, b)
			if err = os.Remove(filepath.Join(dir, name)); err != nil {
				t.Fatal(err)
			}
			if err = os.Symlink(external, filepath.Join(dir, name)); err != nil {
				t.Fatal(err)
			}
			if _, err = Inspect(dir); err == nil {
				t.Fatal("followed external symlink")
			}
		})
	}
	dir := twoShards(t)
	if err := os.Rename(filepath.Join(dir, "one.safetensors"), filepath.Join(dir, "local.safetensors")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("local.safetensors", filepath.Join(dir, "one.safetensors")); err != nil {
		t.Fatal(err)
	}
	if _, err := Inspect(dir); err != nil {
		t.Fatal("rejected local symlink", err)
	}
	if err := os.Remove(filepath.Join(dir, "two.safetensors")); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(dir, "two.safetensors"), 0700); err != nil {
		t.Fatal(err)
	}
	if _, err := Inspect(dir); err == nil {
		t.Fatal("accepted directory shard")
	}
}

func TestInvalidHeaders(t *testing.T) {
	for name, header := range map[string]string{
		"duplicate":       `{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`,
		"duplicate dtype": `{"a":{"dtype":"F16","dtype":"F32","shape":[1],"data_offsets":[0,4]}}`,
		"negative shape":  `{"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}`,
		"null shape":      `{"a":{"dtype":"F32","shape":null,"data_offsets":[0,4]}}`,
		"null dimension":  `{"a":{"dtype":"F32","shape":[null],"data_offsets":[0,0]},"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`,
		"null offset":     `{"a":{"dtype":"F32","shape":[1],"data_offsets":[null,4]}}`,
		"overflow":        `{"a":{"dtype":"F32","shape":[2147483647,2147483647,2147483647],"data_offsets":[0,4]}}`,
		"short data":      `{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}`,
		"negative offset": `{"a":{"dtype":"F32","shape":[1],"data_offsets":[-1,3]}}`,
		"bad length":      `{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}`,
		"overlap":         `{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`,
		"hole":            `{"a":{"dtype":"U8","shape":[3],"data_offsets":[1,4]}}`,
		"trailing data":   `{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}`,
		"metadata type":   `{"__metadata__":{"format":12}}`,
		"null metadata":   `{"__metadata__":{"format":null}}`,
		"invalid name":    `{"a\u0000":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`,
	} {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			safeFile(t, filepath.Join(dir, singleName), header, floats(1))
			if _, err := Inspect(dir); err == nil {
				t.Fatal("accepted invalid header")
			}
		})
	}
	for _, dtype := range []string{"F8_E4M3", "F4", "UNKNOWN"} {
		dir := t.TempDir()
		safeFile(t, filepath.Join(dir, singleName), `{"a":{"dtype":"`+dtype+`","shape":[4],"data_offsets":[0,4]}}`, floats(1))
		if _, err := Inspect(dir); !errors.Is(err, ErrUnsupportedDType) {
			t.Fatal("missing dtype error", err)
		}
	}
	for _, size := range []uint64{0, 1, maxJSONBytes + 1, math.MaxUint64} {
		dir := t.TempDir()
		b := make([]byte, 8)
		binary.LittleEndian.PutUint64(b, size)
		write(t, filepath.Join(dir, singleName), b)
		if _, err := Inspect(dir); err == nil {
			t.Fatal("accepted invalid header size")
		}
	}
}

func FuzzObject(f *testing.F) {
	for _, seed := range []string{`{}`, `{"a":1,"a":2}`, `{"a":{"b":1}}`, `null`, `[]`, `{"a":1} {}`} {
		f.Add([]byte(seed))
	}
	f.Fuzz(func(t *testing.T, b []byte) {
		if len(b) > 1<<16 {
			return
		}
		_, _ = object(b)
	})
}
