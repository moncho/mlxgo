package checkpoint

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// FileHash returns the SHA-256 of the exact file bytes.
func FileHash(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return "", err
	}
	if !info.Mode().IsRegular() {
		return "", fmt.Errorf("checkpoint: expected regular file")
	}
	h := sha256.New()
	if _, err = io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// Hash binds adapters to a complete local checkpoint. Single-file bundles retain
// the legacy raw-file SHA-256. Sharded hashes use a versioned canonical encoding
// of sorted name-to-file routing and each shard's name and raw SHA-256. Index
// whitespace/key order does not affect identity; reshuffling shards does.
func Hash(dir string) (string, error) {
	m, err := Inspect(dir)
	if err != nil {
		return "", err
	}
	if !m.Sharded() {
		return FileHash(m.paths[singleName])
	}
	h := sha256.New()
	_, _ = io.WriteString(h, "mlxgo-sharded-checkpoint-v1\n")
	encoder := json.NewEncoder(h)
	for _, name := range m.Names() {
		if err = encoder.Encode([3]string{"tensor", name, m.tensors[name].File}); err != nil {
			return "", err
		}
	}
	for _, file := range m.layout.files {
		if err = m.unchanged(file); err != nil {
			return "", err
		}
		sum, e := FileHash(m.paths[file])
		if e != nil {
			return "", e
		}
		if err = m.unchanged(file); err != nil {
			return "", err
		}
		if err = encoder.Encode([3]string{"shard", file, sum}); err != nil {
			return "", err
		}
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
