package checkpoint

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestMetadataOnlyFP8(t *testing.T) {
	for _, dtype := range []string{"F8_E4M3", "F8_E8M0"} {
		t.Run(dtype, func(t *testing.T) {
			header := []byte(fmt.Sprintf(`{"a":{"dtype":%q,"shape":[32],"data_offsets":[0,32]}}`, dtype))
			prefix := make([]byte, 8)
			binary.LittleEndian.PutUint64(prefix, uint64(len(header)))
			prefix = append(prefix, header...)
			// The reader has no payload to read, but the known file size includes it.
			r := bytes.NewReader(append(append([]byte{}, prefix...), []byte("unread sentinel")...))
			m, err := ReadMetadata(r, int64(len(prefix)+32))
			if err != nil || m["a"].DType != dtype || m["a"].Bytes != 32 {
				t.Fatal(m, err)
			}
			if r.Len() != len("unread sentinel") {
				t.Fatal("read beyond metadata")
			}
			if _, err = ReadMetadata(bytes.NewReader(prefix), int64(len(prefix)+31)); err == nil {
				t.Fatal("accepted truncated payload size")
			}
			dir := t.TempDir()
			if err = os.WriteFile(filepath.Join(dir, singleName), append(prefix, make([]byte, 32)...), 0600); err != nil {
				t.Fatal(err)
			}
			if _, err = Inspect(dir); !errors.Is(err, ErrUnsupportedDType) {
				t.Fatalf("native guard changed: %v", err)
			}
		})
	}
}

func FuzzReadMetadata(f *testing.F) {
	f.Add([]byte("{}"), int64(0))
	f.Add([]byte(`{"a":{"dtype":"F8_E4M3","shape":[1],"data_offsets":[0,1]}}`), int64(1))
	f.Fuzz(func(t *testing.T, h []byte, payload int64) {
		if len(h) > 4096 || payload < 0 || payload > 1<<30 {
			t.Skip()
		}
		p := make([]byte, 8)
		binary.LittleEndian.PutUint64(p, uint64(len(h)))
		p = append(p, h...)
		_, _ = ReadMetadata(bytes.NewReader(p), int64(len(p))+payload)
	})
}
