package deepseek

import (
	"compress/gzip"
	"encoding/json"
	"io"
	"os"
	"testing"
)

func readQuantizedModelFixture(t *testing.T) modelFixture {
	t.Helper()
	f, err := os.Open("testdata/model_quantized.json.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	z, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	defer z.Close()
	d := json.NewDecoder(io.LimitReader(z, 16<<20))
	var r modelFixture
	if err := d.Decode(&r); err != nil {
		t.Fatal(err)
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		t.Fatal("trailing quantized fixture", err)
	}
	if r.CacheMode != "fp8_fp4_float32_v1" || r.Torch != "2.14.0" || r.SHA256["model.py"] != "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65" || r.SHA256["kernel.py"] != "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455" {
		t.Fatal("quantized fixture provenance mismatch")
	}
	return r
}

func TestQuantizedModelFixtureContract(t *testing.T) {
	f := readQuantizedModelFixture(t)
	checkModelFixtureContract(t, f)
	if f.Config.HeadDim != 32 || f.Config.IndexDim != 32 || len(f.Second) != len(f.Parameters) {
		t.Fatal("unexpected quantized fixture dimensions or missing second model")
	}
}
