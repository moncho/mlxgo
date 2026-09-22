package deepseekaudit

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/moncho/mlxgo/checkpoint"
)

type Count struct {
	Tensors int   `json:"tensors"`
	Bytes   int64 `json:"bytes"`
}

type Family struct {
	Pattern       string `json:"pattern"`
	DType         string `json:"dtype"`
	Shape         []int  `json:"storage_shape"`
	LogicalShape  []int  `json:"logical_shape,omitempty"`
	Action        string `json:"required_action"`
	ExampleSource string `json:"example_source"`
	ExampleTarget string `json:"example_target,omitempty"`
	Count
}

type Shard struct {
	File         string `json:"file"`
	FileBytes    int64  `json:"file_bytes"`
	HeaderBytes  int    `json:"header_bytes"`
	HeaderSHA256 string `json:"header_sha256"`
	Tensors      int    `json:"tensors"`
}

type Report struct {
	Repository       string            `json:"repository"`
	Revision         string            `json:"revision"`
	Sources          map[string]string `json:"source_sha256"`
	MetadataValid    bool              `json:"metadata_valid"`
	ReadyToLoad      bool              `json:"ready_to_load"`
	PayloadBytesRead int               `json:"payload_bytes_read"`
	HeaderBytes      int64             `json:"header_bytes"`
	PayloadBytes     int64             `json:"payload_bytes"`
	TextWeights      int               `json:"text_weights"`
	TextParameters   int64             `json:"text_parameters"`
	Float32TextBytes int64             `json:"float32_text_bytes"`
	ByStorage        map[string]Count  `json:"by_storage"`
	ByAction         map[string]Count  `json:"by_action"`
	Shards           []Shard           `json:"shards"`
	Families         []Family          `json:"families"`
	Issues           []string          `json:"issues"`
	Limitations      []string          `json:"limitations"`
}

// Audit validates the entire indexed inventory, then maps all backbone weights
// and their scales. Successful mapping is not numerical or runtime parity.
func Audit(s Snapshot) (Report, error) {
	r := Report{Repository: Repository, Revision: Revision, ByStorage: map[string]Count{}, ByAction: map[string]Count{}, Issues: []string{}, Sources: map[string]string{"config.json": ConfigSHA256, "model.safetensors.index.json": IndexSHA256, "inference/convert.py": ConvertSHA256}}
	if err := verifySources(s); err != nil {
		return r, err
	}
	i, err := checkpoint.ParseIndex(s.Index)
	if err != nil {
		return r, err
	}
	files := map[string]bool{}
	for _, file := range i.WeightMap {
		files[file] = true
	}
	if len(files) != len(s.Headers) {
		return r, fmt.Errorf("audit: snapshot shard inventory differs from index")
	}
	all := map[string]checkpoint.Tensor{}
	source := map[string]string{}
	names := make([]string, 0, len(files))
	for file := range files {
		names = append(names, file)
	}
	sort.Strings(names)
	for _, file := range names {
		h, ok := s.Headers[file]
		if !ok || len(h.Prefix) < 8 || binary.LittleEndian.Uint64(h.Prefix[:8]) != uint64(len(h.Prefix)-8) {
			return r, fmt.Errorf("audit: missing/invalid header prefix for %s", file)
		}
		metadata, e := checkpoint.ReadMetadata(bytes.NewReader(h.Prefix), h.Size)
		if e != nil {
			return r, fmt.Errorf("audit: %s: %w", file, e)
		}
		r.HeaderBytes += int64(len(h.Prefix))
		r.Shards = append(r.Shards, Shard{file, h.Size, len(h.Prefix), digest(h.Prefix), len(metadata)})
		for name, t := range metadata {
			if i.WeightMap[name] != file {
				return r, fmt.Errorf("audit: %s disagrees with index", name)
			}
			n := normalize(name)
			if _, exists := all[n]; exists {
				return r, fmt.Errorf("audit: normalized tensor collision for %s", n)
			}
			t.File = file
			all[n], source[n] = t, name
			if t.Bytes > math.MaxInt64-r.PayloadBytes {
				return r, fmt.Errorf("audit: payload overflow")
			}
			r.PayloadBytes += t.Bytes
		}
	}
	if len(all) != len(i.WeightMap) {
		return r, fmt.Errorf("audit: indexed tensors missing from headers")
	}
	if i.TotalSize == nil || *i.TotalSize != r.PayloadBytes {
		return r, fmt.Errorf("audit: index total_size does not match payload")
	}
	expected, err := readContract(s.Config)
	if err != nil {
		return r, err
	}
	return mapTensors(r, all, source, expected), nil
}

func mapTensors(r Report, all map[string]checkpoint.Tensor, sources map[string]string, expected map[string][]int) Report {
	families := map[string]*Family{}
	consumed := map[string]bool{}
	add := func(name, target, action string, logical []int) {
		t := all[name]
		consumed[name] = true
		storage := r.ByStorage[t.DType]
		storage.Tensors++
		storage.Bytes += t.Bytes
		r.ByStorage[t.DType] = storage
		classified := r.ByAction[action]
		classified.Tensors++
		classified.Bytes += t.Bytes
		r.ByAction[action] = classified
		parts := strings.Split(name, ".")
		for i, p := range parts {
			if _, e := strconv.Atoi(p); e == nil {
				parts[i] = "*"
			}
		}
		pattern := strings.Join(parts, ".")
		key := fmt.Sprint(pattern, "/", t.DType, "/", t.Shape, "/", logical, "/", action)
		family := families[key]
		if family == nil {
			family = &Family{Pattern: pattern, DType: t.DType, Shape: t.Shape, LogicalShape: logical, Action: action, ExampleSource: sources[name], ExampleTarget: target}
			families[key] = family
		}
		family.Tensors++
		family.Bytes += t.Bytes
	}
	names := make([]string, 0, len(expected))
	for name := range expected {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		want := expected[name]
		t, ok := all[name]
		if !ok {
			r.Issues = append(r.Issues, "missing text weight: "+name)
			continue
		}
		action, scaleShape, e := weightRecipe(name, t, want)
		if e != nil {
			r.Issues = append(r.Issues, e.Error())
			add(name, name, "invalid", want)
			continue
		}
		add(name, name, action, want)
		r.TextWeights++
		n := int64(1)
		for _, d := range want {
			n *= int64(d)
		}
		r.TextParameters += n
		if scaleShape != nil {
			scaleName := strings.TrimSuffix(name, "weight") + "scale"
			scale, ok := all[scaleName]
			if !ok {
				r.Issues = append(r.Issues, "missing scale: "+scaleName)
				continue
			}
			if scale.DType != "F8_E8M0" || !slices.Equal(scale.Shape, scaleShape) {
				r.Issues = append(r.Issues, fmt.Sprintf("%s: expected F8_E8M0 %v, got %s %v", scaleName, scaleShape, scale.DType, scale.Shape))
			}
			add(scaleName, name, "decode_ue8m0_scale", want)
		}
	}
	names = names[:0]
	for name := range all {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		if consumed[name] {
			continue
		}
		action := "unmapped"
		switch {
		case strings.HasPrefix(name, "mtp."):
			action = "excluded_dspark"
		case strings.HasPrefix(name, "vision."), strings.HasPrefix(name, "aligner."), name == "image_start", name == "image_end", name == "image_newline":
			action = "excluded_vision"
		case strings.HasSuffix(name, ".ffn.gate.bias_vl"):
			base := strings.TrimSuffix(name, "_vl")
			if shape, ok := expected[base]; ok && slices.Equal(shape, all[name].Shape) && all[name].DType == "F32" {
				action = "excluded_vision"
			}
		}
		if action == "unmapped" {
			r.Issues = append(r.Issues, "unmapped tensor: "+name)
		}
		add(name, "", action, nil)
	}
	keys := make([]string, 0, len(families))
	for k := range families {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		r.Families = append(r.Families, *families[k])
	}
	sort.Strings(r.Issues)
	r.Float32TextBytes = 4 * r.TextParameters
	r.MetadataValid = len(r.Issues) == 0
	r.Limitations = []string{
		"Metadata and scale shapes only: tensor payloads, scale values and numerical parity were not checked.",
		"FP8/UE8M0 decoding, packed E2M1 expert decoding and BF16 rounding are not implemented by this audit.",
		"Vision and DSpark are explicitly excluded from the text-backbone mapping.",
		"Released tokenizer/Engram hash metadata and cache quantization remain unsupported.",
		"Runtime context and memory limits are unchanged; this report does not create a runnable model.",
		"Header SHA256 values identify observed metadata, not the full shard payload or a verified LFS object.",
	}
	return r
}

func weightRecipe(name string, t checkpoint.Tensor, want []int) (string, []int, error) {
	storage := slices.Clone(want)
	var scale []int
	action := ""
	switch t.DType {
	case "F32":
		action = "direct_float32"
	case "BF16":
		action = "cast_bf16_to_float32"
	case "F8_E4M3":
		if len(want) != 2 || want[0]%32 != 0 || want[1]%32 != 0 {
			if !strings.HasSuffix(name, ".engram.embed.weight") || len(want) != 2 || want[1]%32 != 0 {
				return "", nil, fmt.Errorf("%s: invalid FP8 block dimensions", name)
			}
		}
		action = "decode_fp8_block32"
		scale = []int{(want[0] + 31) / 32, (want[1] + 31) / 32}
		if strings.HasSuffix(name, ".engram.embed.weight") {
			action = "decode_fp8_row32"
			scale[0] = want[0]
		}
		if strings.HasSuffix(name, ".attn.wo_a.weight") {
			action = "decode_fp8_block32_then_bf16"
		}
	case "I8":
		if !strings.Contains(name, ".ffn.experts.") || len(want) != 2 || want[1]%32 != 0 {
			return "", nil, fmt.Errorf("%s: I8 is not a recognized packed expert", name)
		}
		action = "unpack_e2m1_low_high_row32"
		storage[1] /= 2
		scale = []int{want[0], want[1] / 32}
	default:
		return "", nil, fmt.Errorf("%s: unsupported weight dtype %s", name, t.DType)
	}
	if !slices.Equal(storage, t.Shape) {
		return "", nil, fmt.Errorf("%s: expected storage shape %v, got %v", name, storage, t.Shape)
	}
	return action, scale, nil
}
