package deepseekaudit

import (
	"bytes"
	"encoding/json"
	"os"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/moncho/mlxgo/checkpoint"
	"github.com/moncho/mlxgo/deepseek"
)

func TestPinnedReleasedMetadata(t *testing.T) {
	f, err := os.Open("../../deepseek/testdata/released-checkpoint.metadata.json.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	s, err := ReadSnapshot(f)
	if err != nil {
		t.Fatal(err)
	}
	r, err := Audit(s)
	if err != nil {
		t.Fatal(err)
	}
	if !r.MetadataValid || r.ReadyToLoad || r.PayloadBytesRead != 0 || r.TextWeights != 46966 || r.PayloadBytes != 510286023000 || len(r.Shards) != 48 {
		t.Fatalf("unexpected audit totals: %+v", r)
	}
	b, err := os.ReadFile("../../deepseek/testdata/released-checkpoint.audit.json")
	if err != nil {
		t.Fatal(err)
	}
	var want Report
	if err = json.Unmarshal(b, &want); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(r, want) {
		t.Fatal("report differs from pinned checkpoint audit")
	}
	var total Count
	for _, c := range r.ByAction {
		total.Tensors += c.Tensors
		total.Bytes += c.Bytes
	}
	if total.Tensors != 96085 || total.Bytes != r.PayloadBytes {
		t.Fatal("classification did not cover all storage", total)
	}
	first := r.Shards[0].File
	original := s.Headers[first]
	delete(s.Headers, first)
	if _, err = Audit(s); err == nil {
		t.Fatal("accepted missing shard header")
	}
	s.Headers[first] = original
	bad := original
	bad.Prefix = append(slices.Clone(original.Prefix), 0)
	s.Headers[first] = bad
	if _, err = Audit(s); err == nil {
		t.Fatal("accepted payload byte in metadata snapshot")
	}
	bad.Prefix = bytes.Replace(original.Prefix, []byte("vision.patch_embed.proj.weight"), []byte("bogusn.patch_embed.proj.weight"), 1)
	s.Headers[first] = bad
	if _, err = Audit(s); err == nil {
		t.Fatal("accepted header/index mismatch")
	}
	s.Headers[first] = original
	// Config/source tampering must fail before the untrusted shape contract runs.
	s.Config = append(s.Config, ' ')
	if _, err = Audit(s); err == nil {
		t.Fatal("accepted changed config")
	}
}

func TestContractMatchesRuntimeShapes(t *testing.T) {
	for _, file := range []string{"model.json", "model_engram.json"} {
		t.Run(file, func(t *testing.T) {
			b, err := os.ReadFile("../../deepseek/testdata/" + file)
			if err != nil {
				t.Fatal(err)
			}
			var fixture struct {
				Config deepseek.Config `json:"config"`
			}
			if err = json.Unmarshal(b, &fixture); err != nil {
				t.Fatal(err)
			}
			c := fixture.Config
			x := textConfig{Vocab: c.VocabSize, Dim: c.Dim, Inter: c.InterDim, Layers: c.Layers, Heads: c.Heads, HeadDim: c.HeadDim, QRank: c.QRank, ORank: c.ORank, Groups: c.Groups, Experts: c.Experts, Hyper: c.Hyper.Streams, Ratios: c.Ratios, KVSources: c.KVSources, IndexSources: c.IndexSources, IndexDim: c.IndexDim, IndexHeads: c.IndexHeads}
			if c.Engram != nil {
				e := c.Engram
				x.EngramLayers = e.Layers
				x.EngramRows = e.Rows
				x.EngramDim = e.HeadDim
				x.EngramHeads = e.Heads
				x.EngramNGram = e.MaxNGram
			}
			want, err := c.ParameterShapes()
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(shapes(x), want) {
				t.Fatal("audit shape contract drifted from runtime")
			}
		})
	}
}

func TestWeightRecipes(t *testing.T) {
	for _, tc := range []struct {
		name, dtype, action string
		stored, want, scale []int
	}{
		{"layers.0.attn.wq_a.weight", "F8_E4M3", "decode_fp8_block32", []int{64, 128}, []int{64, 128}, []int{2, 4}},
		{"layers.0.attn.wo_a.weight", "F8_E4M3", "decode_fp8_block32_then_bf16", []int{64, 128}, []int{64, 128}, []int{2, 4}},
		{"layers.1.engram.embed.weight", "F8_E4M3", "decode_fp8_row32", []int{103, 256}, []int{103, 256}, []int{103, 8}},
		{"layers.0.ffn.experts.0.w1.weight", "I8", "unpack_e2m1_low_high_row32", []int{64, 64}, []int{64, 128}, []int{64, 4}},
		{"layers.0.ffn.experts.0.w2.weight", "I8", "unpack_e2m1_low_high_row32", []int{128, 32}, []int{128, 64}, []int{128, 2}},
		{"norm.weight", "BF16", "cast_bf16_to_float32", []int{64}, []int{64}, nil},
		{"layers.0.hc_attn_fn", "F32", "direct_float32", []int{24, 20480}, []int{24, 20480}, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, scale, err := weightRecipe(tc.name, checkpoint.Tensor{DType: tc.dtype, Shape: tc.stored}, tc.want)
			if err != nil || a != tc.action || !slices.Equal(scale, tc.scale) {
				t.Fatal(a, scale, err)
			}
			bad := slices.Clone(tc.stored)
			bad[0]++
			if _, _, err = weightRecipe(tc.name, checkpoint.Tensor{DType: tc.dtype, Shape: bad}, tc.want); err == nil {
				t.Fatal("accepted shape mismatch")
			}
		})
	}
	for _, name := range []string{"norm.weight", "layers.0.ffn.shared_experts.w1.weight"} {
		if _, _, err := weightRecipe(name, checkpoint.Tensor{DType: "I8", Shape: []int{64, 64}}, []int{64, 128}); err == nil {
			t.Fatal("treated arbitrary I8 as packed FP4", name)
		}
	}
}

func TestMappingErrorsAndNormalization(t *testing.T) {
	for from, want := range map[string]string{
		"model.layers.1.self_attn.wq_a.weight_scale_inv":  "layers.1.attn.wq_a.scale",
		"model.layers.1.mlp.gate.e_score_correction_bias": "layers.1.ffn.gate.bias",
		"vision.blocks.0.mlp.w1.weight":                   "vision.blocks.0.mlp.w1.weight",
		"model.mtp.layers.0.mlp.experts.0.w1.weight":      "mtp.layers.0.ffn.experts.0.w1.weight",
	} {
		if got := normalize(from); got != want {
			t.Fatal(from, got)
		}
	}
	weight := "layers.0.attn.wq_a.weight"
	scale := "layers.0.attn.wq_a.scale"
	for _, mode := range []string{"good", "missing weight", "missing scale", "bad scale", "orphan", "bad shape", "unknown dtype", "bad vision bias"} {
		t.Run(mode, func(t *testing.T) {
			all := map[string]checkpoint.Tensor{weight: {DType: "F8_E4M3", Shape: []int{32, 32}, Bytes: 1024}, scale: {DType: "F8_E8M0", Shape: []int{1, 1}, Bytes: 1}}
			switch mode {
			case "missing weight":
				delete(all, weight)
			case "missing scale":
				delete(all, scale)
			case "bad scale":
				all[scale] = checkpoint.Tensor{DType: "F32", Shape: []int{1, 1}, Bytes: 4}
			case "orphan":
				all["unknown.scale"] = all[scale]
			case "bad shape":
				all[weight] = checkpoint.Tensor{DType: "F8_E4M3", Shape: []int{32, 64}, Bytes: 2048}
			case "unknown dtype":
				all[weight] = checkpoint.Tensor{DType: "U8", Shape: []int{32, 32}, Bytes: 1024}
			case "bad vision bias":
				all["layers.999.ffn.gate.bias_vl"] = checkpoint.Tensor{DType: "F32", Shape: []int{384}}
			}
			r := mapTensors(Report{ByStorage: map[string]Count{}, ByAction: map[string]Count{}}, all, map[string]string{}, map[string][]int{weight: {32, 32}})
			if r.MetadataValid != (mode == "good") || r.ReadyToLoad {
				t.Fatal(r.Issues)
			}
		})
	}
}

func TestSnapshotRoundTrip(t *testing.T) {
	s := Snapshot{Revision: "test", Config: []byte("{\n  }\n"), Index: []byte("{}"), Headers: map[string]Header{"x": {Size: 42, Prefix: []byte{0, 1, 2}}}}
	var b bytes.Buffer
	if err := WriteSnapshot(&b, s); err != nil {
		t.Fatal(err)
	}
	got, err := ReadSnapshot(bytes.NewReader(b.Bytes()))
	if err != nil || !reflect.DeepEqual(s, got) {
		t.Fatal(got, err)
	}
	if _, err = ReadSnapshot(strings.NewReader("not gzip")); err == nil {
		t.Fatal("accepted garbage")
	}
}
