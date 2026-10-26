//go:build mlx && mlxruntime

package inference

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/deepseek"
	"github.com/moncho/mlxgo/deepseek/quant"
	"github.com/moncho/mlxgo/lm"
)

const deviceDecodeBenchmarkSeed uint32 = 197
const deviceDecodeBenchmarkTokens = 32

type deviceDecodeBenchmarkModel struct {
	Name         string               `json:"name"`
	Config       deepseek.Config      `json:"config"`
	Storage      string               `json:"expert_storage"`
	Parameters   int64                `json:"logical_parameters"`
	PayloadBytes int64                `json:"safetensors_file_bytes"`
	SHA256       map[string]string    `json:"bundle_sha256"`
	Weights      deepseek.LoadOptions `json:"load_options"`
}

type deviceDecodeBenchmarkSample struct {
	Case                  string    `json:"case"`
	Backend               string    `json:"backend"`
	Round                 int       `json:"round"`
	Order                 int       `json:"order"`
	Prompt                []int32   `json:"prompt"`
	Tokens                []int32   `json:"tokens"`
	EmissionSeconds       []float64 `json:"emission_seconds"`
	LoadSeconds           float64   `json:"load_seconds"`
	WarmupSeconds         float64   `json:"warmup_seconds"`
	TTFTSeconds           float64   `json:"ttft_seconds"`
	DecodeSeconds         float64   `json:"decode_seconds"`
	DecodeTokensPerSec    float64   `json:"decode_tokens_per_second"`
	TotalSeconds          float64   `json:"total_seconds"`
	PrefillForwardSec     float64   `json:"prefill_forward_seconds"`
	DecodeForwardSec      float64   `json:"decode_forward_seconds"`
	BaselineActiveBytes   uint64    `json:"baseline_active_bytes"`
	SteadyActiveBytes     uint64    `json:"steady_active_bytes"`
	PeakActiveBytes       uint64    `json:"peak_active_bytes"`
	WorkingPeakBytes      uint64    `json:"working_peak_bytes"`
	AfterCloseActiveBytes uint64    `json:"after_close_active_bytes"`
}

type deviceDecodeBenchmarkReport struct {
	Schema       int                           `json:"schema"`
	CreatedUTC   string                        `json:"created_utc"`
	Completed    bool                          `json:"completed"`
	Go           string                        `json:"go"`
	OS           string                        `json:"os"`
	Arch         string                        `json:"arch"`
	Seed         uint32                        `json:"seed"`
	Rounds       int                           `json:"rounds"`
	NewTokens    int                           `json:"new_tokens"`
	Filter       string                        `json:"filter"`
	Method       []string                      `json:"method"`
	SourceSHA256 map[string]string             `json:"source_sha256"`
	Models       []deviceDecodeBenchmarkModel  `json:"models"`
	Samples      []deviceDecodeBenchmarkSample `json:"samples"`
}

// TestDeviceDecodeGenerationBenchmark is deliberately opt-in: it performs full
// generation on untrained synthetic models, rather than being a CI timing test.
// Run native jobs sequentially so allocator counters describe this process's
// one resident model. MLX_DEVICE_DECODE_OUT must name a new file when supplied.
func TestDeviceDecodeGenerationBenchmark(t *testing.T) {
	if os.Getenv("MLX_DEVICE_DECODE_BENCH") != "1" {
		t.Skip("set MLX_DEVICE_DECODE_BENCH=1 to run synthetic end-to-end generation measurements")
	}
	rounds := 3
	if value := os.Getenv("MLX_DEVICE_DECODE_ROUNDS"); value != "" {
		var err error
		rounds, err = strconv.Atoi(value)
		if err != nil || rounds < 1 || rounds > 20 {
			t.Fatal("MLX_DEVICE_DECODE_ROUNDS must be between 1 and 20")
		}
	}
	filter := os.Getenv("MLX_DEVICE_DECODE_FILTER")
	pattern, err := regexp.Compile(filter)
	if err != nil {
		t.Fatal(err)
	}
	if err := mlx.SetDefaultGPU(); err != nil {
		t.Fatal(err)
	}
	defer mlx.SetDefaultCPU()
	report := deviceDecodeBenchmarkReport{
		Schema: 1, CreatedUTC: time.Now().UTC().Format(time.RFC3339), Go: runtime.Version(),
		OS: runtime.GOOS, Arch: runtime.GOARCH, Seed: deviceDecodeBenchmarkSeed,
		Rounds: rounds, NewTokens: deviceDecodeBenchmarkTokens, Filter: filter,
		SourceSHA256: deviceDecodeSourceHashes(t),
		Method: []string{
			"Untrained synthetic complete DeepSeek text models; no pretrained checkpoint or text quality claim.",
			"Prepared bundle loaded through inference.Open; actual Model.generate and lm.Stream greedy token selection, cache updates and token readback.",
			"Sparse baseline: SparseExperts=true. Device backend: DeviceDecode=true; sparse prefill and device dispatch only after the first step.",
			"One model resident at a time; same bundle and prompt per pair. Backend order alternates each round. One complete generation warmup before each measurement.",
			"Load and warmup excluded. Total generation includes fresh session creation and close. TTFT includes first token selection; decode interval spans first through last emitted token.",
			"32 tokens generated without stop IDs. Every warmup and measured token sequence must agree with the paired backend and earlier rounds.",
			"Stream synchronization before allocator samples is outside timing. Native allocation excludes Go buffers, allocator cache, RSS and loading peaks.",
		},
	}
	output := os.Getenv("MLX_DEVICE_DECODE_OUT")
	if output != "" {
		f, err := os.OpenFile(output, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
		if err != nil {
			t.Fatal(err)
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
	}
	save := func(t *testing.T) { deviceDecodeSaveReport(t, output, &report) }
	save(t)
	for _, size := range [][2]int{{128, 128}, {512, 1024}} {
		for _, topK := range []int{2, 8} {
			for _, storage := range []string{"float32", "fp4"} {
				name := fmt.Sprintf("dim%d_inter%d/k%d/%s", size[0], size[1], topK, storage)
				var promptSizes []int
				for _, n := range []int{16, 128} {
					if pattern.MatchString(fmt.Sprintf("%s/prompt%d", name, n)) {
						promptSizes = append(promptSizes, n)
					}
				}
				if len(promptSizes) == 0 {
					continue
				}
				config := deviceDecodeBenchmarkConfig(size[0], size[1], topK)
				dir, model := deviceDecodeBenchmarkBundle(t, name, storage, config)
				report.Models = append(report.Models, model)
				save(t)
				for _, n := range promptSizes {
					caseName := fmt.Sprintf("%s/prompt%d", name, n)
					if !t.Run(caseName, func(t *testing.T) {
						prompt := make([]int32, n)
						for i := range prompt {
							prompt[i] = int32((i*37 + i*i*3 + 11) % config.VocabSize)
						}
						var want []int32
						for round := 0; round < rounds; round++ {
							order := []string{"sparse", "device"}
							if round%2 == 1 {
								slices.Reverse(order)
							}
							for i, backend := range order {
								sample := deviceDecodeMeasure(t, dir, backend, prompt, model.Weights)
								sample.Case, sample.Round, sample.Order = caseName, round+1, i+1
								report.Samples = append(report.Samples, sample)
								save(t)
								if want == nil {
									want = slices.Clone(sample.Tokens)
								} else if !slices.Equal(sample.Tokens, want) {
									t.Fatalf("%s round %d generated tokens differ: got %v, want %v", backend, round+1, sample.Tokens, want)
								}
								t.Logf("round=%d backend=%s ttft=%.3fms decode=%.2ftok/s total=%.3fms peak=%.2fMiB", round+1, backend, sample.TTFTSeconds*1e3, sample.DecodeTokensPerSec, sample.TotalSeconds*1e3, float64(sample.PeakActiveBytes)/(1<<20))
							}
						}
					}) {
						t.FailNow()
					}
				}
			}
		}
	}
	if len(report.Samples) == 0 {
		t.Fatal("benchmark filter selected no cases")
	}
	report.Completed = !t.Failed()
	save(t)
	if output != "" {
		t.Logf("saved %d generation measurements to %s", len(report.Samples), output)
	}
}

func deviceDecodeBenchmarkConfig(dim, inter, topK int) deepseek.Config {
	return deepseek.Config{
		Format: deepseek.Float32Format, VocabSize: 256, Dim: dim, InterDim: inter,
		Layers: 4, Heads: 4, Experts: 16,
		Router: deepseek.RouterConfig{TopK: topK, Score: "sigmoid", Temperature: 1, Scale: 1, Normalize: true},
		Limit:  1, QRank: 32, HeadDim: 32, RopeDim: 16, Groups: 2, ORank: 32,
		Window: 64, MaxSeq: 256, Ratios: []int{0, 4, 4, 0},
		KVSources: []int{1}, IndexSources: []int{1, 2},
		RopeTheta: 10000, CompressTheta: 1000, RopeFactor: 4, BetaFast: 32, BetaSlow: 1,
		IndexHeads: 2, IndexDim: 32, IndexTopK: 16, CandidateSource: -1,
		Hyper: deepseek.HyperConfig{Streams: 3, Iterations: 20, NormEpsilon: 1e-6, Epsilon: 1e-6},
	}
}

func deviceDecodeBenchmarkBundle(t *testing.T, name, storage string, config deepseek.Config) (string, deviceDecodeBenchmarkModel) {
	t.Helper()
	dir := t.TempDir()
	writeConfig(t, dir, config)
	shapes, err := config.ParameterShapes()
	if err != nil {
		t.Fatal(err)
	}
	model := deviceDecodeBenchmarkModel{Name: name, Config: config, Storage: storage, SHA256: map[string]string{}}
	if storage == "fp4" {
		for layer := 0; layer < config.Layers; layer++ {
			for expert := -1; expert < config.Experts; expert++ {
				model.Weights.FP4Experts = append(model.Weights.FP4Experts, deepseek.ExpertID{Layer: layer, Index: expert})
			}
		}
	}
	arrays := make(map[string]mlx.Array, len(shapes))
	defer func() {
		for _, a := range arrays {
			a.Close()
		}
	}()
	var names []string
	for name := range shapes {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		shape := shapes[name]
		count := 1
		for _, d := range shape {
			count *= d
		}
		model.Parameters += int64(count)
		hash := fnv.New32a()
		_, _ = hash.Write([]byte(name))
		state := hash.Sum32() ^ deviceDecodeBenchmarkSeed
		next := func() uint32 { state = state*1664525 + 1013904223; return state }
		var values []float32
		if strings.Contains(name, ".ffn.experts.") || strings.Contains(name, ".ffn.shared_experts.") {
			data, scales := make([]byte, count/2), make([]byte, count/32)
			for i := range data {
				data[i] = byte(next() >> 24)
			}
			for i := range scales {
				scales[i] = byte(120 + next()%3)
			}
			if storage == "fp4" {
				arrays[name], err = mlx.NewUInt8(data, []int{shape[0], shape[1] / 2})
				if err != nil {
					t.Fatal(err)
				}
				arrays[strings.TrimSuffix(name, ".weight")+".scale"], err = mlx.NewUInt8(scales, []int{shape[0], shape[1] / 32})
				if err != nil {
					t.Fatal(err)
				}
				continue
			}
			values = make([]float32, count)
			if err = quant.Decode(values, data, scales, shape[0], shape[1], quant.FP4Row32, quant.Float32); err != nil {
				t.Fatal(err)
			}
		} else {
			values = make([]float32, count)
			scale := float32(.5 / math.Sqrt(float64(shape[len(shape)-1])))
			for i := range values {
				values[i] = (float32(next()>>8)/float32(1<<24)*2 - 1) * scale
				if strings.Contains(name, "norm.weight") {
					values[i] = 1
				} else if strings.HasSuffix(name, "_scale") {
					values[i] = .1
				} else if name == "embed.weight" {
					values[i] *= 4
				}
			}
		}
		arrays[name], err = mlx.NewFloat32(values, shape)
		if err != nil {
			t.Fatal(err)
		}
	}
	file := filepath.Join(dir, "model.safetensors")
	if err := mlx.SaveSafetensors(file, arrays, map[string]string{"synthetic": "true", "generator": "device_decode_runtime_test.go", "seed": strconv.FormatUint(uint64(deviceDecodeBenchmarkSeed), 10)}); err != nil {
		t.Fatal(err)
	}
	stat, err := os.Stat(file)
	if err != nil {
		t.Fatal(err)
	}
	model.PayloadBytes = stat.Size()
	for _, part := range []string{"config.json", "model.safetensors"} {
		model.SHA256[part] = deviceDecodeFileHash(t, filepath.Join(dir, part))
	}
	return dir, model
}

func deviceDecodeMeasure(t *testing.T, dir, backend string, prompt []int32, weights deepseek.LoadOptions) deviceDecodeBenchmarkSample {
	t.Helper()
	sample := deviceDecodeBenchmarkSample{
		Backend: backend, Prompt: slices.Clone(prompt),
		EmissionSeconds: make([]float64, 0, deviceDecodeBenchmarkTokens),
	}
	if err := mlx.Synchronize(); err != nil {
		t.Fatal(err)
	}
	before, err := mlx.GetMemoryUsage()
	if err != nil {
		t.Fatal(err)
	}
	sample.BaselineActiveBytes = before.ActiveBytes
	options := deepseek.SessionOptions{SparseExperts: true}
	if backend == "device" {
		options = deepseek.SessionOptions{DeviceDecode: true}
	}
	start := time.Now()
	m, err := Open(dir, Options{DeepSeek: &DeepSeekOptions{Weights: weights, Session: options}})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if err := mlx.Synchronize(); err != nil {
		t.Fatal(err)
	}
	sample.LoadSeconds = time.Since(start).Seconds()
	start = time.Now()
	warm, err := m.GenerateTokens(prompt, deviceDecodeBenchmarkTokens)
	if err != nil {
		t.Fatal(err)
	}
	if err := mlx.Synchronize(); err != nil {
		t.Fatal(err)
	}
	sample.WarmupSeconds = time.Since(start).Seconds()
	steady, err := mlx.GetMemoryUsage()
	if err != nil {
		t.Fatal(err)
	}
	if err := mlx.ResetPeakMemory(); err != nil {
		t.Fatal(err)
	}
	start = time.Now()
	result, err := m.generate(prompt, deviceDecodeBenchmarkTokens, lm.SamplingOptions{}, func(int32) error {
		sample.EmissionSeconds = append(sample.EmissionSeconds, time.Since(start).Seconds())
		return nil
	}, nil)
	sample.TotalSeconds = time.Since(start).Seconds()
	if err != nil {
		t.Fatal(err)
	}
	sample.Tokens = result.Tokens
	if len(result.Tokens) != deviceDecodeBenchmarkTokens || len(sample.EmissionSeconds) != deviceDecodeBenchmarkTokens || !slices.Equal(result.Tokens, warm.Tokens) {
		t.Fatal("warmup and measurement did not generate the same complete sequence")
	}
	sample.TTFTSeconds = sample.EmissionSeconds[0]
	sample.DecodeSeconds = sample.EmissionSeconds[len(sample.EmissionSeconds)-1] - sample.TTFTSeconds
	sample.DecodeTokensPerSec = float64(deviceDecodeBenchmarkTokens-1) / sample.DecodeSeconds
	sample.PrefillForwardSec, sample.DecodeForwardSec = result.PrefillSeconds, result.DecodeSeconds
	if err := mlx.Synchronize(); err != nil {
		t.Fatal(err)
	}
	usage, err := mlx.GetMemoryUsage()
	if err != nil {
		t.Fatal(err)
	}
	if steady.ActiveBytes < before.ActiveBytes || usage.PeakBytes < steady.ActiveBytes {
		t.Fatal("native allocation counters decreased; run without concurrent MLX work")
	}
	sample.SteadyActiveBytes = steady.ActiveBytes - before.ActiveBytes
	sample.PeakActiveBytes = usage.PeakBytes - before.ActiveBytes
	sample.WorkingPeakBytes = usage.PeakBytes - steady.ActiveBytes
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if err := mlx.Synchronize(); err != nil {
		t.Fatal(err)
	}
	after, err := mlx.GetMemoryUsage()
	if err != nil {
		t.Fatal(err)
	}
	sample.AfterCloseActiveBytes = after.ActiveBytes
	return sample
}

func deviceDecodeFileHash(t *testing.T, path string) string {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		t.Fatal(err)
	}
	return hex.EncodeToString(h.Sum(nil))
}

func deviceDecodeSourceHashes(t *testing.T) map[string]string {
	t.Helper()
	out := map[string]string{}
	for _, dir := range []string{"..", "../deepseek", "../deepseek/quant", "../checkpoint", "../lm", "."} {
		paths, err := filepath.Glob(filepath.Join(dir, "*.go"))
		if err != nil {
			t.Fatal(err)
		}
		for _, path := range paths {
			if strings.HasSuffix(path, "_test.go") && filepath.Base(path) != "device_decode_runtime_test.go" {
				continue
			}
			name := strings.TrimPrefix(filepath.ToSlash(path), "../")
			if dir == "." {
				name = "inference/" + filepath.Base(path)
			}
			out[name] = deviceDecodeFileHash(t, path)
		}
	}
	return out
}

func deviceDecodeSaveReport(t *testing.T, path string, report *deviceDecodeBenchmarkReport) {
	t.Helper()
	if path == "" {
		return
	}
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".device-decode-*.json")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(f.Name())
	if _, err := f.Write(append(data, '\n')); err != nil {
		f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(f.Name(), path); err != nil {
		t.Fatal(err)
	}
}
