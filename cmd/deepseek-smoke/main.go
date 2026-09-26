package main

import (
	"compress/gzip"
	"encoding/json"
	"flag"
	"fmt"
	mlx "github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/deepseek"
	"github.com/moncho/mlxgo/lm"
	"io"
	"os"
	"path/filepath"
)

func run() error {
	path := flag.String("fixture", "", "untrained fixture (defaults to the matching float32 or quantized-cache fixture)")
	quantized := flag.Bool("quantized-caches", false, "experimental FP8/FP4 caches with float32 weights")
	device := flag.String("device", "gpu", "cpu or gpu")
	export := flag.String("export", "", "export synthetic bundle to a new directory")
	flag.Parse()
	if *path == "" {
		*path = "deepseek/testdata/model.json"
		if *quantized {
			*path = "deepseek/testdata/model_quantized.json.gz"
		}
	}
	switch *device {
	case "cpu":
		if err := mlx.SetDefaultCPU(); err != nil {
			return err
		}
	case "gpu":
		if err := mlx.SetDefaultGPU(); err != nil {
			return err
		}
	default:
		return fmt.Errorf("unknown device %q", *device)
	}
	var f struct {
		Revision   string          `json:"revision"`
		Config     deepseek.Config `json:"config"`
		Parameters map[string]struct {
			Shape []int     `json:"shape"`
			Data  []float32 `json:"data"`
		} `json:"parameters"`
		Tokens []int32 `json:"tokens"`
	}
	file, err := os.Open(*path)
	if err != nil {
		return err
	}
	defer file.Close()
	var reader io.Reader = file
	if filepath.Ext(*path) == ".gz" {
		z, err := gzip.NewReader(file)
		if err != nil {
			return err
		}
		defer z.Close()
		reader = z
	}
	decoder := json.NewDecoder(io.LimitReader(reader, 32<<20))
	if err = decoder.Decode(&f); err != nil {
		return err
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		return fmt.Errorf("trailing fixture data: %v", err)
	}
	if f.Revision != deepseek.ReferenceRevision || len(f.Tokens) < 5 {
		return fmt.Errorf("incompatible reduced fixture")
	}
	weights := make(map[string]mlx.Array, len(f.Parameters))
	defer func() {
		for _, a := range weights {
			a.Close()
		}
	}()
	for name, value := range f.Parameters {
		a, err := mlx.NewFloat32(value.Data, value.Shape)
		if err != nil {
			return err
		}
		weights[name] = a
	}
	model, err := deepseek.NewModel(f.Config, weights)
	if err != nil {
		return err
	}
	defer model.Close()
	if *export != "" {
		data, err := json.MarshalIndent(f.Config, "", "  ")
		if err != nil {
			return err
		}
		if err = os.Mkdir(*export, 0755); err != nil {
			return err
		}
		if err = os.WriteFile(filepath.Join(*export, "config.json"), data, 0644); err != nil {
			return err
		}
		if err = mlx.SaveSafetensors(filepath.Join(*export, "model.safetensors"), weights, map[string]string{"synthetic": "true", "format": deepseek.Float32Format, "reference_revision": f.Revision}); err != nil {
			return err
		}
		fmt.Printf("Exported synthetic bundle: %s\n", *export)
	}
	session, err := model.NewSessionWithOptions(deepseek.SessionOptions{QuantizedCaches: *quantized})
	if err != nil {
		return err
	}
	defer session.Close()
	tokens, err := lm.Greedy(session, f.Tokens[:5], 4)
	if err != nil {
		return err
	}
	fmt.Printf("Untrained float32 DeepSeek text backbone (%d layers, %s)\n", f.Config.Layers, *device)
	if *quantized {
		fmt.Println("Experimental FP8/FP4 caches and index queries; float32 projections.")
	}
	if f.Config.Engram != nil {
		fmt.Printf("Engram enabled on layers %v (prepared synthetic token map)\n", f.Config.Engram.Layers)
	}
	fmt.Printf("Prompt token IDs: %v\nGenerated token IDs: %v\n", f.Tokens[:5], tokens)
	fmt.Println("Synthetic weights only; this is not a pretrained text completion.")
	return nil
}
func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
