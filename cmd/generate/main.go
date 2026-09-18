package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/moncho/mlxgo"
	"github.com/moncho/mlxgo/inference"
)

type options struct {
	dir, prompt, adapters, device string
	maxTokens                     int
	tokens                        []int32
}

func parse(args []string) (options, error) {
	var o options
	f := flag.NewFlagSet("generate", flag.ContinueOnError)
	f.StringVar(&o.dir, "model", "models/Qwen2.5-0.5B-Instruct", "supported local model directory")
	f.StringVar(&o.prompt, "prompt", "Explain why the sky is blue in one sentence.", "user prompt")
	f.StringVar(&o.adapters, "adapters", "", "optional mlxgo LoRA safetensors")
	f.StringVar(&o.device, "device", "gpu", "cpu or gpu")
	f.IntVar(&o.maxTokens, "max-tokens", 64, "maximum generated tokens")
	var raw string
	f.StringVar(&raw, "tokens", "", "raw prompt token IDs, comma-separated (instead of -prompt)")
	if err := f.Parse(args); err != nil {
		return o, err
	}
	if f.NArg() != 0 {
		return o, fmt.Errorf("unexpected positional arguments")
	}
	if o.maxTokens <= 0 {
		return o, fmt.Errorf("max-tokens must be positive")
	}
	if o.device != "cpu" && o.device != "gpu" {
		return o, fmt.Errorf("device must be cpu or gpu")
	}
	var hasTokens, hasPrompt bool
	f.Visit(func(v *flag.Flag) {
		if v.Name == "tokens" {
			hasTokens = true
		}
		if v.Name == "prompt" {
			hasPrompt = true
		}
	})
	if hasTokens && hasPrompt {
		return o, fmt.Errorf("use either -tokens or -prompt")
	}
	if hasTokens {
		for _, v := range strings.Split(raw, ",") {
			n, err := strconv.ParseInt(strings.TrimSpace(v), 10, 32)
			if err != nil || n < 0 {
				return o, fmt.Errorf("invalid token ID %q", v)
			}
			o.tokens = append(o.tokens, int32(n))
		}
	}
	return o, nil
}

func run(o options, output io.Writer) error {
	info, err := inference.Inspect(o.dir)
	if err != nil {
		return err
	}
	if o.tokens == nil && !info.Text {
		return inference.ErrTextUnsupported
	}
	if o.device == "gpu" {
		err = mlx.SetDefaultGPU()
	} else {
		err = mlx.SetDefaultCPU()
	}
	if err != nil {
		return err
	}
	m, err := inference.Open(o.dir, inference.Options{Adapters: o.adapters})
	if err != nil {
		return err
	}
	defer m.Close()
	var r inference.Result
	if o.tokens != nil {
		r, err = m.GenerateTokens(o.tokens, o.maxTokens)
	} else {
		r, err = m.Generate(o.prompt, o.maxTokens)
	}
	if err != nil {
		return err
	}
	if o.tokens != nil {
		fmt.Fprintf(output, "Generated token IDs: %v\n", r.Tokens)
	} else {
		fmt.Fprintln(output, r.Text)
	}
	if info.Format == "mlxgo.deepseek.float32.v1" {
		fmt.Fprintln(output, "Experimental float32 bundle; not a released DeepSeek checkpoint.")
	}
	if r.DecodeSeconds > 0 {
		fmt.Fprintf(output, "\n%s (%s): prefill=%d tokens (%.3fs), decode forward=%.1f tokens/s\n", info.Architecture, o.device, r.PrefillTokens, r.PrefillSeconds, float64(max(0, len(r.Tokens)-1))/r.DecodeSeconds)
	}
	return nil
}

func main() {
	o, err := parse(os.Args[1:])
	if errors.Is(err, flag.ErrHelp) {
		return
	}
	if err == nil {
		err = run(o, os.Stdout)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
