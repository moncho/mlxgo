# Local Model Loading And Generation

`inference` selects a supported architecture from a local `config.json`, owns
its weights and sessions, and uses the shared `lm.Greedy` decoder. It does not
download anything, execute repository code, or evaluate chat-template scripts.

| Bundle | Text generation | Raw token IDs | LoRA adapters |
| --- | --- | --- | --- |
| Standard Qwen2, single BF16 safetensors file | Fixed Qwen single-turn chat template | Yes | mlxgo q/v adapters |
| `mlxgo.deepseek.float32.v1` experimental bundle | No | Yes | No |
| Released DeepSeek-V4.1-Flash checkpoint | Unsupported | Unsupported | Unsupported |

Qwen requires tied embeddings, SiLU, full attention, unscaled RoPE and zero
dropout. Quantized or sharded checkpoints and unknown architectures return
errors. Every required tensor's shape and dtype is checked. DeepSeek bundles
use `deepseek.Config` and `Config.ParameterShapes`, not the released checkpoint
format; unknown configuration fields are rejected. Additional tensor entries
are not loaded. Optional float32 Engram uses prepared hash/token-map metadata
and validated table/projection weights. Released FP8 Engram storage, vision,
DSpark and production quantization remain future work, not capabilities implied
by the common API.

## Go API

```go
import (
    "fmt"
    mlx "github.com/moncho/mlxgo"
    "github.com/moncho/mlxgo/inference"
)

func generate() error {
    if err := mlx.SetDefaultGPU(); err != nil { return err }
    model, err := inference.Open("models/Qwen2.5-0.5B-Instruct", inference.Options{})
    if err != nil { return err }
    defer model.Close()
    result, err := model.Generate("Explain why the sky is blue.", 64)
    if err != nil { return err }
    fmt.Println(result.Text)
    return nil
}
```

Use `Options{Adapters: "checkpoints/qwen-lora.safetensors"}` to load a saved
mlxgo adapter. Its base-checkpoint SHA-256 and configuration are validated.
The existing `cmd/finetune-qwen` training workflow is unchanged; this loader
provides a common inference path for its output, not architecture-independent
training.

`Inspect(dir)` validates configuration without loading native arrays and also
works in the default stub build. `Info.Text` and `Info.Adapters` describe available
architecture support, not proof that files exist; `Open` additionally validates
the tokenizer, special IDs and weights. A Qwen bundle requires `tokenizer.json`
even when raw token generation will be used. Tokenizer IDs must fit the model's
vocabulary. Padded embedding rows are allowed.

`GenerateTokens(promptIDs, count, stopIDs...)` bypasses chat formatting and
returns raw IDs, including a matching stop token. `Generate` encodes the Qwen
chat prompt and omits terminal EOS from its returned IDs and text. The requested
generation budget must fit the context, allowing for the last generated token
not yet having been consumed. Timing fields measure forward passes, including
worker queue time, not end-to-end decoding throughput.

For incremental use, call `NewSession`, `Step`, `Position`, and `Close` through
`lm.Session`. Returned logits are caller-owned `[1,1,vocab]` arrays. DeepSeek
accepts a multi-token prefill and then one token per step. Use independent
sessions for separate sequences. Concurrent model generation is supported;
native steps run on the existing MLX worker. `Model.Close` closes outstanding
sessions before releasing weights. Concurrent operations finish their current
step or return `ErrClosed`. Close methods are idempotent. As with `mlx.Batch`,
never block the MLX worker waiting for MLX work delegated to another goroutine.

## Commands

Real pretrained Qwen (download instructions in the root README):

```sh
go run -tags mlx ./cmd/generate -model models/Qwen2.5-0.5B-Instruct \
  -device gpu -prompt "Explain why the sky is blue." -max-tokens 64
```

Export the reduced untrained fixture to a **new** directory, then use the same
command. Export refuses to overwrite an existing directory:

```sh
go run -tags mlx ./cmd/deepseek-smoke -export models/deepseek-synthetic
go run -tags mlx ./cmd/generate -model models/deepseek-synthetic \
  -device gpu -tokens 1,5,2,9,4 -max-tokens 4
```

Expected IDs are `[21 21 21 18]`. These synthetic weights do not represent a
pretrained language model. `-tokens` and an explicit `-prompt` are mutually
exclusive; DeepSeek text requests fail with `ErrTextUnsupported`.

To exercise Engram too, export a separate bundle using
`-fixture deepseek/testdata/model_engram.json -export models/deepseek-engram`.
The same `cmd/generate -model models/deepseek-engram -tokens ...` path then uses
its prepared metadata and Engram weights. It still produces raw synthetic IDs.

## Verification

```sh
go test ./...
go test -tags "mlx mlxruntime" ./...
MLXGO_QWEN2_DIR="$PWD/models/Qwen2.5-0.5B-Instruct" \
MLXGO_QWEN2_ADAPTERS="$PWD/checkpoints/qwen-lora.safetensors" \
  go test -tags "mlx mlxruntime" ./inference -run TestRealQwenLoaderParity -v
```

CI runs reduced Qwen BF16 loading with trained adapters, reduced DeepSeek CPU/GPU
generation, concurrent sessions, close-during-use, and bad-config/weight tests
without downloading models. Real-checkpoint parity is opt-in: it compares the
first 32 Qwen reference tokens and the existing architecture-specific generator,
then repeats with saved adapters when supplied.
