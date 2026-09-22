# Local Safetensors Checkpoints

`checkpoint` reads either `model.safetensors` or
`model.safetensors.index.json` plus its referenced shards. Qwen loading,
fine-tuning and experimental DeepSeek loading use the same implementation.
No downloads or repository code execution occur.

## API

```go
reader, err := checkpoint.Open(modelDirectory)
if err != nil { return err }
defer reader.Close()
weight, err := reader.Get("model.embed_tokens.weight")
if err != nil { return err }
defer weight.Close()
```

- `Inspect(dir)` validates the index and safetensors headers without reading
  tensor payloads or needing native MLX. `Manifest.Names` and `Manifest.Tensor`
  expose copies of the validated metadata.
- `Open(dir)` performs the same inspection, then loads a native shard on demand
  through `Reader.Get`. Native decoding requires the `mlx` build tag.
- `Get` returns caller-owned arrays that survive switching shards or closing
  the reader. Only one native shard map is retained at a time. Native reads and
  close are serialized on the MLX worker; close is idempotent.
- `SourceFiles(dir)` lists the index and shard paths for overwrite protection.
  It validates index syntax and names, not tensor headers. Without an index it
  returns the single-file path even if that file is absent.
- `Hash(dir)` binds adapters and training state to a checkpoint. `FileHash(path)`
  hashes one file's exact bytes.
- `ReadMetadata(reader, fileSize)` validates a length prefix and header without
  reading payloads. It also describes F8_E4M3/F8_E8M0 storage for auditing;
  `Inspect` and `Open` still reject those formats. `ParseIndex(bytes)` validates
  an index independently of the filesystem. Neither API downloads anything.

The index follows Hugging Face's [sharded checkpoint format](https://huggingface.co/docs/transformers/main/big_models):
`weight_map` maps tensor names to shard filenames, with optional
`metadata.total_size` describing tensor payload bytes. Header validation follows
the [safetensors format](https://github.com/safetensors/safetensors#format).
For compatibility with MLX's serializer, an empty `__metadata__` value may also
be `null`; entries in a nonempty metadata map must still be strings.

## Validation And Limits

Duplicate names, missing or misrouted tensors, byte-count mismatches, overlapping
ranges, gaps, truncated files, invalid shapes and inconsistent total sizes are
rejected before loading native arrays. Every tensor in a referenced shard must
be indexed, even if the architecture does not use it. Scalar and empty tensors
are valid storage. Indexes and each header are limited to 16 MiB.

Supported storage dtypes are BOOL, U8/I8, U16/I16, U32/I32, U64/I64, F16, BF16,
F32 and F64. FP8, packed FP4 and unknown dtypes return `ErrUnsupportedDType`.
This is not a quantization decoder: an architecture must still validate its
required tensor names, shapes, dtypes and configuration.

Shard names must be local basenames, not paths or URLs. Files must be regular;
symlinks may resolve inside the model directory but not outside it. Materialize
a local directory (for example using `hf download --local-dir`), rather than
passing a Hugging Face cache snapshot whose symlinks point to external blobs.
Having both a single-file checkpoint and an index is rejected as ambiguous.

Keep bundles immutable while inspecting, hashing, loading and using them.
Changed shard identity, size or modification time is checked before reads;
these checks are not a security boundary against an actively changing tree.
Sharding does not provide memory offloading: models still retain their loaded
parameters. It does not make the full released DeepSeek checkpoint runnable.

## Checkpoint Identity

Single-file bundles keep their raw-file SHA-256, preserving existing adapters.
Sharded bundles hash a versioned canonical sequence of sorted tensor-to-file
routing and sorted shard filenames with their raw-file SHA-256 values. Index
whitespace and key order do not change identity. Renaming or repacking shards
does, even when the logical tensor values are identical. Adapters are not
automatically transferable between those layouts.

## Verification

```sh
go test ./checkpoint
go test -race -tags "mlx mlxruntime" ./checkpoint
go test -tags "mlx mlxruntime" ./inference -run TestSharded
```

Tests cover invalid metadata, path boundaries, hashing, array ownership,
concurrent reads/close, and reduced Qwen training and DeepSeek generation.
See [inference verification](../inference/README.md#verification) for the opt-in
pretrained Qwen resharding/reference-token test.
