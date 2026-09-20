"""Run the pinned official Transformer on small, untrained float32 weights.

Architectural forward methods, index selection and block wiring are unchanged.
One cache-publication correction is explicitly recorded below. Replacements:
float32 Linear, no-op cache quantization,
CPU mathematical kernels from reference.py, all-position head output, and
disabled vision/DSpark and optional float32 Engram with prepared synthetic
tokenizer metadata. This is NOT released-checkpoint parity.
"""
import argparse
import ast
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache, partial
import json
import math
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from reference import HASHES, REVISION, load_sources, sinkhorn, sparse_attention

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-dir", type=Path)
parser.add_argument("--engram", action="store_true", help="Enable prepared float32 Engram in a separate fixture")
args = parser.parse_args()
sources = load_sources(args.source_dir, engram=args.engram)


class FloatLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, dtype=None):
        super().__init__(in_features, out_features, bias=bias, dtype=torch.float32)


class NoEngram:
    @staticmethod
    def from_args(args):
        assert not args.engram_layer_ids
        return None


namespace = dict(torch=torch, nn=nn, F=F, dist=dist, math=math,
                 contextmanager=contextmanager, dataclass=dataclass, Literal=Literal,
                 lru_cache=lru_cache, __name__=__name__,
                 world_size=1, rank=0, default_dtype=torch.float32,
                 Linear=FloatLinear, ColumnParallelLinear=FloatLinear,
                 RowParallelLinear=FloatLinear, linear=F.linear,
                 fp8_block_size=32, fp4_block_size=32, scale_fmt="ue8m0",
                 scale_dtype=torch.float32, act_quant=lambda *a, **k: None,
                 fp4_act_quant=lambda *a, **k: None,
                 hc_split_sinkhorn=sinkhorn, sparse_attn=sparse_attention,
                 EngramLayout=NoEngram)
if args.engram:
    from engram_reference import prepare
    from reference import ENGRAM_HASHES
    engram_ns, tokenizer, engram_settings, engram_config = prepare(sources)
    namespace.update(EngramLayout=engram_ns["EngramLayout"], NgramHashState=engram_ns["NgramHashState"],
                     ParallelEngramEmbedding=nn.Embedding)
names = {"set_dtype", "ModelArgs", "ParallelEmbedding", "RMSNorm",
         "precompute_freqs_cis", "apply_rotary_emb", "get_window_topk_idxs",
         "Compressor", "Indexer", "select_candidate_blocks", "Attention",
         "Gate", "Expert", "MoE", "Block", "ParallelHead",
         "make_identity_pre_mix", "SharedAttentionRuntime", "Transformer", "sample"}
if args.engram:
    names.add("Engram")
nodes = [n for n in ast.parse(sources["model.py"]).body
         if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + nodes,
                    type_ignores=[])
exec(compile(ast.fix_missing_locations(module), "pinned-model.py", "exec"), namespace)

config = dict(format="mlxgo.deepseek.float32.v1", vocab_size=32, dim=8,
              moe_inter_dim=6, n_layers=8, n_heads=4, n_routed_experts=4,
              router=dict(TopK=2, Score="sqrtsoftplus", Temperature=.7, Scale=1.5, Normalize=True),
              swiglu_limit=.7, q_lora_rank=4, head_dim=8, rope_head_dim=4,
              o_groups=2, o_lora_rank=3, window_size=3, max_seq_len=32,
              compress_ratios=[0, 2, 2, 2, 1, 1, 1, 1],
              kv_source_layers=[1, 4], index_source_layers=[1, 2, 4, 6],
              rope_theta=1000., compress_rope_theta=100., original_seq_len=0,
              rope_factor=4., beta_fast=32., beta_slow=1.,
              index_n_heads=2, index_head_dim=8, index_topk=2,
              candidate_source_layer=4, candidate_topk_blocks=2, candidate_block_size=2,
              hyper=dict(Streams=3, Iterations=20, NormEpsilon=1e-20, Epsilon=1e-6))
if args.engram:
    config["engram"] = engram_config
overrides = {k: v for k, v in config.items() if k not in ("format", "router", "hyper", "engram")}
overrides.update(max_batch_size=1, n_mtp_layers=0, dtype="bf16", expert_dtype=None,
                 n_activated_experts=2, gate_temp=.7, route_scale=1.5,
                 hc_mult=3, hc_sinkhorn_iters=20, hc_eps=1e-6, norm_eps=1e-20,
                 temperature=0)
if args.engram:
    overrides.update(engram_settings)
model_args = namespace["ModelArgs"](**overrides)
torch.set_num_threads(1)
torch.manual_seed(9217)


def fresh(state=None, yarn=False, publish_index_cache=True):
    namespace["shared_attn"] = namespace["SharedAttentionRuntime"]()
    model_args.original_seq_len = 4 if yarn else 0
    model = namespace["Transformer"](model_args, tokenizer=tokenizer) if args.engram else namespace["Transformer"](model_args)
    with torch.no_grad():
        if state is not None:
            model.load_state_dict(state, strict=True)
        else:
            for name, p in model.named_parameters():
                if "norm.weight" in name:
                    p.copy_(1 + torch.randn_like(p) * .03)
                elif name == "embed.weight":
                    p.uniform_(.1, .4)
                elif any(key in name for key in ("wq_a.weight", "compressor.wkv.weight", "indexer.wk.weight", "indexer.wq_b.weight")):
                    p.uniform_(.03, .2)
                else:
                    p.normal_(std=.15)
    # Only the head's position slicing changes, not the Transformer forward.
    model.head.forward = partial(model.head.forward, full_logits=True)
    if publish_index_cache:
        # The upstream owner publishes only when a NEW latent is produced. On
        # partial groups that leaves the previous token's last owner's keys in
        # the global slot. Publish the current owner's existing cache as well.
        def publish(module, inputs):
            namespace["shared_attn"].index_k = module.k_cache
        for layer in model.layers:
            indexer = layer.attn.indexer
            if indexer is not None and indexer.owns_k:
                indexer.register_forward_pre_hook(publish)
    return model


def tensor(value):
    return dict(shape=list(value.shape), data=value.detach().cpu().flatten().tolist())


base = fresh()
state = {k: v.clone() for k, v in base.state_dict().items()}
tokens = [1, 5, 2, 9, 4, 12, 6, 3, 8, 15, 7, 11, 14]
cases = []
# Preserve evidence that this adjustment is necessary for this pinned source.
unpatched = fresh(state, publish_index_cache=False)
unpatched_full = unpatched(torch.tensor([tokens]))[1]
unpatched = fresh(state, publish_index_cache=False)
unpatched_steps = torch.cat([unpatched(torch.tensor([[t]]), i)[1] for i, t in enumerate(tokens)], dim=1)
unpatched_error = (unpatched_full-unpatched_steps).abs().max().item()
if unpatched_error < 1e-3:
    raise AssertionError("Pinned reference cache-publication regression was not reproduced")
for yarn in (False, True):
    full_model = fresh(state, yarn)
    full = full_model(torch.tensor([tokens]))[1]
    for prefill in (1, 2, 3, 4, 5, 8, len(tokens)):
        model = fresh(state, yarn)
        outputs = [model(torch.tensor([tokens[:prefill]]))[1]]
        for position in range(prefill, len(tokens)):
            outputs.append(model(torch.tensor([[tokens[position]]]), position)[1])
        incremental = torch.cat(outputs, dim=1)
        max_error = (incremental - full).abs().max().item()
        if max_error > 2e-5:
            print("Per-position errors:", (incremental-full).abs().amax(-1).tolist())
            raise AssertionError(f"Official reference prefill/decode mismatch: {prefill=} {yarn=} {max_error=}")
        cases.append(dict(prefill=prefill, yarn=yarn, logits=tensor(incremental), reference_chunk_error=max_error))

if max(abs(a-b) for a,b in zip(cases[0]["logits"]["data"],cases[7]["logits"]["data"])) < 1e-4:
    raise AssertionError("Fixture does not meaningfully exercise YaRN")

# A second model with different weights detects accidentally process-global caches.
second = fresh()
second_state = {k: v.clone() for k, v in second.state_dict().items()}
second_logits = second(torch.tensor([tokens]))[1]

result = dict(revision=REVISION, sha256=HASHES | (ENGRAM_HASHES if args.engram else {}), torch=torch.__version__, seed=9217,
              reference_adjustments=["Publish Indexer owner's existing k_cache before every invocation, including incomplete compression groups"],
              unpatched_reference_chunk_error=unpatched_error,
              config=config, parameters={k: tensor(v) for k, v in state.items()},
              tokens=tokens, cases=cases, second_parameters={k: tensor(v) for k, v in second_state.items()},
              second_logits=tensor(second_logits),
              limitations=["Untrained float32 text backbone", "Cache quantization disabled",
                           "Sparse and Sinkhorn CUDA kernels replaced by CPU mathematical translations",
                           "Engram, vision, DSpark and pretrained tokenization disabled"])
if args.engram:
    result["limitations"][-1] = "Float32 Engram table, prepared synthetic token map; vision, DSpark and pretrained tokenization disabled"
    import numpy, sympy, tokenizers
    result["engram_metadata_versions"] = dict(numpy=numpy.__version__, sympy=sympy.__version__, tokenizers=tokenizers.__version__)
destination = Path(__file__).parent / "testdata" / ("model_engram.json" if args.engram else "model.json")
destination.write_text(json.dumps(result, separators=(",", ":")) + "\n")
print(f"Wrote {len(state)} parameters, {len(cases)} decode schedules; {sum(v.numel() for v in state.values())} scalar parameters")
print("Maximum reference chunk error:", max(case["reference_chunk_error"] for case in cases))
