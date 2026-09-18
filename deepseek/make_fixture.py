"""Generate small CPU float32 fixtures; never downloads model weights.

Gate, Expert, MoE, Compressor and Block's hc_* methods are executed directly
from the pinned official AST. The two CUDA/TileLang kernels are replaced by
explicit PyTorch mathematical translations; their BF16/FP4 rounding is NOT
covered by these fixtures. Run with Python 3.12 and torch==2.14.0.
"""

import argparse
import ast
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from reference import HASHES, REVISION, load_sources, sinkhorn, sparse_attention

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-dir", type=Path, help="Use already downloaded, hash-verified sources")
args = parser.parse_args()
sources = load_sources(args.source_dir)


# Strip constructors and unrelated class attributes, not the tested methods.
namespace = {"torch": torch, "F": F, "linear": F.linear, "world_size": 1,
             "hc_split_sinkhorn": sinkhorn}
methods = {"Gate": ["forward"], "Expert": ["forward"], "MoE": ["forward"],
           "Compressor": ["forward"], "Block": ["hc_mixes", "hc_pre", "hc_post"]}
for node in ast.parse(sources["model.py"]).body:
    if isinstance(node, ast.ClassDef) and node.name in methods:
        node.bases = []
        node.decorator_list = []
        node.body = [method for method in node.body if isinstance(method, ast.FunctionDef)
                     and method.name in methods[node.name]]
        exec(compile(ast.Module(body=[node], type_ignores=[]), "pinned-model.py", "exec"), namespace)
        cls = namespace[node.name]
        if hasattr(cls, "forward"):
            cls.__call__ = cls.forward

torch.set_num_threads(1)
torch.manual_seed(719)
tensors = {}


def save(name, tensor):
    tensor = tensor.detach().cpu()
    tensors[name] = {"shape": list(tensor.shape), "data": tensor.flatten().tolist()}
    return tensor


def rand(name, shape, scale=0.4):
    return save(name, torch.randn(shape, dtype=torch.float32) * scale)


x = rand("x", (6, 4), 1.5).requires_grad_()
gate = namespace["Gate"]()
gate.weight = rand("router_weight", (4, 4))
gate.bias = rand("router_bias", (4,), 0.3)
gate.bias_vl = None
gate.gate_temp = 0.7
gate.route_scale = 1.5
gate.topk = 2
gate.norm_topk_prob = True
for score in ("sqrtsoftplus", "sigmoid", "softmax"):
    gate.score_func = score
    weights, indices = gate(x)
    save("route_" + score, weights)
    save("indices_" + score, indices)
gate.score_func = "sqrtsoftplus"
gate.topk = 1
weights, indices = gate(x)
save("route_top1", weights)
save("indices_top1", indices)
gate.topk = 2
gate.norm_topk_prob = False
weights, indices = gate(x)
save("route_unnormalized", weights)
gate.norm_topk_prob = True


def make_expert(prefix):
    expert = namespace["Expert"]()
    expert.swiglu_limit = 0.7  # deliberately exercise asymmetric clamps
    w1 = rand(prefix + "_gate", (7, 4), 0.7)
    w3 = rand(prefix + "_up", (7, 4), 0.7)
    w2 = rand(prefix + "_down", (4, 7), 0.7)
    expert.w1 = lambda value: F.linear(value, w1)
    expert.w3 = lambda value: F.linear(value, w3)
    expert.w2 = lambda value: F.linear(value, w2)
    return expert


moe = namespace["MoE"]()
moe.dim = 4
moe.gate = gate
moe.n_routed_experts = moe.experts_end_idx = 4
moe.experts_start_idx = 0
moe.experts = [make_expert(f"expert{i}") for i in range(4)]
moe.shared_experts = make_expert("shared")
y = moe(x)
save("moe", y)
save("moe_gradient", torch.autograd.grad(y.square().mean(), x)[0])

block = namespace["Block"]()
block.hc_mult = 3
block.hc_sinkhorn_iters = 20
block.hc_eps = 1e-6
block.norm_eps = 1e-20
hx = rand("hx", (2, 3, 3, 4)).requires_grad_()
projection = rand("hyper_projection", (15, 12))
scale = rand("hyper_scale", (3,))
base = rand("hyper_base", (15,))
pre, post, comb = block.hc_mixes(hx, projection, scale, base)
for name, value in (("pre", pre), ("post", post), ("comb", comb)):
    save(name, value)
collapsed = block.hc_pre(hx, pre)
save("collapsed", collapsed)
expanded = block.hc_post(collapsed, hx, post, comb)
save("expanded", expanded)
save("hyper_gradient", torch.autograd.grad(expanded.square().mean(), hx)[0])

q = rand("q", (2, 3, 2, 4)).requires_grad_()
kv = rand("kv", (2, 5, 4))
sink = rand("sink", (2,), 1)
indices = torch.tensor([[[0, -1, -1], [0, 1, -1], [2, 0, 1]],
                        [[-1, -1, -1], [3, 1, -1], [4, 2, 0]]])
save("sparse_indices", indices)
attention = sparse_attention(q, kv, sink, indices, 0.5)
save("attention", attention)
save("attention_gradient", torch.autograd.grad(attention.square().mean(), q)[0])

compressor = namespace["Compressor"]()
cx = rand("cx", (2, 6, 4))
ckv = rand("compress_kv", (3, 4))
cg = rand("compress_gate", (3, 4))
cnorm = rand("compress_norm", (3,))
compressor.wkv = lambda value: F.linear(value, ckv)
compressor.wgate = lambda value: F.linear(value, cg)
compressor.norm = lambda value: value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-20) * cnorm
for ratio in (1, 2, 3):
    compressor.compress_ratio = ratio
    save(f"compress{ratio}", compressor(cx, 0))

result = {"revision": REVISION, "sha256": HASHES, "torch": torch.__version__,
          "dtype": "float32", "seed": 719, "tensors": tensors,
          "limitations": ["No released weights or quantization", "No CUDA kernel execution",
                          "Sinkhorn and sparse attention use mathematical translations",
                          "No complete Transformer, CSA2 cache, Engram, vision, or DSpark"]}
destination = Path(__file__).parent / "testdata" / "reference.json"
destination.parent.mkdir(exist_ok=True)
destination.write_text(json.dumps(result, indent=2) + "\n")
print(f"Wrote {len(tensors)} deterministic reference tensors to {destination}")
