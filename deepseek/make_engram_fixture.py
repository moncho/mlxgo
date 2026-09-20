"""Generate exact integer hashes and unquantized Engram forward/gradient oracles.

Run with Python 3.12, torch==2.14.0, numpy==2.5.3, sympy==1.14.0,
tokenizers==0.22.2. All source files are checked against pinned SHA-256 hashes.
"""
import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy
import sympy
import tokenizers
import torch
import torch.nn as nn

from engram_reference import prepare
from reference import HASHES, ENGRAM_HASHES, REVISION, load_sources

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-dir", type=Path)
args = parser.parse_args()
sources = load_sources(args.source_dir, engram=True)
ns, tokenizer, settings, config = prepare(sources)
torch.set_num_threads(1)
torch.manual_seed(482)
hash_args = SimpleNamespace(**settings, max_batch_size=2, max_seq_len=32)
layout = ns["EngramLayout"].from_args(hash_args)

tokens = [[0, 1, 2, 9, 10, 6, 4, 5, 11, 13, 18], [3, 0, 2, 10, 9, 7, 5, 4, 12, 14, 19]]
mask = [[True, True, False, True, True, False, False, True, True, True, True],
        [False, True, True, True, False, True, True, True, True, False, True]]
cases = []
for masked in (False, True):
    tensor_mask = torch.tensor(mask) if masked else None
    reference = ns["NgramHashState"](hash_args, layout, tokenizer)(torch.tensor(tokens), 0, tensor_mask)
    for chunks in ([11], [1] * 11, [2, 1, 3, 2, 3], [5, 6]):
        state = ns["NgramHashState"](hash_args, layout, tokenizer)
        start, outputs = 0, []
        for count in chunks:
            part_mask = tensor_mask[:, start:start+count] if masked else None
            outputs.append(state(torch.tensor(tokens)[:, start:start+count], start, part_mask))
            start += count
        assert torch.equal(torch.cat(outputs, 1), reference)
    cases.append(dict(mask=mask if masked else None, hashes=reference.flatten().tolist()))


class FloatLinear(nn.Linear):
    def __init__(self, i, o):
        super().__init__(i, o, bias=False, dtype=torch.float32)


model_ns = dict(torch=torch, nn=nn, Linear=FloatLinear, ParallelEngramEmbedding=nn.Embedding)
node = next(n for n in ast.parse(sources["model.py"]).body if isinstance(n, ast.ClassDef) and n.name == "Engram")
future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
exec(compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), "pinned-model.py", "exec"), model_ns)
engram = model_ns["Engram"](SimpleNamespace(dim=4, hc_mult=3, norm_eps=1e-5), 3, layout)
with torch.no_grad():
    for name, param in engram.named_parameters():
        param.normal_(std=.4)
    # Nontrivial normalization weights, signs and gate saturation.
    engram.q_weight.mul_(3)
    engram.k_weight.mul_(3)
x = torch.randn(2, 11, 3, 4, requires_grad=True)
hashes = torch.tensor(cases[1]["hashes"]).reshape(2, 11, 3, 6)[:, :, 1, :]
tensors = {}


def save(name, value):
    tensors[name] = dict(shape=list(value.shape), data=value.detach().flatten().tolist())


save("x", x)
for name, param in engram.named_parameters():
    save(name, param)
for name, token_mask in (("unmasked", None), ("masked", torch.tensor(mask))):
    y = engram(x, hashes, token_mask)
    save(name, y)
    grads = torch.autograd.grad(y.square().mean(), [x, *engram.parameters()])
    for grad_name, grad in zip(["x", *[n for n, _ in engram.named_parameters()]], grads):
        save(name + "_grad_" + grad_name, grad)
zero = torch.zeros_like(x)
save("zero_output", engram(zero, hashes))

# Isolate the signed square-root floor: positive/negative sub-threshold dots
# must not collapse to the same gate, and exactly zero uses the positive sign.
edge_layout = SimpleNamespace(layer_ids=(0,), num_embeddings=(1,), head_dim=1,
                              max_ngram_size=2, n_heads=1)
edge = model_ns["Engram"](SimpleNamespace(dim=2, hc_mult=1, norm_eps=1e-5), 0, edge_layout)
with torch.no_grad():
    edge.embed.weight.fill_(1)
    edge.wkv.weight.copy_(torch.tensor([[1.], [0.], [2.], [-1.]]))
edge_x = torch.tensor([[[[v, 1.]] for v in (-1., -1e-7, -0., 0., 1e-7, 1.)]], requires_grad=True)
edge_y = edge(edge_x, torch.zeros(1, 6, 1, dtype=torch.int64))
save("edge_x", edge_x)
for name, param in edge.named_parameters():
    save("edge_" + name, param)
save("edge_output", edge_y)
save("edge_grad_x", torch.autograd.grad(edge_y.square().mean(), edge_x)[0])

result = dict(revision=REVISION, sha256=HASHES | ENGRAM_HASHES,
              versions=dict(torch=torch.__version__, numpy=numpy.__version__,
                            sympy=sympy.__version__, tokenizers=tokenizers.__version__),
              config=config, tokens=tokens, cases=cases, epsilon=1e-5,
              lookup_hashes=hashes.flatten().tolist(), tensors=tensors,
              limitations=["Synthetic single-token decode/raw strings, not released tokenizer",
                           "FP8 embedding/dequantization replaced by float32 nn.Embedding",
                           "Untrained float32 weights; no BF16 rounding"])
destination = Path(__file__).parent / "testdata" / "engram.json"
destination.write_text(json.dumps(result, separators=(",", ":")) + "\n")
print(f"Wrote {len(cases)} batched hash cases and {len(tensors)} forward/gradient tensors")
