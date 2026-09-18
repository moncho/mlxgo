"""Pinned sources and explicitly unquantized CPU kernel translations for tests."""
import hashlib
import urllib.request

import torch

REVISION = "df42c109f1defefcbfcedbe7d905718a12266e40"
HASHES = {
    "model.py": "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65",
    "kernel.py": "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455",
}


def load_sources(directory=None):
    sources = {}
    for name, checksum in HASHES.items():
        if directory:
            source = (directory / name).read_bytes()
        else:
            url = f"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/{REVISION}/inference/{name}"
            with urllib.request.urlopen(url, timeout=60) as response:
                source = response.read()
        if hashlib.sha256(source).hexdigest() != checksum:
            raise ValueError(f"Source checksum mismatch: {name}")
        sources[name] = source.decode()
    return sources


def sinkhorn(mixes, scale, base, hc, iterations, eps):
    """Translation of hc_split_sinkhorn_kernel, not a CUDA run."""
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2*hc] * scale[1] + base[hc:2*hc])
    comb = (mixes[..., 2*hc:] * scale[2] + base[2*hc:]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def sparse_attention(q, kv, sink, indices, scale):
    """Mathematical sparse_attn oracle, without BF16 rounding."""
    rows = []
    for b in range(q.shape[0]):
        seq = []
        for t in range(q.shape[1]):
            idx = indices[b, t]
            selected = kv[b, idx[idx >= 0]]
            logits = q[b, t] @ selected.T * scale
            prob = torch.cat([logits, sink[:, None]], dim=-1).softmax(-1)
            seq.append(prob[:, :-1] @ selected)
        rows.append(torch.stack(seq))
    return torch.stack(rows)
