"""Generate independent Hugging Face tokenizer and mlx-lm inference fixtures.

Run from the repo root with the requirements in requirements-reference.txt.
The reduced tokenizer retains only merge rules exercised by this corpus; no
model weights or full tokenizer vocabulary are committed.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import random
import unicodedata

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import ConcatenateKVCache
from tokenizers import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/Qwen2.5-0.5B-Instruct")
args = parser.parse_args()
root = Path(__file__).resolve().parent.parent
model_dir = Path(args.model)
tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
full = json.loads((model_dir / "tokenizer.json").read_text())
prompt = "Explain why the sky is blue in one sentence."
model, chat_tokenizer = load(model_dir)
chat = chat_tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
texts = ["", "Hello, world!", "I'm WE'RE he'll they've I'd", "1234567890 12.5", "hello", " hello", "  hello", "   ", "\tword\t\n", "foo\r\nbar", "café", "cafe\u0301", "中文测试 日本語", "😀🚀👩‍💻", "\u00a0hello\u2003world\u3000!", "a\u0085b\u2028c", "punctuation!?...\n\n", "<|im_start|>user\nHi<|im_end|>", chat]
rng = random.Random(42)
alphabet = "abc XYZ123!?\t\n\r\u00a0\u2003é中😀"
texts += ["".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 80))) for _ in range(100)]
entries, pieces = [], set()
for text in texts:
    normalized = unicodedata.normalize("NFC", text)
    parts = [p for p, _ in tokenizer.pre_tokenizer.pre_tokenize_str(normalized)]
    pieces.update(parts)
    entries.append({"text": text, "normalized": normalized, "pieces": parts,
                    "ids": tokenizer.encode(text, add_special_tokens=False).ids})
merges = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in full["model"]["merges"]]
ranks = {pair: i for i, pair in enumerate(merges)}
used, symbols = set(), {s for s in full["model"]["vocab"] if len(s) == 1}
for piece in pieces:
    parts = list(piece)
    while len(parts) > 1:
        choices = [(ranks[(a, b)], i) for i, (a, b) in enumerate(zip(parts, parts[1:])) if (a, b) in ranks]
        if not choices:
            break
        rank, i = min(choices)
        used.add(rank)
        symbols.update([parts[i], parts[i+1], parts[i]+parts[i+1]])
        parts[i:i+2] = [parts[i]+parts[i+1]]
full["model"]["merges"] = [list(merges[r]) for r in sorted(used)]
full["model"]["vocab"] = {s: i for s, i in full["model"]["vocab"].items() if s in symbols}
bpe_dir = root / "bpe" / "testdata"
bpe_dir.mkdir(parents=True, exist_ok=True)
(bpe_dir / "tokenizer.json").write_text(json.dumps(full, ensure_ascii=True, separators=(",", ":")) + "\n")
(bpe_dir / "parity.json").write_text(json.dumps(entries, ensure_ascii=True, indent=2) + "\n")

ids = chat_tokenizer.encode(chat, add_special_tokens=False)
# Match the selected cache strategy using mlx-lm's own implementation. Cache
# strides can select different bf16 kernels even when the equations are equal.
cache = [ConcatenateKVCache() for _ in model.layers]
logits = model(mx.array([ids]), cache=cache)[:, -1, :]
mx.eval(logits)
out_dir = root / "qwen2" / "testdata"
out_dir.mkdir(parents=True, exist_ok=True)
mx.save(str(out_dir / "prefill.npy"), logits.astype(mx.float32))
tokens = []
for step in range(32):
    token = int(mx.argmax(logits, axis=-1).item())
    tokens.append(token)
    if step < 31:
        logits = model(mx.array([[token]]), cache=cache)[:, -1, :]
        mx.eval(logits)
fixture = {"model": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
           "prompt": prompt, "chat": chat, "input_ids": ids, "tokens": tokens,
           "cache": "mlx_lm.models.cache.ConcatenateKVCache",
           "max_logit_error": 0.25, "minimum_matching_tokens": 32,
           "versions": {p: importlib.metadata.version(p) for p in ("mlx", "mlx-lm", "tokenizers", "transformers")}}
(out_dir / "golden.json").write_text(json.dumps(fixture, indent=2) + "\n")
print("Reference completion:", chat_tokenizer.decode(tokens))
print("Wrote", len(entries), "tokenizer cases and 32 fixed golden tokens")
