"""Prepared metadata from verified official Engram code, using a tiny tokenizer.

The tokenizer's single-token decoded/raw strings are explicit test inputs, not
the released tokenizer. Its normalization, prime layout and NumPy multipliers
are computed by the unmodified reference and exported verbatim for Go.
"""
from types import SimpleNamespace


class TinyTokenizer:
    decoded = [" The", "the", "<pad>", "THE", "caf\u00e9", "CAFE", " ", "\t",
               "", "\ufffd", "\ufffd", "\uff21", "a", "\u00c5", "\u00e5", "\n\t",
               "foo", "bar", "BAZ", "baz", "hello", "world", "1", "2",
               "3", "4", "5", "6", "7", "8", "9", "!"]

    def __init__(self):
        self.backend_tokenizer = self

    def __len__(self):
        return len(self.decoded)

    def decode(self, ids, skip_special_tokens=False):
        assert len(ids) == 1 and not skip_special_tokens
        return self.decoded[ids[0]]

    def id_to_token(self, token_id):
        return f"raw-byte-{token_id}"


def prepare(sources):
    namespace = {"__name__": __name__}
    exec(compile(sources["engram.py"], "pinned-engram.py", "exec"), namespace)
    tokenizer = TinyTokenizer()
    token_map, size = namespace["build_compressed_token_map"](tokenizer)
    settings = dict(engram_layer_ids=(0, 3, 6), engram_max_ngram_size=4,
                    engram_n_heads=2, engram_head_dim=3, engram_vocab_size=11,
                    engram_num_embeddings=(1, 1, 1), engram_pad_id=2,
                    engram_compressed_vocab_size=size)
    layout = namespace["EngramLayout"].from_args(SimpleNamespace(**settings))
    settings["engram_num_embeddings"] = tuple(sum(p for sizes in layer for p in sizes) + 3
                                               for layer in layout.primes)
    args = SimpleNamespace(**settings, max_batch_size=2, max_seq_len=32)
    layout = namespace["EngramLayout"].from_args(args)
    hasher = namespace["NgramHashState"](args, layout, tokenizer)
    config = dict(layers=list(layout.layer_ids), max_ngram=layout.max_ngram_size,
                  heads=layout.n_heads, head_dim=layout.head_dim,
                  rows=list(layout.num_embeddings),
                  primes=[[p for sizes in layer for p in sizes] for layer in layout.primes],
                  multipliers=hasher.multipliers.tolist(), token_map=token_map,
                  compressed_vocab_size=size, pad_id=args.engram_pad_id)
    return namespace, tokenizer, settings, config
