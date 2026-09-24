"""cbpe_fast: a native encoder/decoder for Combinatorial BPE tokenizers
(github.com/SwayStar123/CombinatorialBPE), producing exactly the reference's tokens.

    from cbpe_fast import FastCBPE
    tok = FastCBPE("wiki_en_16384_comb.json")   # any file CombinatorialBPE.save() wrote
    ids = tok.encode(text)                      # (n, 4) int32: variation, prefix, core, suffix
    tok.encode_tuples(text)                     # the reference's list of 4-tuples
    tok.encode_batch(texts)                     # one array per text; all cores, GIL released
    tok.decode(ids) == text
"""
import json
import os

import numpy as np

from . import _native
from .tables import load as _load_tables

__all__ = ["FastCBPE", "load", "VARIATIONS"]
__version__ = "0.1.0"

VARIATIONS = ["as-is", "Capitalised", "UPPER", "Traditional"]
V_NONE, V_CAP, V_UPPER, V_TRAD = 0, 1, 2, 3


class FastCBPE:
    """Loads a Combinatorial BPE tokenizer JSON file.

    han_tables: the reference's cbpe/data/han_st.json, needed only for tokenizers trained with
    fold_han=True (Traditional/Simplified Chinese). Defaults to $CBPE_HAN_TABLES, then to the
    copy inside an installed `cbpe` package.
    cache_limit: most units kept in the encode cache (per worker for encode_batch).
    """

    def __init__(self, path, han_tables=None, cache_limit=2_000_000):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("type") != "combinatorial":
            raise ValueError(f"{path} is not a Combinatorial BPE tokenizer")
        alphabet = list(d["alphabet"])
        merges = [tuple(m) for m in d["merges"]]
        vocab = [bytes([b]) for b in range(256)] + alphabet + [a + b for a, b in merges]
        tok2id = {}
        for i, t in enumerate(vocab):
            if isinstance(t, str):
                tok2id.setdefault(t, i)             # a re-created string keeps its first id
        chars = sorted((ord(t), i) for t, i in tok2id.items() if len(t) == 1)
        ma, mb, mr, mo = [], [], [], []
        for r, (a, b) in enumerate(merges):
            ma.append(tok2id[a])
            mb.append(tok2id[b])
            mr.append(r)
            mo.append(tok2id[a + b])
        self.prefixes, self.suffixes = list(d["prefixes"]), list(d["suffixes"])
        self.fold_case = d.get("fold_case", True)
        self.fold_han = d.get("fold_han", False)
        self.split_camel = d.get("split_camel", False)
        self.punct_to_next = d.get("punct_to_next", False)
        han_fold, to_trad = {}, {}
        if self.fold_han:
            han_tables = han_tables or os.environ.get("CBPE_HAN_TABLES") or _installed_han_tables()
            if not han_tables:
                raise ValueError("fold_han tokenizer: pass han_tables=<cbpe/data/han_st.json>")
            with open(han_tables, encoding="utf-8") as f:
                h = json.load(f)
            han_fold, to_trad = h["fold"], h["to_trad"]

        def variation(v, s):
            if v == V_CAP:
                return s[0].upper() + s[1:]
            if v == V_UPPER:
                return s.upper()
            if v == V_TRAD:
                return "".join(to_trad.get(c, c) for c in s)
            return s

        dec_core = [[t if isinstance(t, bytes) else variation(v, t).encode("utf-8") for t in vocab]
                    for v in range(4)]
        cls, fsrc, fdst = _load_tables()
        self.sizes = {"variation": 4 if self.fold_han else (3 if self.fold_case else 1),
                      "prefix": len(self.prefixes), "core": len(vocab), "suffix": len(self.suffixes)}
        self.native = _native.Native(
            cls.tobytes(), fsrc.tolist(), fdst.tolist(), [c for c, _ in chars], [i for _, i in chars],
            ma, mb, mr, mo,
            [p.encode("utf-8") for p in self.prefixes], [s.encode("utf-8") for s in self.suffixes],
            max(map(len, self.prefixes)), max(map(len, self.suffixes)),
            self.fold_case, self.fold_han, self.split_camel, self.punct_to_next,
            [ord(k) for k in han_fold], [ord(v) for v in han_fold.values()], [ord(k) for k in to_trad],
            dec_core, cache_limit)

    @property
    def vocab_size(self):
        """Embedding rows the tokenizer needs (sum of the four tables), as in the reference."""
        return sum(self.sizes.values())

    def encode(self, text: str) -> np.ndarray:
        return self.native.encode(text)

    def encode_tuples(self, text: str) -> list:
        return self.native.encode_tuples(text)

    def encode_batch(self, texts) -> list:
        return self.native.encode_batch(list(texts))

    def encode_batch_flat(self, texts):
        """-> ((total, 4) int32 ids, (len(texts) + 1) int64 offsets)"""
        return self.native.encode_batch_flat(list(texts))

    def decode(self, ids) -> str:
        ids = np.ascontiguousarray(np.asarray(ids, dtype=np.int32).reshape(-1, 4))
        return self.native.decode(ids).decode("utf-8", errors="replace")

    def clear_cache(self):
        self.native.clear_cache()

    @property
    def simd(self) -> bool:
        """True when the AVX-512 unit scanner is in use (else the scalar one; same output)."""
        return self.native.simd()


def load(path, **kw) -> FastCBPE:
    return FastCBPE(path, **kw)


def _installed_han_tables():
    try:
        import cbpe
    except ImportError:
        return None
    p = os.path.join(os.path.dirname(cbpe.__file__), "data", "han_st.json")
    return p if os.path.exists(p) else None
