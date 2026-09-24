"""Unicode tables for the native encoder, generated from the SAME `regex` module and Python
string case mappings the reference implementation uses, so every code point is classified and
case-folded exactly as the reference does:

    cls[cp] bit 0  [\\p{L}\\p{M}]      (a word char in COMB_PAT)
            bit 1  \\p{N}              (a digit-run char)
            bit 2  \\s                 (excluded from the suffix run)
            bit 3  \\p{Ll}   bit 4  \\p{Lu}   (the camelCase split)
    fold    cp -> lower cp for every char the reference's normalise_char() folds: c.lower() is
            one char, differs from c, and upper()s back to exactly c

Building takes about 0.2 s. Results are cached per regex version / Unicode version / Python
version in $CBPE_FAST_CACHE, else $XDG_CACHE_HOME/cbpe-fast, else ~/.cache/cbpe-fast.
"""
import os
import sys
import unicodedata

import numpy as np
import regex

N = 0x110000


def _key():
    return (f"regex{regex.__version__}_uni{unicodedata.unidata_version}"
            f"_py{sys.version_info[0]}{sys.version_info[1]}")


def _cache_dir():
    d = os.environ.get("CBPE_FAST_CACHE")
    if not d:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        d = os.path.join(base, "cbpe-fast")
    return d


def build():
    cps = np.array([c for c in range(N) if not 0xD800 <= c <= 0xDFFF], dtype=np.int64)
    chars = "".join(map(chr, cps.tolist()))
    cls = np.zeros(N, dtype=np.uint8)
    for bit, pat in ((1, r"[\p{L}\p{M}]+"), (2, r"\p{N}+"), (4, r"\s+"), (8, r"\p{Ll}+"), (16, r"\p{Lu}+")):
        for m in regex.finditer(pat, chars):
            cls[cps[m.start():m.end()]] |= bit
    src, dst = [], []
    for c in cps.tolist():
        ch = chr(c)
        lo = ch.lower()
        if lo != ch and len(lo) == 1 and lo.upper() == ch:
            src.append(c)
            dst.append(ord(lo))
    return cls, np.array(src, dtype=np.uint32), np.array(dst, dtype=np.uint32)


def load():
    """-> (cls uint8[0x110000], fold_src uint32[], fold_dst uint32[]), cached on disk."""
    path = os.path.join(_cache_dir(), f"tables_{_key()}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["cls"], z["fold_src"], z["fold_dst"]
    cls, src, dst = build()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + f".{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, cls=cls, fold_src=src, fold_dst=dst)
        os.replace(tmp, path)
    except OSError:
        pass                                      # read-only home: rebuild next time, still correct
    return cls, src, dst
