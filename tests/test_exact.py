"""cbpe_fast must reproduce the reference implementation token for token.

    pip install -e ".[test]"                 # installs the reference (package `cbpe`)
    python tests/test_exact.py [--texts texts.json] [tokenizer.json ...]
    pytest tests/                            # the same checks on the built-in text set

The reference comes from an installed `cbpe` package or from a checkout named by $CBPE_REPO.
Tokenizers checked: tests/data/cbpe_fc_16k.json (camelCase splitting), every *comb*.json in the
checkout's pretrained/ (English, a 6-language mix, and Chinese with the Traditional variation)
when $CBPE_REPO is set, and any given on the command line. Texts: random Unicode fuzz (every
character class the pretokenizer distinguishes, case-mapping edge cases, camelCase, astral
chars, all whitespace kinds), multi-byte characters of every class placed across the 64-byte
SIMD block edges, and optionally a JSON list or dict of lists (--texts). For each tokenizer it
checks encode, encode_tuples, encode_batch and decode(encode(x)) == x.
Set CBPE_NO_SIMD=1 to test the scalar scanner instead of the AVX-512 one.
"""
import glob
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

REPO = os.environ.get("CBPE_REPO", "")
if REPO:
    sys.path.insert(0, REPO)

ALPH = ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" "  \n\t\r\x0b\x0c\x85\xa0 　"
        ".,;:!?'\"()[]{}<>-_=+*/\\|@#$%^&~`" "éÉàÀüÜñÑçÇøØåÅßẞİıǅǆǄΣσςΩωЖжЯяİ"
        "́̈ःि" "中文漢字臺灣台湾東京ひらがなカタカナ한국어" "٠١٢٣०१२३" "ⅠⅡⅫ½²³"
        "😀🚀𝔘𝔫𝔦𝔠𝔬𝔡𝔢" "ﬁﬂ" "​‍﻿")
WORDS = ["getUserName", "HTTPServer", "XMLHttpRequest", "iPhone", "McDonald", "ABCdef", "aBC",
         "ПРИВЕТ", "Привет", "ΣΟΦΙΑ", "straße", "STRASSE", "İstanbul", "ǅemal", "naïve", "café"]


def fuzz(n, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        parts = []
        for _ in range(rng.randint(1, 30)):
            k = rng.random()
            if k < 0.3:
                parts.append(rng.choice(WORDS))
            elif k < 0.8:
                parts.append("".join(rng.choice(ALPH) for _ in range(rng.randint(1, 12))))
            else:
                parts.append(chr(rng.randint(0x20, 0x2FFFF)) if rng.random() < 0.5 else chr(rng.randint(0x80, 0x3000)))
        out.append("".join(parts))
    return [t for t in out if not any(0xD800 <= ord(c) <= 0xDFFF for c in t)]


def block_edges():
    """Multi-byte chars of every class placed across the 64-byte SIMD block boundaries."""
    chars = ["é", "Ж", "́", "٣", "\xa0", "　", "—", "“", "中", "😀", "𝔘", "½", " "]
    fills = ["a", " ", ".", "7", "a.", " (", "Ab"]
    out = []
    for ch in chars:
        for f in fills:
            for k in range(58, 72):
                pre = (f * 80)[:k]
                out.append(pre + ch + "b" + ch + " x.")
                out.append(pre + ch * 3 + "Word, end")
    return out


def tokenizers(extra=()):
    paths = [os.path.join(HERE, "data", "cbpe_fc_16k.json")]
    if REPO:
        paths += sorted(glob.glob(os.path.join(REPO, "pretrained", "*comb*.json")))
    return paths + list(extra)


def check(path, texts, verbose=True):
    """-> number of mismatching texts for one tokenizer file"""
    from cbpe import load as ref_load
    from cbpe_fast import FastCBPE
    ref, fast = ref_load(path), FastCBPE(path)
    bad = ntok = 0
    for t in texts:
        r = [tuple(x) for x in ref.encode(t)]
        f = [tuple(x) for x in fast.encode(t).tolist()]
        ntok += len(r)
        if r != f:
            bad += 1
            if verbose and bad <= 3:
                k = next(i for i, (a, b) in enumerate(zip(r + [None], f + [None])) if a != b)
                print(f"  MISMATCH {os.path.basename(path)} text={t[:60]!r} at token {k}: ref {r[k:k+3]} fast {f[k:k+3]}")
        elif fast.decode(fast.encode(t)) != t:
            bad += 1
            if verbose:
                print(f"  DECODE MISMATCH {t[:60]!r}")
    if texts and fast.encode_tuples(texts[0]) != [tuple(x) for x in ref.encode(texts[0])]:
        bad += 1
    batch = fast.encode_batch(texts[:500])
    bad += sum(b.tolist() != [list(x) for x in ref.encode(t)] for b, t in zip(batch, texts[:500]))
    if verbose:
        print(f"{'OK ' if bad == 0 else 'BAD'} {os.path.basename(path)}: {len(texts)} texts, {ntok} tokens, "
              f"{bad} mismatches (simd={fast.simd})")
    return bad


def _reference_available():
    try:
        import cbpe  # noqa: F401
        return True
    except ImportError:
        return False


def test_exact():
    import pytest
    if not _reference_available():
        pytest.skip("reference not installed: pip install -e '.[test]' or set CBPE_REPO")
    texts = fuzz(1500) + block_edges()
    for p in tokenizers():
        assert check(p, texts, verbose=False) == 0, p


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", default="")
    ap.add_argument("tokenizers", nargs="*")
    a = ap.parse_args()
    if not _reference_available():
        sys.exit("reference not installed: pip install -e '.[test]' or set CBPE_REPO")
    texts = fuzz(3000) + block_edges()
    if a.texts:
        extra = json.load(open(a.texts))
        texts += [t for v in (extra.values() if isinstance(extra, dict) else [extra]) for t in v]
    bad = sum(check(p, texts) for p in tokenizers(a.tokenizers))
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
