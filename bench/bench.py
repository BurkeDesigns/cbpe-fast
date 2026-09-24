"""Speed: cbpe_fast vs an HF `tokenizers` tokenizer (whatever tokenizers version is installed),
on the same texts, one text per call and whole-set batches.

    python bench/bench.py --cbpe tok.json --texts texts.json [--hf HuggingFaceTB/SmolLM2-135M]

--texts: a JSON list of strings, or a dict of named lists (each list is reported separately).
--hf:    a tokenizer.json path or a Hugging Face model id (needs huggingface_hub to download).
Per text: "cold" = median over the first pass on a freshly loaded tokenizer, "warm" = median of
the best of three passes. Batch = MB/s for the whole list in one call. Results are printed and
written to bench/results_<tokenizers version>.json.

Note tokens per second is not comparable across tokenizers (a CBPE token carries ~1.5-1.7x more
text); compare per text or per byte.
"""
import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))


def _gpt2_byte_map():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def load_hf(spec):
    """tokenizers 1.0.0-rc.2 refuses a byte-level BPE whose vocabulary lacks any of the 256 byte
    atoms (SmolLM2 lacks 21, most of which never occur in UTF-8). Adding them past the end of the
    vocabulary changes no id for text without those bytes."""
    from tokenizers import Tokenizer
    path = spec
    if not os.path.exists(spec):
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(spec, "tokenizer.json")
    try:
        tk = Tokenizer.from_file(path)
    except Exception as e:                                    # noqa: BLE001
        if "Byte atom" not in str(e):
            raise
        d = json.load(open(path))
        vocab = d["model"]["vocab"]
        nxt = max(vocab.values()) + 1
        for ch in _gpt2_byte_map().values():
            if ch not in vocab:
                vocab[ch] = nxt
                nxt += 1
        fixed = os.path.join(HERE, "_bytefix_tokenizer.json")
        json.dump(d, open(fixed, "w"))
        tk = Tokenizer.from_file(fixed)
    if hasattr(tk, "no_truncation"):          # 0.x API
        tk.no_truncation()
        tk.no_padding()
    else:                                      # 1.0 API: properties, None switches off
        tk.truncation = None
        tk.padding = None
    return tk


def single(fn, texts, reps=3):
    first, best = None, None
    for _ in range(reps):
        per = []
        for t in texts:
            s = time.perf_counter()
            fn(t)
            per.append(time.perf_counter() - s)
        first = first or per
        best = per if best is None or statistics.median(per) < statistics.median(best) else best
    return statistics.median(first) * 1e6, statistics.median(best) * 1e6


def batch(fn, texts, reps=3):
    b = []
    for _ in range(reps):
        s = time.perf_counter()
        fn(texts)
        b.append(time.perf_counter() - s)
    return sum(map(len, texts)) / min(b) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cbpe", required=True, help="Combinatorial BPE tokenizer JSON")
    ap.add_argument("--texts", required=True)
    ap.add_argument("--hf", default="", help="HF tokenizer to compare against (path or model id)")
    a = ap.parse_args()
    sets = json.load(open(a.texts))
    if isinstance(sets, list):
        sets = {"texts": sets}
    from cbpe_fast import FastCBPE
    systems = []
    ver = "none"
    if a.hf:
        import tokenizers
        ver = tokenizers.__version__
        tk = load_hf(a.hf)
        systems.append((f"HF tokenizers {ver} (.ids)", lambda t: tk.encode(t, add_special_tokens=False).ids,
                        lambda ts: tk.encode_batch(ts, add_special_tokens=False),
                        lambda t: len(tk.encode(t, add_special_tokens=False).ids), None))
    fast = FastCBPE(a.cbpe)
    systems.append(("cbpe_fast (int32 array)", fast.encode, fast.encode_batch, lambda t: len(fast.encode(t)),
                    fast.clear_cache))
    rows = []
    for name, one, many, count, reset in systems:
        for sname, texts in sets.items():
            if reset:
                reset()
            cold, warm = single(one, texts)
            row = {"system": name, "set": sname, "cold_us": cold, "warm_us": warm,
                   "batch_mb_s": batch(many, texts), "tokens": sum(map(count, texts)) / len(texts),
                   "chars": sum(map(len, texts)) / len(texts)}
            rows.append(row)
            print(f"{name:32s} {sname:14s} cold {cold:8.2f} us  warm {warm:8.2f} us  batch {row['batch_mb_s']:8.1f} MB/s"
                  f"  {row['tokens']:7.1f} tok/text  ({row['chars']:.0f} chars)", flush=True)
    json.dump(rows, open(os.path.join(HERE, f"results_{ver}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
