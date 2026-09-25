# Fast Combinatorial BPE

A native encoder and decoder for [Combinatorial BPE](https://github.com/SwayStar123/CombinatorialBPE)
tokenizers. Its output is identical to the Python reference, token for token, and it is 3-4.5x
faster per text than HF `tokenizers` 1.0.0-rc.2 on the same texts.

Combinatorial BPE factors each token into (variation, prefix, core, suffix). `"Hello"`,
`" hello"` and `"HELLO,"` share one core. At the same number of embedding rows, its authors
measured 12-28% fewer tokens than standard BPE on natural-language corpora; on tool-calling
prompts we measured 35-38% fewer. The reference encoder is pure Python (about
150 us for a 1.6 KB prompt). This one is written in Rust and handles the same text in
1.5 us.

## Install

Needs a Rust toolchain ([rustup.rs](https://rustup.rs)).

```bash
pip install .                 # builds the extension with maturin
pip install -e ".[test]"      # plus the reference implementation, for the exactness tests
./build.sh                    # or, without maturin: cargo build -> python/cbpe_fast/_native.so
```

## Use

```python
from cbpe_fast import FastCBPE

tok = FastCBPE("wiki_en_16384_comb.json")   # any file CombinatorialBPE.save() wrote
ids = tok.encode(text)                      # (n, 4) int32 array: variation, prefix, core, suffix
tok.decode(ids) == text                     # True
tok.encode_tuples(text)                     # the reference's return type: a list of 4-tuples
tok.encode_batch(texts)                     # one array per text, all cores, GIL released
tok.encode_batch_flat(texts)                # (total, 4) array + offsets
```

Tokenizers are trained with the reference (`CombinatorialBPE.train`); this package only encodes
and decodes. Tokenizers trained with `fold_han=True` also need the reference's
`cbpe/data/han_st.json`. It is found automatically when the `cbpe` package is installed;
otherwise pass `han_tables=` or set `$CBPE_HAN_TABLES`. `tests/data/cbpe_fc_16k.json` is a
16k tokenizer (with camelCase splitting) trained on tool-calling prompts.

## Speed

Ryzen 9 9950X, Python 3.12, `bench/bench.py`. Each text is encoded in its own call; the table
shows the median of the best of three passes, and batch is the whole set in one call. The HF
side is SmolLM2's 49k BPE; the CBPE side is `tests/data/cbpe_fc_16k.json` for the tool-calling
rows and the reference's `wiki_en_16384_comb.json` for Wikipedia and TinyStories.

| per text | tokenizers 0.23.2 | tokenizers 1.0.0-rc.2 | CBPE reference (Python) | **cbpe-fast** |
|---|---|---|---|---|
| tool-calling prompt, 1.6 KB | 163 us | 5.87 us | 151 us | **1.48 us** |
| bare user query, 130 B | 15.1 us | 1.12 us | 10.5 us | **0.25 us** |
| JSON tool schemas, 2 KB | 169 us | 5.55 us | 113 us | **1.34 us** |
| long document, 3 KB | 282 us | 7.83 us | 194 us | **2.33 us** |
| Wikipedia paragraph | 61 us | 2.84 us | - | **0.78 us** |
| TinyStories story | 127 us | 3.83 us | - | **1.33 us** |
| first pass (cold cache), prompt | 173 us | 7.05 us | - | **3.67 us** |
| batch, prompts | 50 MB/s | 1,505 MB/s | - | **2,556 MB/s** |
| batch, long documents | 47 MB/s | 2,404 MB/s | - | **4,737 MB/s** |
| batch, Wikipedia paragraphs | 50 MB/s | 569 MB/s | - | **773 MB/s** |
| decode one prompt | - | 4.90 us | 35.5 us | **2.95 us** |

- The rc.2 figures include building its `.ids` Python list. `encode()` alone is 3.56 us, so
  cbpe-fast is still 2.4x faster by that measure.
- The same prompt is 285 CBPE tokens against 438 SmolLM2 tokens. Per output token cbpe-fast is
  still 3x faster (4.7 vs 14.1 ns). Tokens per second is the wrong unit for comparing
  tokenizers, because a CBPE token carries about 1.6x more text; compare per text or per byte.
- rc.2 refuses to load SmolLM2's `tokenizer.json` ("Byte atom `0x04` not found in the
  vocabulary"). `bench.py` adds the 21 missing byte atoms past the end of the vocabulary,
  which changes no id for real text.
- CPUs without AVX-512 VBMI use the scalar scanner (force it with `CBPE_NO_SIMD=1`). Its output
  is the same, at 2.88 us per prompt, 0.40 us per query and 3.75 us per long document, still
  1.6-2.8x faster than rc.2.

## Exactness

```bash
CBPE_REPO=<CombinatorialBPE checkout> python tests/test_exact.py [--texts texts.json] [tokenizer.json ...]
```

This checks every tokenizer in the reference's `pretrained/` (English, a 6-language mix, and
Chinese with the Traditional/Simplified variation) plus `tests/data/cbpe_fc_16k.json`. The
texts are Unicode fuzz, multi-byte characters of every class placed across the 64-byte SIMD
block edges, and any `--texts` file. For each tokenizer it checks `encode`, `encode_tuples`,
`encode_batch` and `decode(encode(x)) == x`. The last full run covered 36,647 texts
(tool-calling prompts, JSON schemas, Wikipedia, TinyStories, fuzz) and 5.4-9.4M tokens per
tokenizer: 0 mismatches for both the AVX-512 and scalar scanners. Separate runs over 18.8 MB
of Wikipedia and 8 MB of binary noise also matched.

## How it works

1. **Unit boundaries by SIMD.** The reference splits text with
   `([^LMN]*)([LM]+|N+)([^\s LMN]*)|([^LMN]+)`, where L is letters, M marks and N digits.
   Each 64-byte block is classified with one AVX-512 VBMI byte-table lookup into letter,
   digit, space and other masks; blocks holding non-ASCII bytes get a per-character fill. A
   run of "other" characters right after a letter or digit is the unit's suffix, and one
   carry-propagating add finds all such runs in the block. A new unit starts at every
   character whose predecessor is a letter, digit or suffix and that is a space, a letter not
   after a letter, or a digit not after a digit. There are no per-byte branches, and scanning
   is about 3% of the time.
2. **A unit cache.** A unit's tokens depend only on its text, so they are cached by the unit's
   bytes, in an open-addressing table with 32-byte slots. The common case (a key of 16 bytes
   or less and one token with 16-bit ids) fits entirely in its slot, so a hit is one cache-line
   read, and hashing uses a single masked 16-byte load.
3. **Two phases per text.** The first phase records every unit and prefetches its slot; the
   second probes. Cache misses to L3 overlap instead of queueing. With the inline slots, this
   took the per-unit cost from 11.7 ns (a hash map of boxed keys and values) to about 5 ns.
4. **Cache misses** run a direct port of the reference: longest-affix matching, the camelCase
   split, round-trip case folding, and rank-ordered char BPE with UTF-8 byte fallback.
5. **Output** is the token Vec itself, reinterpreted as an int32 numpy array without a copy.
   Batches read the Python strings in place, release the GIL, and run on rayon with one
   cache per worker.

The Unicode classes (`\p{L}\p{M}`, `\p{N}`, `\s`, `\p{Ll}`, `\p{Lu}`) and case folds come
from `cbpe_fast/tables.py`. It generates them from the same `regex` module and
`str.lower`/`upper` the reference uses, so classification cannot drift from the reference, and
caches them per regex/Unicode/Python version in `~/.cache/cbpe-fast`.

## Layout

```
src/lib.rs                 the encoder/decoder (PyO3 extension cbpe_fast._native)
python/cbpe_fast/          the Python API (FastCBPE) and the Unicode table generator
tests/test_exact.py        token-for-token comparison with the reference
tests/data/                a 16k tokenizer trained on tool-calling prompts
bench/bench.py             speed against HF tokenizers
```

## Credits

Combinatorial BPE, its training code and its file format are by the
[Combinatorial BPE contributors](https://github.com/SwayStar123/CombinatorialBPE) (MIT). This
package reimplements their encoder and decoder; see LICENSE.
