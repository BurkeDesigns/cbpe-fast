//! Native Combinatorial BPE encoder, output-identical to the Python reference
//! (github.com/SwayStar123/CombinatorialBPE: cbpe/tokenizers.py `CombinatorialBPE.encode`,
//! cbpe/bpe.py `CharBPE`). Built as the `cbpe_fast._native` extension module.
//!
//! Text is cut into units exactly as COMB_PAT does:
//!     unit = [^LMN]*  ( [LM]+ | N+ )  [^\s LMN]*   |   [^LMN]+   (trailing junk)
//! and every unit becomes (variation, prefix, core, suffix) tokens. A unit's tokens depend on
//! its text alone, so they are cached by the unit's bytes: after warm-up almost every unit is
//! one hash probe and a copy. Unicode classes and case folds come from tables the Python
//! wrapper generates with the reference's own `regex` module and str case mappings.

use numpy::ndarray::Array2;
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyString, PyTuple};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::Mutex;

const C_WORD: u8 = 1; // \p{L} | \p{M}
const C_NUM: u8 = 2; // \p{N}
const C_WS: u8 = 4; // \s
const C_LL: u8 = 8;
const C_LU: u8 = 16;
const OOA: u32 = 0x8000_0000; // symbol flag: char outside the core alphabet (low bits = code point)

const F_UP: u8 = 1;
const F_TRAD: u8 = 2;
const V_NONE: u32 = 0;
const V_CAP: u32 = 1;
const V_UPPER: u32 = 2;
const V_TRAD: u32 = 3;

type Tok = [u32; 4];

#[inline(always)]
fn decode_at(s: &[u8], i: usize) -> (u32, usize) {
    let b0 = s[i] as u32;
    if b0 < 0x80 {
        (b0, 1)
    } else if b0 < 0xE0 {
        (((b0 & 0x1F) << 6) | (s[i + 1] as u32 & 0x3F), 2)
    } else if b0 < 0xF0 {
        (((b0 & 0x0F) << 12) | ((s[i + 1] as u32 & 0x3F) << 6) | (s[i + 2] as u32 & 0x3F), 3)
    } else {
        (
            ((b0 & 0x07) << 18)
                | ((s[i + 1] as u32 & 0x3F) << 12)
                | ((s[i + 2] as u32 & 0x3F) << 6)
                | (s[i + 3] as u32 & 0x3F),
            4,
        )
    }
}

#[inline(always)]
fn push_utf8(cp: u32, out: &mut Vec<Tok>) {
    let mut buf = [0u8; 4];
    let s = char::from_u32(cp).unwrap_or('\u{FFFD}').encode_utf8(&mut buf);
    for &b in s.as_bytes() {
        out.push([V_NONE, 0, b as u32, 0]);
    }
}

struct Model {
    cls: Vec<u8>,
    fold: FxHashMap<u32, u32>,
    han_fold: FxHashMap<u32, u32>,
    han_to_trad: FxHashSet<u32>,
    ascii_id: [u32; 128],
    char_id: FxHashMap<u32, u32>,
    merges: FxHashMap<u64, (u32, u32)>,
    prefixes: FxHashMap<Vec<u8>, u32>,
    suffixes: FxHashMap<Vec<u8>, u32>,
    max_p: usize,
    max_s: usize,
    fold_case: bool,
    fold_han: bool,
    split_camel: bool,
    punct_to_next: bool,
    simd: bool,
    // decode: bytes of prefix/suffix ids, and of every (variation, core) pair
    dec_prefix: Vec<Vec<u8>>,
    dec_suffix: Vec<Vec<u8>>,
    dec_core: Vec<Vec<Vec<u8>>>,
}

impl Model {
    #[inline(always)]
    fn class_at(&self, s: &[u8], i: usize) -> (u8, usize) {
        let b = s[i];
        if b < 0x80 {
            (self.cls[b as usize], 1)
        } else {
            let (cp, l) = decode_at(s, i);
            (self.cls[cp as usize], l)
        }
    }

    #[inline(always)]
    fn sym_of(&self, cp: u32) -> u32 {
        if cp < 128 {
            let id = self.ascii_id[cp as usize];
            if id != u32::MAX {
                return id;
            }
            return OOA | cp;
        }
        match self.char_id.get(&cp) {
            Some(&id) => id,
            None => OOA | cp,
        }
    }

    /// CharBPE.split: repeatedly merge every occurrence of the lowest-ranked adjacent pair.
    /// `lens` (optional) tracks how many chars each symbol covers.
    fn bpe(&self, syms: &mut Vec<u32>, mut lens: Option<&mut Vec<u32>>) {
        while syms.len() > 1 {
            let mut best = u32::MAX;
            let mut pair = 0u64;
            let mut merged = 0u32;
            for w in syms.windows(2) {
                if (w[0] | w[1]) & OOA != 0 {
                    continue;
                }
                let key = ((w[0] as u64) << 32) | w[1] as u64;
                if let Some(&(r, m)) = self.merges.get(&key) {
                    if r < best {
                        best = r;
                        pair = key;
                        merged = m;
                    }
                }
            }
            if best == u32::MAX {
                break;
            }
            let (a, b) = ((pair >> 32) as u32, pair as u32);
            let n = syms.len();
            let (mut i, mut o) = (0usize, 0usize);
            while i < n {
                if i + 1 < n && syms[i] == a && syms[i + 1] == b {
                    syms[o] = merged;
                    if let Some(l) = lens.as_deref_mut() {
                        l[o] = l[i] + l[i + 1];
                    }
                    i += 2;
                } else {
                    syms[o] = syms[i];
                    if let Some(l) = lens.as_deref_mut() {
                        l[o] = l[i];
                    }
                    i += 1;
                }
                o += 1;
            }
            syms.truncate(o);
            if let Some(l) = lens.as_deref_mut() {
                l.truncate(o);
            }
        }
    }

    /// CharBPE.encode on a raw string (no case folding), byte fallback for unknown chars.
    fn core_encode(&self, s: &[u8], out: &mut Vec<Tok>) {
        if s.is_empty() {
            return;
        }
        let mut syms = Vec::with_capacity(s.len());
        let mut i = 0;
        while i < s.len() {
            let (cp, l) = decode_at(s, i);
            syms.push(self.sym_of(cp));
            i += l;
        }
        self.bpe(&mut syms, None);
        for sym in syms {
            if sym & OOA != 0 {
                push_utf8(sym & !OOA, out);
            } else {
                out.push([V_NONE, 0, sym, 0]);
            }
        }
    }

    #[inline(always)]
    fn fold_char(&self, cp: u32) -> (u32, u8) {
        if self.fold_case {
            if (b'A' as u32..=b'Z' as u32).contains(&cp) {
                return (cp + 32, F_UP);
            }
            if cp >= 128 {
                if let Some(&lo) = self.fold.get(&cp) {
                    return (lo, F_UP);
                }
            }
        }
        if self.fold_han {
            if let Some(&s) = self.han_fold.get(&cp) {
                return (s, F_TRAD);
            }
        }
        (cp, 0)
    }

    fn piece_variation(&self, piece: &[u32], flags: &[u8]) -> Option<u32> {
        if flags.iter().all(|&f| f == 0) {
            return Some(V_NONE);
        }
        if flags.iter().all(|&f| f != F_TRAD) {
            let up0 = flags[0] == F_UP;
            if up0 && !flags[1..].iter().any(|&f| f == F_UP) {
                return Some(V_CAP);
            }
            if flags.iter().all(|&f| f == F_UP) {
                return Some(V_UPPER);
            }
            return None;
        }
        if flags.iter().all(|&f| f != F_UP)
            && piece.iter().zip(flags).all(|(c, &f)| f == F_TRAD || !self.han_to_trad.contains(c))
        {
            return Some(V_TRAD);
        }
        None
    }

    /// _encode_segment: fold, BPE over the normalised chars, per piece a variation that
    /// restores the original, else per-char tokens; unknown chars -> bytes of the ORIGINAL.
    fn encode_segment(&self, cps: &[u32], out: &mut Vec<Tok>) {
        if !(self.fold_case || self.fold_han) {
            let mut syms: Vec<u32> = cps.iter().map(|&c| self.sym_of(c)).collect();
            self.bpe(&mut syms, None);
            for sym in syms {
                if sym & OOA != 0 {
                    push_utf8(sym & !OOA, out);
                } else {
                    out.push([V_NONE, 0, sym, 0]);
                }
            }
            return;
        }
        let n = cps.len();
        let mut norm = Vec::with_capacity(n);
        let mut flags = Vec::with_capacity(n);
        for &c in cps {
            let (x, f) = self.fold_char(c);
            norm.push(x);
            flags.push(f);
        }
        let mut syms: Vec<u32> = norm.iter().map(|&c| self.sym_of(c)).collect();
        let mut lens = vec![1u32; n];
        self.bpe(&mut syms, Some(&mut lens));
        let mut pos = 0usize;
        for (sym, len) in syms.iter().zip(lens.iter()) {
            let len = *len as usize;
            if sym & OOA != 0 {
                push_utf8(cps[pos], out);
            } else {
                match self.piece_variation(&norm[pos..pos + len], &flags[pos..pos + len]) {
                    Some(v) => out.push([v, 0, *sym, 0]),
                    None => {
                        for k in pos..pos + len {
                            let ci = self.sym_of(norm[k]);
                            if ci & OOA != 0 {
                                push_utf8(cps[k], out);
                            } else {
                                let v = match flags[k] {
                                    F_UP => V_CAP,
                                    F_TRAD => V_TRAD,
                                    _ => V_NONE,
                                };
                                out.push([v, 0, ci, 0]);
                            }
                        }
                    }
                }
            }
            pos += len;
        }
    }

    /// CAMEL_SPLIT: (?<=Ll)(?=Lu) | (?<=Lu)(?=Lu Ll)
    fn encode_word(&self, word: &[u8], out: &mut Vec<Tok>) {
        let mut cps = Vec::with_capacity(word.len());
        let mut i = 0;
        while i < word.len() {
            let (cp, l) = decode_at(word, i);
            cps.push(cp);
            i += l;
        }
        if !self.split_camel {
            self.encode_segment(&cps, out);
            return;
        }
        let cl: Vec<u8> = cps.iter().map(|&c| self.cls[c as usize]).collect();
        let n = cps.len();
        let mut start = 0usize;
        for i in 1..n {
            let split = (cl[i - 1] & C_LL != 0 && cl[i] & C_LU != 0)
                || (cl[i - 1] & C_LU != 0 && cl[i] & C_LU != 0 && i + 1 < n && cl[i + 1] & C_LL != 0);
            if split {
                self.encode_segment(&cps[start..i], out);
                start = i;
            }
        }
        self.encode_segment(&cps[start..], out);
    }

    fn char_starts(s: &[u8]) -> Vec<usize> {
        let mut v = Vec::with_capacity(s.len() + 1);
        let mut i = 0;
        while i < s.len() {
            v.push(i);
            let b = s[i];
            i += if b < 0x80 { 1 } else if b < 0xE0 { 2 } else if b < 0xF0 { 3 } else { 4 };
        }
        v.push(s.len());
        v
    }

    /// longest_suffix_in(pre, p2id, max_p) -> (id, byte length of the uncovered head)
    fn longest_prefix_affix(&self, pre: &[u8]) -> (u32, usize) {
        if pre.is_empty() {
            return (0, 0);
        }
        let st = Self::char_starts(pre);
        let nch = st.len() - 1;
        for k in (1..=nch.min(self.max_p)).rev() {
            let from = st[nch - k];
            if let Some(&id) = self.prefixes.get(&pre[from..]) {
                return (id, from);
            }
        }
        (0, pre.len())
    }

    /// longest_prefix_in(suf, s2id, max_s) -> (id, byte offset where the uncovered tail starts)
    fn longest_suffix_affix(&self, suf: &[u8]) -> (u32, usize) {
        if suf.is_empty() {
            return (0, 0);
        }
        let st = Self::char_starts(suf);
        let nch = st.len() - 1;
        for k in (1..=nch.min(self.max_s)).rev() {
            let to = st[k];
            if let Some(&id) = self.suffixes.get(&suf[..to]) {
                return (id, to);
            }
        }
        (0, 0)
    }

    /// encode_unit for the unit s[a..d]: a..b prefix run, b..c word, c..d suffix run;
    /// junk units (no word) are core-encoded whole.
    fn encode_unit(&self, s: &[u8], a: usize, b: usize, c: usize, d: usize, junk: bool, out: &mut Vec<Tok>) {
        if junk {
            self.core_encode(&s[a..d], out);
            return;
        }
        let pre = &s[a..b];
        let suf = &s[c..d];
        let (p, head) = self.longest_prefix_affix(pre);
        let (sx, tail) = self.longest_suffix_affix(suf);
        self.core_encode(&pre[..head], out);
        let body = out.len();
        self.encode_word(&s[b..c], out);
        out[body][1] = p;
        let last = out.len() - 1;
        out[last][3] = sx;
        self.core_encode(&suf[tail..], out);
    }
}

fn simd_ok() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if std::env::var_os("CBPE_NO_SIMD").is_some() {
            return false;
        }
        return is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx512vbmi");
    }
    #[allow(unreachable_code)]
    false
}

/// Unit cache: open addressing, linear probing, 32-byte slots. The common case (a key of at
/// most 16 bytes whose unit is one token with 16-bit ids) lives entirely in its slot, so a hit
/// is one cache-line read, no pointer chase. Longer keys / multi-token units spill to arenas.
#[derive(Clone, Copy, Default)]
#[repr(C, align(32))]
struct Slot {
    tag: u32,       // high hash bits | 1; 0 = empty
    klen: u16,      // key bytes
    ntok: u16,      // tokens; bit 15 set = tokens in the arena
    key: [u8; 16],  // key, zero-padded (klen <= 16), else [0..4] = key arena offset
    tok: [u16; 4],  // the one token (inline), else [0..2] = token arena offset
}

const ARENA: u16 = 0x8000;

struct Cache {
    slots: Vec<Slot>,
    mask: usize,
    len: usize,
    limit: usize,
    keys: Vec<u8>,
    toks: Vec<Tok>,
}

#[inline(always)]
fn mum(a: u64, b: u64) -> u64 {
    let r = (a as u128).wrapping_mul(b as u128);
    (r as u64) ^ ((r >> 64) as u64)
}

const H1: u64 = 0xa076_1d64_78bd_642f;
const H2: u64 = 0xe703_7ed1_a0b4_28db;
const H3: u64 = 0x8ebc_6af0_9c88_c6e3;

/// first 16 bytes of the key, zero padded (safe scalar version)
#[inline(always)]
fn load16(k: &[u8]) -> (u64, u64) {
    let mut buf = [0u8; 16];
    let n = k.len().min(16);
    buf[..n].copy_from_slice(&k[..n]);
    (u64::from_le_bytes(buf[..8].try_into().unwrap()), u64::from_le_bytes(buf[8..].try_into().unwrap()))
}

#[inline(always)]
fn hash_key(k: &[u8], lo: u64, hi: u64) -> u64 {
    let mut h = mum(lo ^ H1, hi ^ H2) ^ (k.len() as u64).wrapping_mul(H3);
    if k.len() > 16 {
        for ch in k[16..].chunks(8) {
            let mut buf = [0u8; 8];
            buf[..ch.len()].copy_from_slice(ch);
            h = mum(h ^ u64::from_le_bytes(buf), H1);
        }
    }
    h
}

impl Cache {
    fn new(limit: usize) -> Self {
        Cache { slots: vec![Slot::default(); 4096], mask: 4095, len: 0, limit, keys: Vec::new(), toks: Vec::new() }
    }

    fn clear(&mut self) {
        *self = Cache::new(self.limit);
    }

    #[inline(always)]
    fn slot_ptr(&self, h: u64) -> *const Slot {
        unsafe { self.slots.as_ptr().add((h as usize) & self.mask) }
    }

    /// on a hit, append the unit's tokens and return true
    #[inline(always)]
    fn get_emit(&self, key: &[u8], lo: u64, hi: u64, h: u64, out: &mut Vec<Tok>) -> bool {
        let tag = ((h >> 32) as u32) | 1;
        let mut i = (h as usize) & self.mask;
        loop {
            let sl = unsafe { self.slots.get_unchecked(i) };
            if sl.tag == 0 {
                return false;
            }
            if sl.tag == tag && sl.klen as usize == key.len() {
                let eq = if key.len() <= 16 {
                    u64::from_le_bytes(sl.key[..8].try_into().unwrap()) == lo
                        && u64::from_le_bytes(sl.key[8..].try_into().unwrap()) == hi
                } else {
                    let off = u32::from_le_bytes(sl.key[..4].try_into().unwrap()) as usize;
                    &self.keys[off..off + key.len()] == key
                };
                if eq {
                    if sl.ntok & ARENA == 0 {
                        let t = sl.tok;
                        out.push([t[0] as u32, t[1] as u32, t[2] as u32, t[3] as u32]);
                    } else {
                        let off = sl.tok[0] as usize | ((sl.tok[1] as usize) << 16);
                        let n = (sl.ntok & !ARENA) as usize;
                        out.extend_from_slice(&self.toks[off..off + n]);
                    }
                    return true;
                }
            }
            i = (i + 1) & self.mask;
        }
    }

    fn place(&mut self, sl: Slot, h: u64) {
        let mut i = (h as usize) & self.mask;
        while self.slots[i].tag != 0 {
            i = (i + 1) & self.mask;
        }
        self.slots[i] = sl;
    }

    fn slot_hash(&self, sl: &Slot) -> u64 {
        let k = sl.klen as usize;
        if k <= 16 {
            let lo = u64::from_le_bytes(sl.key[..8].try_into().unwrap());
            let hi = u64::from_le_bytes(sl.key[8..].try_into().unwrap());
            hash_key(&sl.key[..k], lo, hi)
        } else {
            let off = u32::from_le_bytes(sl.key[..4].try_into().unwrap()) as usize;
            let key = &self.keys[off..off + k];
            let (lo, hi) = load16(key);
            hash_key(key, lo, hi)
        }
    }

    fn grow(&mut self) {
        let old = std::mem::take(&mut self.slots);
        self.slots = vec![Slot::default(); old.len() * 2];
        self.mask = self.slots.len() - 1;
        for sl in old {
            if sl.tag != 0 {
                let h = self.slot_hash(&sl);
                self.place(sl, h);
            }
        }
    }

    fn insert(&mut self, key: &[u8], lo: u64, hi: u64, h: u64, toks: &[Tok]) {
        if self.len >= self.limit || key.len() > u16::MAX as usize || toks.len() >= ARENA as usize {
            return;
        }
        if (self.len + 1) * 2 > self.slots.len() {
            self.grow();
        }
        let mut sl = Slot { tag: ((h >> 32) as u32) | 1, klen: key.len() as u16, ..Default::default() };
        if key.len() <= 16 {
            sl.key[..8].copy_from_slice(&lo.to_le_bytes());
            sl.key[8..].copy_from_slice(&hi.to_le_bytes());
        } else {
            sl.key[..4].copy_from_slice(&(self.keys.len() as u32).to_le_bytes());
            self.keys.extend_from_slice(key);
        }
        if toks.len() == 1 && toks[0].iter().all(|&x| x <= u16::MAX as u32) {
            sl.ntok = 1;
            let t = toks[0];
            sl.tok = [t[0] as u16, t[1] as u16, t[2] as u16, t[3] as u16];
        } else {
            let off = self.toks.len();
            self.toks.extend_from_slice(toks);
            sl.ntok = toks.len() as u16 | ARENA;
            sl.tok[0] = off as u16;
            sl.tok[1] = (off >> 16) as u16;
        }
        self.place(sl, h);
        self.len += 1;
    }
}

/// Groups of the unit starting at `a` (COMB_PAT): -> (word start, word end, unit end, junk)
#[inline]
fn parse_unit(m: &Model, s: &[u8], a: usize) -> (usize, usize, usize, bool) {
    let n = s.len();
    let mut i = a;
    while i < n {
        let (c, l) = m.class_at(s, i);
        if c & (C_WORD | C_NUM) != 0 {
            break;
        }
        i += l;
    }
    if i == n {
        return (n, n, n, true);
    }
    let b = i;
    let (c0, _) = m.class_at(s, i);
    let want = if c0 & C_WORD != 0 { C_WORD } else { C_NUM };
    while i < n {
        let (c, l) = m.class_at(s, i);
        if c & want == 0 {
            break;
        }
        i += l;
    }
    let c_ = i;
    if !m.punct_to_next {
        while i < n {
            let (c, l) = m.class_at(s, i);
            if c & (C_WORD | C_NUM | C_WS) != 0 {
                break;
            }
            i += l;
        }
    }
    (b, c_, i, false)
}

/// Tokens of the unit s[a..d] (cache hit: one probe and a copy); `lo`/`hi` are its first 16
/// bytes zero-padded and `h` its hash.
#[inline(always)]
fn emit_unit_h(m: &Model, s: &[u8], a: usize, d: usize, lo: u64, hi: u64, h: u64, cache: &mut Cache,
               out: &mut Vec<Tok>) {
    let key = &s[a..d];
    if cache.get_emit(key, lo, hi, h, out) {
        return;
    }
    let (b, c_, e, junk) = parse_unit(m, s, a);
    debug_assert_eq!(e, d);
    let start = out.len();
    m.encode_unit(s, a, b, c_, d, junk, out);
    cache.insert(key, lo, hi, h, &out[start..]);
}

#[inline(always)]
fn emit_unit(m: &Model, s: &[u8], a: usize, d: usize, cache: &mut Cache, out: &mut Vec<Tok>) {
    let key = &s[a..d];
    let (lo, hi) = load16(key);
    let h = hash_key(key, lo, hi);
    emit_unit_h(m, s, a, d, lo, hi, h, cache, out)
}

fn encode_text_scalar(m: &Model, s: &[u8], cache: &mut Cache, out: &mut Vec<Tok>) {
    let mut a = 0usize;
    while a < s.len() {
        let (_, _, d, _) = parse_unit(m, s, a);
        emit_unit(m, s, a, d, cache, out);
        a = d;
    }
}

/// Unit boundaries 64 bytes at a time. Per block: classes by one VBMI byte-table lookup
/// (per-char fill only in blocks holding non-ASCII bytes), then masks W (letters/marks),
/// D (digits), S (space), P (the rest). A P run that directly follows an alnum char is the
/// unit's suffix; one carry-propagating add finds all such runs at once. With T = W|D|suffix,
/// a new unit starts at every char whose predecessor is in T and that is a space, or a
/// letter not preceded by a letter, or a digit not preceded by a digit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn encode_text_avx512(m: &Model, s: &[u8], cache: &mut Cache, out: &mut Vec<Tok>) {
    use std::arch::x86_64::*;
    let n = s.len();
    let tlo = _mm512_loadu_si512(m.cls.as_ptr() as *const _);
    let thi = _mm512_loadu_si512(m.cls.as_ptr().add(64) as *const _);
    let vw = _mm512_set1_epi8(C_WORD as i8);
    let vd = _mm512_set1_epi8(C_NUM as i8);
    let vs = _mm512_set1_epi8(C_WS as i8);
    let (mut pw, mut pd, mut pt) = (0u64, 0u64, 0u64);
    let mut ustart = 0usize;
    let mut pos = 0usize;
    let mut units: Vec<Unit> = UNITS.with(|c| std::mem::take(&mut *c.borrow_mut()));
    let mut carry_cls = 0u8;
    let mut carry_left = 0usize;
    let mut buf = [0u8; 64];
    while pos < n {
        let len = (n - pos).min(64);
        let valid: u64 = if len == 64 { !0 } else { (1u64 << len) - 1 };
        let bytes = _mm512_maskz_loadu_epi8(valid, s.as_ptr().add(pos) as *const i8);
        let hi = _mm512_movepi8_mask(bytes) & valid;
        let cls = if hi == 0 && carry_left == 0 {
            _mm512_permutex2var_epi8(tlo, bytes, thi)
        } else {
            let mut j = 0usize;
            while j < len && carry_left > 0 {
                buf[j] = carry_cls;
                j += 1;
                carry_left -= 1;
            }
            while j < len {
                let b = *s.get_unchecked(pos + j);
                if b < 0x80 {
                    buf[j] = m.cls[b as usize];
                    j += 1;
                } else {
                    let (cp, l) = decode_at(s, pos + j);
                    let c = m.cls[cp as usize];
                    let take = l.min(len - j);
                    for k in 0..take {
                        buf[j + k] = c;
                    }
                    j += take;
                    if take < l {
                        carry_cls = c;
                        carry_left = l - take;
                    }
                }
            }
            _mm512_maskz_loadu_epi8(valid, buf.as_ptr() as *const i8)
        };
        let w = _mm512_test_epi8_mask(cls, vw) & valid;
        let d = _mm512_test_epi8_mask(cls, vd) & valid;
        let sp = _mm512_test_epi8_mask(cls, vs) & valid;
        let p = valid & !(w | d | sp);
        let starts = p & (((w | d) << 1) | pt);
        let psuf = if m.punct_to_next { 0 } else { p & !(p.wrapping_add(starts)) };
        let t = w | d | psuf;
        let tsh = (t << 1) | pt;
        let mut bnd = tsh & (sp | p & !psuf | (w & !((w << 1) | pw)) | (d & !((d << 1) | pd)));
        while bnd != 0 {
            let q = pos + bnd.trailing_zeros() as usize;
            push_unit(s, ustart, q, cache, &mut units);
            ustart = q;
            bnd &= bnd - 1;
        }
        pw = w >> 63;
        pd = d >> 63;
        pt = t >> 63;
        pos += len;
    }
    if ustart < n {
        push_unit(s, ustart, n, cache, &mut units);
    }
    for u in units.iter() {
        emit_unit_h(m, s, u.a as usize, u.d as usize, u.lo, u.hi, u.h, cache, out);
    }
    units.clear();
    UNITS.with(|c| *c.borrow_mut() = units);
}

#[derive(Clone, Copy)]
struct Unit {
    a: u32,
    d: u32,
    lo: u64,
    hi: u64,
    h: u64,
}

thread_local! {
    static UNITS: std::cell::RefCell<Vec<Unit>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// phase 1 of a text: record the unit, hash it (one masked 16-byte load) and prefetch its
/// cache slot, so phase 2's probes find their lines already on the way
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
unsafe fn push_unit(s: &[u8], a: usize, d: usize, cache: &Cache, units: &mut Vec<Unit>) {
    use std::arch::x86_64::*;
    let len = d - a;
    let mask: u16 = if len >= 16 { 0xFFFF } else { ((1u32 << len) - 1) as u16 };
    let v = _mm_maskz_loadu_epi8(mask, s.as_ptr().add(a) as *const i8);
    let lo = _mm_cvtsi128_si64(v) as u64;
    let hi = _mm_extract_epi64::<1>(v) as u64;
    let h = hash_key(&s[a..d], lo, hi);
    _mm_prefetch::<_MM_HINT_T0>(cache.slot_ptr(h) as *const i8);
    units.push(Unit { a: a as u32, d: d as u32, lo, hi, h });
}

/// profiling twin of encode_text_avx512 that only counts units (ASCII blocks)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn count_units_avx512(m: &Model, s: &[u8]) -> usize {
    use std::arch::x86_64::*;
    let n = s.len();
    let tlo = _mm512_loadu_si512(m.cls.as_ptr() as *const _);
    let thi = _mm512_loadu_si512(m.cls.as_ptr().add(64) as *const _);
    let vw = _mm512_set1_epi8(C_WORD as i8);
    let vd = _mm512_set1_epi8(C_NUM as i8);
    let vs = _mm512_set1_epi8(C_WS as i8);
    let (mut pw, mut pd, mut pt) = (0u64, 0u64, 0u64);
    let mut pos = 0usize;
    let mut units = 1usize;
    while pos < n {
        let len = (n - pos).min(64);
        let valid: u64 = if len == 64 { !0 } else { (1u64 << len) - 1 };
        let bytes = _mm512_maskz_loadu_epi8(valid, s.as_ptr().add(pos) as *const i8);
        let cls = _mm512_permutex2var_epi8(tlo, bytes, thi);
        let w = _mm512_test_epi8_mask(cls, vw) & valid;
        let d = _mm512_test_epi8_mask(cls, vd) & valid;
        let sp = _mm512_test_epi8_mask(cls, vs) & valid;
        let p = valid & !(w | d | sp);
        let starts = p & (((w | d) << 1) | pt);
        let psuf = p & !(p.wrapping_add(starts));
        let t = w | d | psuf;
        let bnd = ((t << 1) | pt) & (sp | (w & !((w << 1) | pw)) | (d & !((d << 1) | pd)));
        units += bnd.count_ones() as usize;
        pw = w >> 63;
        pd = d >> 63;
        pt = t >> 63;
        pos += len;
    }
    units
}

fn encode_text(m: &Model, s: &[u8], cache: &mut Cache, out: &mut Vec<Tok>) {
    #[cfg(target_arch = "x86_64")]
    {
        if m.simd && s.len() < u32::MAX as usize {
            unsafe { encode_text_avx512(m, s, cache, out) };
            return;
        }
    }
    encode_text_scalar(m, s, cache, out)
}

fn scan_units(m: &Model, s: &[u8]) -> usize {
    let n = s.len();
    let mut i = 0usize;
    let mut units = 0usize;
    while i < n {
        while i < n {
            let (c, l) = m.class_at(s, i);
            if c & (C_WORD | C_NUM) != 0 {
                break;
            }
            i += l;
        }
        if i < n {
            let (c0, _) = m.class_at(s, i);
            let want = if c0 & C_WORD != 0 { C_WORD } else { C_NUM };
            while i < n {
                let (c, l) = m.class_at(s, i);
                if c & want == 0 {
                    break;
                }
                i += l;
            }
            while i < n {
                let (c, l) = m.class_at(s, i);
                if c & (C_WORD | C_NUM | C_WS) != 0 {
                    break;
                }
                i += l;
            }
        }
        units += 1;
    }
    units
}

fn to_array<'py>(py: Python<'py>, toks: Vec<Tok>) -> Bound<'py, PyAny> {
    // [u32; 4] and i32 share alignment and ids are < 2^31: reinterpret, no copy
    let n = toks.len();
    let mut v = std::mem::ManuallyDrop::new(toks);
    let flat = unsafe { Vec::from_raw_parts(v.as_mut_ptr() as *mut i32, v.len() * 4, v.capacity() * 4) };
    Array2::from_shape_vec((n, 4), flat).unwrap().into_pyarray(py).into_any()
}

#[pyclass(module = "cbpe_fast._native")]
struct Native {
    m: Model,
    cache: Mutex<Cache>,
    pool: Vec<Mutex<Cache>>,
}

#[pymethods]
impl Native {
    #[new]
    #[pyo3(signature = (cls, fold_src, fold_dst, char_cps, char_ids, merge_a, merge_b, merge_rank, merge_out,
                        prefixes, suffixes, max_p, max_s, fold_case, fold_han, split_camel, punct_to_next,
                        han_src, han_dst, han_trad_keys, dec_core, cache_limit=2_000_000))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        cls: &Bound<'_, PyBytes>,
        fold_src: Vec<u32>,
        fold_dst: Vec<u32>,
        char_cps: Vec<u32>,
        char_ids: Vec<u32>,
        merge_a: Vec<u32>,
        merge_b: Vec<u32>,
        merge_rank: Vec<u32>,
        merge_out: Vec<u32>,
        prefixes: Vec<Vec<u8>>,
        suffixes: Vec<Vec<u8>>,
        max_p: usize,
        max_s: usize,
        fold_case: bool,
        fold_han: bool,
        split_camel: bool,
        punct_to_next: bool,
        han_src: Vec<u32>,
        han_dst: Vec<u32>,
        han_trad_keys: Vec<u32>,
        dec_core: Vec<Vec<Vec<u8>>>,
        cache_limit: usize,
    ) -> PyResult<Self> {
        let cls: Vec<u8> = cls.as_bytes().to_vec();
        if cls.len() != 0x110000 {
            return Err(pyo3::exceptions::PyValueError::new_err("cls table must cover 0x110000 code points"));
        }
        let mut ascii_id = [u32::MAX; 128];
        let mut char_id = FxHashMap::default();
        for (&cp, &id) in char_cps.iter().zip(&char_ids) {
            if cp < 128 {
                ascii_id[cp as usize] = id;
            }
            char_id.insert(cp, id);
        }
        let mut merges = FxHashMap::default();
        for i in 0..merge_a.len() {
            // later ranks overwrite earlier ones for a repeated pair, like the reference's dict
            merges.insert(((merge_a[i] as u64) << 32) | merge_b[i] as u64, (merge_rank[i], merge_out[i]));
        }
        let pmap: FxHashMap<Vec<u8>, u32> = prefixes
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.is_empty())
            .map(|(i, p)| (p.clone(), i as u32))
            .collect();
        let smap: FxHashMap<Vec<u8>, u32> = suffixes
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.is_empty())
            .map(|(i, p)| (p.clone(), i as u32))
            .collect();
        let m = Model {
            cls,
            fold: fold_src.into_iter().zip(fold_dst).collect(),
            han_fold: han_src.into_iter().zip(han_dst).collect(),
            han_to_trad: han_trad_keys.into_iter().collect(),
            ascii_id,
            char_id,
            merges,
            prefixes: pmap,
            suffixes: smap,
            max_p,
            max_s,
            fold_case,
            fold_han,
            split_camel,
            punct_to_next,
            simd: simd_ok(),
            dec_prefix: prefixes,
            dec_suffix: suffixes,
            dec_core,
        };
        let nt = rayon::current_num_threads().max(1);
        Ok(Native {
            m,
            cache: Mutex::new(Cache::new(cache_limit)),
            pool: (0..nt).map(|_| Mutex::new(Cache::new(cache_limit / nt + 1))).collect(),
        })
    }

    /// ids as an (n, 4) int32 array: columns variation, prefix, core, suffix
    fn encode<'py>(&self, py: Python<'py>, text: &Bound<'py, PyString>) -> PyResult<Bound<'py, PyAny>> {
        let s = text.to_str()?.as_bytes();
        let mut out = Vec::with_capacity(s.len() / 4 + 8);
        {
            let mut c = self.cache.lock().unwrap();
            encode_text(&self.m, s, &mut c, &mut out);
        }
        Ok(to_array(py, out))
    }

    /// the reference's return type: a list of (variation, prefix, core, suffix) tuples
    fn encode_tuples<'py>(&self, py: Python<'py>, text: &Bound<'py, PyString>) -> PyResult<Bound<'py, PyList>> {
        let s = text.to_str()?.as_bytes();
        let mut out = Vec::with_capacity(s.len() / 4 + 8);
        {
            let mut c = self.cache.lock().unwrap();
            encode_text(&self.m, s, &mut c, &mut out);
        }
        let items: Vec<Bound<'py, PyTuple>> = out
            .iter()
            .map(|t| PyTuple::new(py, t.iter().copied()))
            .collect::<PyResult<_>>()?;
        PyList::new(py, items)
    }

    /// many texts in parallel (rayon, one cache per worker), GIL released while encoding.
    /// The Python strings are read in place (no copy); returns one (n_i, 4) array per text.
    fn encode_batch<'py>(&self, py: Python<'py>, texts: &Bound<'py, PyList>) -> PyResult<Bound<'py, PyList>> {
        let results = self.batch_encode(py, texts)?;
        let arrays: Vec<Bound<'py, PyAny>> = results.into_iter().map(|r| to_array(py, r)).collect();
        PyList::new(py, arrays)
    }

    /// like encode_batch, but one (total, 4) array plus (len(texts) + 1) int64 offsets: no
    /// per-text Python objects, and the concatenation is a parallel copy into disjoint slices
    fn encode_batch_flat<'py>(&self, py: Python<'py>, texts: &Bound<'py, PyList>)
        -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let results = self.batch_encode(py, texts)?;
        let mut offs = Vec::with_capacity(results.len() + 1);
        offs.push(0i64);
        for r in &results {
            offs.push(offs[offs.len() - 1] + r.len() as i64);
        }
        let total = *offs.last().unwrap() as usize;
        let flat: Vec<Tok> = py.detach(|| {
            let mut flat: Vec<Tok> = Vec::with_capacity(total);
            // every element below is written exactly once before the Vec is read
            unsafe { flat.set_len(total) };
            let mut parts: Vec<&mut [Tok]> = Vec::with_capacity(results.len());
            let mut rest: &mut [Tok] = &mut flat[..];
            for r in &results {
                let (a, b) = rest.split_at_mut(r.len());
                parts.push(a);
                rest = b;
            }
            parts.into_par_iter().zip(results.par_iter()).for_each(|(dst, src)| dst.copy_from_slice(src));
            flat
        });
        Ok((to_array(py, flat), numpy::ndarray::Array1::from_vec(offs).into_pyarray(py).into_any()))
    }

    /// (n, 4) int32 ids -> the UTF-8 bytes of the text (the wrapper decodes them with
    /// errors="replace", as the reference decode does)
    fn decode<'py>(&self, py: Python<'py>, ids: numpy::PyReadonlyArray2<'py, i32>) -> PyResult<Bound<'py, PyBytes>> {
        let a = ids.as_array();
        if a.ncols() != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err("ids must have shape (n, 4)"));
        }
        let m = &self.m;
        let mut buf: Vec<u8> = Vec::with_capacity(a.nrows() * 6);
        for row in a.rows() {
            let (v, p, c, s) = (row[0] as usize, row[1] as usize, row[2] as usize, row[3] as usize);
            let (Some(pp), Some(cc), Some(ss)) = (m.dec_prefix.get(p), m.dec_core.get(v).and_then(|t| t.get(c)), m.dec_suffix.get(s)) else {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("token out of range: {:?}", (v, p, c, s))));
            };
            buf.extend_from_slice(pp);
            buf.extend_from_slice(cc);
            buf.extend_from_slice(ss);
        }
        Ok(PyBytes::new(py, &buf))
    }

    /// profiling aid: seconds for `reps` passes over `texts` inside Rust (no Python objects),
    /// and how many of those seconds went to scanning units vs everything else
    fn time_encode(&self, texts: Vec<String>, reps: usize) -> (f64, f64) {
        let mut c = self.cache.lock().unwrap();
        let mut out: Vec<Tok> = Vec::with_capacity(1 << 16);
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            for t in &texts {
                out.clear();
                encode_text(&self.m, t.as_bytes(), &mut c, &mut out);
            }
        }
        let total = t0.elapsed().as_secs_f64();
        // scan-only pass: the same unit boundaries, no cache or output
        let t1 = std::time::Instant::now();
        let mut units = 0usize;
        for _ in 0..reps {
            for t in &texts {
                #[cfg(target_arch = "x86_64")]
                {
                    if self.m.simd {
                        units += unsafe { count_units_avx512(&self.m, t.as_bytes()) };
                        continue;
                    }
                }
                units += scan_units(&self.m, t.as_bytes());
            }
        }
        let scan = t1.elapsed().as_secs_f64();
        std::hint::black_box(units);
        (total, scan)
    }

    fn simd(&self) -> bool {
        self.m.simd
    }

    fn cache_size(&self) -> usize {
        self.cache.lock().unwrap().len
    }

    fn clear_cache(&self) {
        self.cache.lock().unwrap().clear();
        for c in &self.pool {
            c.lock().unwrap().clear();
        }
    }
}

impl Native {
    fn batch_encode(&self, py: Python<'_>, texts: &Bound<'_, PyList>) -> PyResult<Vec<Vec<Tok>>> {
        let objs: Vec<Bound<'_, PyString>> = texts
            .iter()
            .map(|o| o.cast_into::<PyString>().map_err(PyErr::from))
            .collect::<PyResult<_>>()?;
        let strs: Vec<&str> = objs.iter().map(|o| o.to_str()).collect::<PyResult<_>>()?;
        let m = &self.m;
        let pool = &self.pool;
        Ok(py.detach(|| {
            strs.par_iter()
                .map(|t| {
                    let s = t.as_bytes();
                    let mut out = Vec::with_capacity(s.len() / 4 + 8);
                    let w = rayon::current_thread_index().unwrap_or(0) % pool.len();
                    let mut c = pool[w].lock().unwrap();
                    encode_text(m, s, &mut c, &mut out);
                    out
                })
                .collect()
        }))
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Native>()?;
    Ok(())
}
