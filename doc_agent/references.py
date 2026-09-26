import json
import math
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

CACHE = Path(os.environ.get("DOC_AGENT_CACHE", Path.home() / ".cache" / "doc_agent"))
N_WORDS = 8000
RANKS = 60


def available():
    try:
        from wordfreq import available_languages
        return sorted(available_languages())
    except ImportError:
        return []


def word_list(lang, n=N_WORDS):
    from wordfreq import top_n_list, word_frequency
    words = [w for w in top_n_list(lang, n) if any(c.isalpha() for c in w)]
    return [(w, word_frequency(w, lang)) for w in words]


def build_profile(lang):
    from .method3 import affix_analysis
    words = word_list(lang)
    chars = Counter()
    wl, wsum = 0.0, 0.0
    for w, f in words:
        letters = [c for c in w if c.isalpha()]
        for c in letters:
            chars[c] += f
        wl += len(letters) * f
        wsum += f
    total = sum(chars.values())
    dist = sorted((v / total for v in chars.values()), reverse=True)
    h1 = -sum(p * math.log2(p) for p in dist if p)
    cum, inv = 0, 0
    for p in dist:
        cum += p
        inv += 1
        if cum >= 0.995:
            break
    types = {w: 1 for w, _ in words[:5000] if w.isalpha()}
    suf = affix_analysis(types)["strength"]
    pre = affix_analysis(types, reverse=True)["strength"]
    return {"lang": lang, "rank_profile": (dist + [0] * RANKS)[:RANKS], "h1": h1, "inventory": inv,
            "word_len": wl / wsum, "suffix_ratio": suf / max(1, suf + pre)}


@lru_cache(maxsize=1)
def all_profiles():
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "profiles.json"
    if path.exists():
        return json.loads(path.read_text())
    profiles = {}
    for lang in available():
        try:
            profiles[lang] = build_profile(lang)
        except Exception:
            continue
    path.write_text(json.dumps(profiles))
    return profiles


@lru_cache(maxsize=16)
def lexicon(lang, n=30000):
    from wordfreq import top_n_list
    return set(top_n_list(lang, n))


@lru_cache(maxsize=16)
def deletion_index(lang, n=30000):
    from unidecode import unidecode
    idx = {}
    for w in lexicon(lang, n):
        a = unidecode(w).lower()
        if len(a) < 3:
            continue
        for v in [a] + [a[:i] + a[i + 1:] for i in range(len(a))]:
            idx.setdefault(v, a)
    return idx


def near_match(word, idx):
    if word in idx:
        return idx[word]
    for i in range(len(word)):
        v = word[:i] + word[i + 1:]
        if v in idx:
            return idx[v]
    return None


@lru_cache(maxsize=16)
def letter_model(lang):
    words = word_list(lang, 20000)
    chars = Counter()
    for w, f in words:
        for c in w:
            if c.isalpha():
                chars[c] += f
    total = sum(chars.values())
    cum, alphabet = 0, []
    for c, v in chars.most_common():
        alphabet.append(c)
        cum += v / total
        if cum >= 0.9995:
            break
    index = {c: i + 1 for i, c in enumerate(alphabet)}
    k = len(alphabet) + 1
    scale = 1e6 / sum(f for _, f in words)
    words = [(w, f * scale) for w, f in words]
    bi = np.zeros((k, k))
    tri = np.zeros((k, k, k))
    for w, f in words:
        seq = [0, 0] + [index[c] for c in w if c in index] + [0]
        for a, b, c in zip(seq, seq[1:], seq[2:]):
            tri[a, b, c] += f
            bi[b, c] += f
    uni = bi.sum(axis=0) + 1e-9
    p_uni = uni / uni.sum()
    p_bi = (bi + 1e-9) / (bi.sum(axis=1, keepdims=True) + 1e-9 * k)
    tri_tot = tri.sum(axis=2, keepdims=True)
    lam = tri_tot / (tri_tot + 5.0)
    p_tri = np.where(tri_tot > 0, tri / np.maximum(tri_tot, 1e-12), 0)
    p = lam * p_tri + (1 - lam) * (0.9 * p_bi[None, :, :] + 0.1 * p_uni[None, None, :])
    return ["#"] + alphabet, np.log(p)
