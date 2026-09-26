import random
from collections import Counter

import numpy as np

from . import references as ref
from .claims import obs, inf, hyp
from .corpus import entropy
from .method2 import shuffled_tokens

LANG_NAMES = {
    "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali", "bs": "Bosnian", "ca": "Catalan", "cs": "Czech",
    "da": "Danish", "de": "German", "el": "Greek", "en": "English", "es": "Spanish", "fa": "Persian",
    "fi": "Finnish", "fil": "Filipino", "fr": "French", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian",
    "hu": "Hungarian", "id": "Indonesian", "is": "Icelandic", "it": "Italian", "ja": "Japanese", "ko": "Korean",
    "lt": "Lithuanian", "lv": "Latvian", "mk": "Macedonian", "ms": "Malay", "nb": "Norwegian", "nl": "Dutch",
    "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sh": "Serbo-Croatian",
    "sk": "Slovak", "sl": "Slovenian", "sr": "Serbian", "sv": "Swedish", "ta": "Tamil", "tr": "Turkish",
    "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese", "zh": "Chinese",
}


def name(lang):
    return LANG_NAMES.get(lang, lang)


def doc_profile(corpus, m3):
    signs = Counter(corpus.signs)
    total = sum(signs.values()) or 1
    dist = sorted((v / total for v in signs.values()), reverse=True)
    suf = sum(c for _, c in m3["suffixes"][:10])
    pre = sum(c for _, c in m3["prefixes"][:10])
    return {"rank_profile": (dist + [0] * ref.RANKS)[:ref.RANKS], "h1": entropy(signs),
            "inventory": len(signs), "word_len": float(np.mean([len(t) for t in corpus.tokens])) if corpus.tokens else 0,
            "suffix_ratio": suf / max(1, suf + pre)}


def profile_distance(a, b):
    shape = float(np.abs(np.array(a["rank_profile"]) - np.array(b["rank_profile"])).sum())
    return shape + abs(a["h1"] - b["h1"]) * 0.25 + abs(a["word_len"] - b["word_len"]) * 0.1 \
        + abs(a["suffix_ratio"] - b["suffix_ratio"]) * 0.3


def readable_script(corpus, detection):
    if corpus.glyph_mode:
        return False
    return detection["script"]["unknown_share"] < 0.2 and detection["script"]["letters"] > 0


def lexical_matching(corpus, langs, seed=0):
    from unidecode import unidecode
    types = [t for t in set(corpus.tokens) if len(t) >= 3]
    control = [t for t in set(shuffled_tokens(corpus.tokens, seed)) if len(t) >= 3]
    out = []
    for lang in langs:
        idx = ref.deletion_index(lang)
        hits = [(t, ref.near_match(unidecode(t).lower(), idx)) for t in types]
        hits = [(t, m) for t, m in hits if m]
        ctrl = sum(1 for t in control if ref.near_match(unidecode(t).lower(), idx))
        rate = len(hits) / max(1, len(types))
        crate = ctrl / max(1, len(control))
        out.append({"lang": lang, "rate": round(rate, 4), "control": round(crate, 4),
                    "lift": round(rate / crate, 2) if crate else None, "examples": hits[:10]})
    out.sort(key=lambda x: -(x["rate"] - x["control"]))
    return out


def trigram_counts(tokens, sign_index):
    k = len(sign_index) + 1
    C = np.zeros((k, k, k))
    for t in tokens:
        seq = [0, 0] + [sign_index[c] for c in t] + [0]
        for a, b, c in zip(seq, seq[1:], seq[2:]):
            C[a, b, c] += 1
    return C


def score_of(C, lp, m):
    return float((C * lp[np.ix_(m, m, m)]).sum())


def climb(C, lp, n_letters, init, iters, rng):
    m = init.copy()
    score = score_of(C, lp, m)
    k = len(m)
    unused = [x for x in range(1, n_letters) if x not in set(m[1:])]
    temp = 20.0
    for _ in range(iters):
        cand = m.copy()
        i = rng.integers(1, k)
        if unused and rng.random() < 0.3:
            j = rng.integers(len(unused))
            cand[i], new_unused = unused[j], unused[:j] + [m[i]] + unused[j + 1:]
        else:
            j = rng.integers(1, k)
            cand[i], cand[j] = cand[j], cand[i]
            new_unused = unused
        s = score_of(C, lp, cand)
        if s > score or rng.random() < np.exp((s - score) / temp):
            m, score, unused = cand, s, new_unused
        temp = max(temp * 0.999, 0.05)
    return m, score


def search_mapping(tokens, lang, iters=6000, restarts=3, seed=0):
    alphabet, lp = ref.letter_model(lang)
    signs = [s for s, _ in Counter(c for t in tokens for c in t).most_common()]
    sign_index = {s: i + 1 for i, s in enumerate(signs)}
    C = trigram_counts(tokens, sign_index)
    n_letters = len(alphabet)
    base = np.zeros(len(signs) + 1, dtype=int)
    for i in range(1, len(signs) + 1):
        base[i] = i if i < n_letters else 1 + (i - 1) % (n_letters - 1)
    rng = np.random.default_rng(seed)
    best = None
    for r in range(restarts):
        init = base.copy()
        if r:
            for _ in range(len(signs) // 3):
                a, b = rng.integers(1, len(signs) + 1, size=2)
                init[a], init[b] = init[b], init[a]
        m, s = climb(C, lp, n_letters, init, iters, rng)
        if best is None or s > best[1]:
            best = (m, s)
    return {s: alphabet[best[0][i]] for s, i in sign_index.items()}


def decode(tokens, mapping):
    return ["".join(mapping.get(c, "?") for c in t) for t in tokens]


def lexicon_rate(words, lang):
    lex = ref.lexicon(lang, 30000)
    hits = [w for w in words if len(w) >= 3 and w in lex]
    return len(hits) / max(1, len([w for w in words if len(w) >= 3])), hits


def decipher(corpus, lang, iters=6000):
    toks = corpus.tokens
    mapping = search_mapping(toks, lang, iters=iters)
    decoded = decode(toks, mapping)
    rate, hits = lexicon_rate(decoded, lang)
    controls = []
    for s in (1, 2):
        sh = shuffled_tokens(toks, 100 + s)
        cm = search_mapping(sh, lang, iters=iters, seed=s)
        controls.append(lexicon_rate(decode(sh, cm), lang)[0])
    ctrl = float(np.mean(controls))
    sample_unit = next((u for u in corpus.units if len(u) >= 5), corpus.units[0] if corpus.units else [])
    return {"lang": lang, "mapping": {corpus.show(k): v for k, v in mapping.items()},
            "hit_rate": round(rate, 4), "control_rate": round(ctrl, 4),
            "top_hits": Counter(hits).most_common(12),
            "sample": " ".join(decode(sample_unit, mapping)),
            "sample_source": " ".join(corpus.show(t) for t in sample_unit)}


def run(corpus, detection, m1, m2, m3, decipher_langs=None, iters=6000):
    claims = []
    profiles = ref.all_profiles()
    dp = doc_profile(corpus, m3)
    ranked = sorted(((profile_distance(dp, p), lang) for lang, p in profiles.items()), key=lambda x: x[0])
    closest = ranked[:6]
    if closest:
        claims.append(inf("Closest statistical profiles among the reference languages (sign-frequency shape, "
                          "entropy, word length, suffix preference): "
                          + ", ".join(f"{name(l)} ({d:.2f})" for d, l in closest) + ". Smaller is closer."))
        claims.append(hyp("A close profile means the text behaves like that language statistically. It does not "
                          "mean the text is in that language or related to it; unrelated languages can share a "
                          "profile, and a syllabary will never match an alphabet profile."))

    lexical = []
    if readable_script(corpus, detection):
        langs = [l for _, l in ranked[:12]]
        for extra in ("en", "la", "it", "de", "fr", "es", "el"):
            if extra in profiles and extra not in langs:
                langs.append(extra)
        lexical = lexical_matching(corpus, langs)
        real = [r for r in lexical if r["rate"] - r["control"] >= 0.1 and (r["lift"] or 0) >= 1.5]
        for row in real[:3]:
            ex = ", ".join(f"{a}~{b}" for a, b in row["examples"][:6])
            claims.append(hyp(f"Vocabulary look-alikes with {name(row['lang'])}: {row['rate']:.0%} of word types are "
                              f"within one letter of a {name(row['lang'])} word, versus {row['control']:.0%} for "
                              f"shuffled nonsense words. Examples: {ex}."))
        if not real and lexical:
            top = lexical[0]
            claims.append(inf(f"No reference language shares vocabulary clearly above chance. The best, "
                              f"{name(top['lang'])}, matches {top['rate']:.0%} of word types against {top['control']:.0%} "
                              f"for shuffled nonsense, which is what coincidence produces. The sound values of the "
                              f"script are known, but no cognate anchor was found."))
    else:
        claims.append(obs("Sign values are unknown, so direct cognate matching is impossible without a "
                          "hypothesised sign-to-sound mapping."))

    deciphered = []
    if decipher_langs:
        if m1["distinct_signs"] > 45:
            claims.append(obs(f"Automatic sign-value search skipped: {m1['distinct_signs']} signs is too many for a "
                              f"one-sign-one-letter mapping. A syllabic grid (as Ventris built for Linear B) must be "
                              f"supplied as a sign_values hypothesis."))
        else:
            for lang in decipher_langs:
                if lang not in profiles:
                    claims.append(obs(f"No reference data for '{lang}', skipped."))
                    continue
                d = decipher(corpus, lang, iters=iters)
                deciphered.append(d)
                strong = d["hit_rate"] >= 0.2 and d["hit_rate"] >= 2 * max(d["control_rate"], 0.02)
                claims.append(hyp(
                    f"Sign-value search against {name(lang)}: best mapping turns {d['hit_rate']:.0%} of words into real "
                    f"{name(lang)} words; the same search on shuffled signs reaches {d['control_rate']:.0%}. "
                    + ("This gap is large, so the mapping deserves checking by hand. "
                       if strong else "No meaningful gap, so this mapping should be treated as noise. ")
                    + f"Sample: \"{d['sample'][:120]}\"."))
    passing = [d for d in deciphered if d["hit_rate"] >= 0.2 and d["hit_rate"] >= 2 * max(d["control_rate"], 0.02)]
    if len(passing) > 1:
        best = max(passing, key=lambda d: d["hit_rate"])
        claims.append(inf(f"Several languages beat their controls; {name(best['lang'])} scores highest "
                          f"({best['hit_rate']:.0%}). Lower scores in other languages usually come from shared "
                          f"or borrowed vocabulary and similar spelling habits."))
    anchors = []
    scripts = detection["script"]["scripts"]
    if len([s for s, v in scripts.items() if v >= 0.05]) >= 2:
        anchors.append("the document mixes two or more scripts; if one part is readable it may be a bilingual key")
    if m3["numeral_signs"]:
        anchors.append("number-like signs give a foothold for accounting words, as in Linear B")
    if m3["names"]:
        anchors.append("name candidates could be matched to known names of places or people from the same region")
    for a in anchors:
        claims.append(inf("Possible anchor: " + a + "."))
    return {"closest_profiles": [(name(l), round(d, 3)) for d, l in closest], "lexical": [
        {**r, "lang": name(r["lang"]), "examples": [(corpus.show(a), b) for a, b in r["examples"]]} for r in lexical[:5]],
        "decipherment": deciphered, "anchors": anchors, "claims": claims}
