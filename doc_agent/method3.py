import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np

from .claims import obs, inf, hyp


def affix_analysis(types, reverse=False, max_len=3):
    words = {(w[::-1] if reverse else w): c for w, c in types.items()}
    splits = defaultdict(set)
    for w in words:
        for k in range(1, max_len + 1):
            if len(w) - k >= 2:
                splits[w[:-k]].add(w[-k:])
    for stem in list(splits):
        if stem in words:
            splits[stem].add("")
    score = Counter()
    for stem, sufs in splits.items():
        if len(sufs) >= 2:
            for s in sufs:
                if s:
                    score[s] += 1
    best_split = {}
    for w in words:
        options = []
        for k in range(1, max_len + 1):
            if len(w) - k >= 2 and len(splits[w[:-k]]) >= 2:
                options.append((score[w[-k:]] * (1 + 0.15 * k), w[:-k], w[-k:]))
        if options:
            _, stem, suf = max(options)
            best_split[w] = (stem, suf)
    paradigms = defaultdict(set)
    for w, (stem, suf) in best_split.items():
        paradigms[stem].add(suf)
    for stem in paradigms:
        if stem in words:
            paradigms[stem].add("")
    paradigms = {s: v for s, v in paradigms.items() if len(v) >= 2}
    sigs = defaultdict(list)
    for stem, sufs in paradigms.items():
        sigs[tuple(sorted(sufs))].append(stem)
    sigs = {k: v for k, v in sigs.items() if len(v) >= 3}
    fix = (lambda s: s[::-1]) if reverse else (lambda s: s)
    return {
        "scores": [(fix(s), c) for s, c in score.most_common(12)],
        "signatures": sorted(((tuple(fix(x) for x in k), [fix(s) for s in v]) for k, v in sigs.items()),
                             key=lambda x: -len(x[1]) * len(x[0]))[:8],
        "paradigm_words": {fix(stem + suf) for stem, sufs in paradigms.items() for suf in sufs},
        "strength": sum(c for _, c in score.most_common(10)),
    }


def dispersion(corpus, n_chunks=10):
    toks = corpus.tokens
    size = max(1, len(toks) // n_chunks)
    chunks = [set(toks[i:i + size]) for i in range(0, len(toks), size)][:n_chunks]
    return {w: sum(w in c for c in chunks) / len(chunks) for w in set(toks)}


def positional(corpus, min_count=5):
    units = [u for u in corpus.units if len(u) >= 3]
    first, last, total = Counter(), Counter(), Counter()
    for u in units:
        first[u[0]] += 1
        last[u[-1]] += 1
        total.update(u)
    n_tok = sum(total.values()) or 1
    base = len(units) / n_tok
    initial, final = [], []
    for w, c in total.items():
        if c < min_count:
            continue
        ri, rf = first[w] / c / base, last[w] / c / base
        if ri >= 2.5:
            initial.append((w, c, round(first[w] / c, 2), round(ri, 1)))
        if rf >= 2.5:
            final.append((w, c, round(last[w] / c, 2), round(rf, 1)))
    initial.sort(key=lambda x: -x[3] * np.log1p(x[1]))
    final.sort(key=lambda x: -x[3] * np.log1p(x[1]))
    return initial[:8], final[:8], len(units)


def numeral_signs(corpus):
    signs = Counter(corpus.signs)
    doubled = Counter()
    for t in corpus.tokens:
        for a, b in zip(t, t[1:]):
            if a == b:
                doubled[a] += 1
    total = sum(signs.values()) or 1
    out = []
    for s, c in signs.items():
        if unicodedata.category(s) == "Nd":
            out.append((s, c, "Unicode digit"))
        elif c >= 5 and doubled[s] / c >= 0.25 and doubled[s] / c > 5 * c / total:
            out.append((s, c, f"repeats next to itself in {doubled[s] / c:.0%} of uses, like tally strokes"))
    return out


def number_context(corpus, num_signs):
    num = set(s for s, _, _ in num_signs)
    is_num = lambda t: t and all(ch in num for ch in t)
    before, after = Counter(), Counter()
    n_num = 0
    for u in corpus.units:
        for i, t in enumerate(u):
            if is_num(t):
                n_num += 1
                if i > 0 and not is_num(u[i - 1]):
                    before[u[i - 1]] += 1
                if i + 1 < len(u) and not is_num(u[i + 1]):
                    after[u[i + 1]] += 1
    freq = Counter(corpus.tokens)
    n = len(corpus.tokens) or 1
    lifts = []
    for side, cnt in (("before", before), ("after", after)):
        for w, c in cnt.items():
            expected = n_num * freq[w] / n
            if c >= 3 and expected > 0 and c / expected >= 3:
                lifts.append((w, side, c, round(c / expected, 1), round(c / freq[w], 2)))
    lifts.sort(key=lambda x: -x[2] * x[3])
    return n_num, lifts[:8]


def name_candidates(corpus, paradigm_words):
    freq = Counter(corpus.tokens)
    if corpus.has_case:
        caps = Counter()
        prev = "."
        for tok in re.findall(r"\S+", corpus.raw):
            core = re.sub(r"^\W+|\W+$", "", tok)
            if core[:1].isupper() and core[1:2].islower() and not re.search(r"[.!?:\"]$", prev):
                caps[core] += 1
            prev = tok
        lower_forms = freq
        out = [(w, c) for w, c in caps.most_common(40)
               if c >= 2 and lower_forms[w.lower()] <= c * 1.2]
        return out[:12], "capitalised in mid-sentence and rarely seen in lower case"
    toks = corpus.tokens
    n = len(toks) or 1
    positions = defaultdict(list)
    for i, t in enumerate(toks):
        positions[t].append(i)
    med = np.median([len(w) for w in freq]) if freq else 0
    out = []
    for w, pos in positions.items():
        c = len(pos)
        if 2 <= c <= 8 and len(w) >= med and w not in paradigm_words and (pos[-1] - pos[0]) / n <= 0.2:
            out.append((w, c))
    out.sort(key=lambda x: -x[1])
    return out[:12], "repeats a few times but only in one stretch of the text and never changes its ending"


def run(corpus, stats):
    from .method2 import shuffled_tokens
    claims = []
    nums = numeral_signs(corpus)
    num_set = set(s for s, _, _ in nums)
    words_only = [t for t in corpus.tokens if not all(c in num_set for c in t)]
    types = Counter(words_only)
    empty = {"scores": [], "signatures": [], "paradigm_words": set(), "strength": 0}
    if corpus.separator == "none" or len(types) < 20:
        claims.append(obs("Too few word types (or no word boundaries) for morphological analysis."))
        suf = pre = empty
    else:
        suf = affix_analysis(types)
        pre = affix_analysis(types, reverse=True)
        import random
        pool = sorted(set(shuffled_tokens(words_only, 3)))
        random.Random(3).shuffle(pool)
        ctrl_types = Counter(pool[:len(types)])
        ctrl_suf = affix_analysis(ctrl_types)
        ctrl_pre = affix_analysis(ctrl_types, reverse=True)
        ctrl_best = max([len(v) for _, v in ctrl_suf["signatures"]] + [2])
        suf["signatures"] = [(k, v) for k, v in suf["signatures"] if len(v) >= 2 * ctrl_best]
        claims.append(obs(f"Control: the same number of nonsense words built from shuffled signs give ending score "
                          f"{ctrl_suf['strength']} and beginning score {ctrl_pre['strength']}, against "
                          f"{suf['strength']} and {pre['strength']} here."))
        if suf["strength"] < ctrl_suf["strength"] * 1.3 and pre["strength"] < ctrl_pre["strength"] * 1.3:
            claims.append(inf("Affix patterns are no stronger than in shuffled nonsense, so no real "
                              "morphology is detected."))
            suf = {**empty, "scores": suf["scores"]}
            pre = {**empty, "scores": pre["scores"]}

    sh = corpus.show
    if suf["scores"]:
        claims.append(obs("Most productive word endings (number of stems they attach to): "
                          + ", ".join(f"-{sh(s)} ({c})" for s, c in suf["scores"][:8]) + "."))
    if pre["scores"]:
        claims.append(obs("Most productive word beginnings: "
                          + ", ".join(f"{sh(s)}- ({c})" for s, c in pre["scores"][:8]) + "."))
    for sig, stems in suf["signatures"][:5]:
        ends = ", ".join("-" + sh(x) if x else "(bare)" for x in sig)
        claims.append(inf(f"Ending set {{{ends}}} is shared by {len(stems)} stems, e.g. "
                          + ", ".join(sh(s) for s in stems[:4])
                          + ". A recurring ending set is how Alice Kober found the Linear B inflection "
                            "triplets; it points to case, number, gender or verb endings."))
    if suf["strength"] or pre["strength"]:
        side = "suffixing" if suf["strength"] >= pre["strength"] * 1.3 else (
            "prefixing" if pre["strength"] >= suf["strength"] * 1.3 else "mixed")
        claims.append(inf(f"The language looks {side} (ending score {suf['strength']} vs beginning score {pre['strength']})."))
    else:
        side = None

    disp = dispersion(corpus)
    med_len = np.median([len(w) for w in types]) if types else 0
    total = len(corpus.tokens) or 1
    func = [(w, c) for w, c in types.most_common(40)
            if c / total >= 0.005 and len(w) <= med_len and disp.get(w, 0) >= 0.7 and w not in suf["paradigm_words"]][:10]
    if func:
        claims.append(inf("Function-word candidates (frequent, short, spread through the whole text, never inflected): "
                          + ", ".join(f"{sh(w)} ({c})" for w, c in func) + "."))

    initial, final, n_units = positional(corpus)
    if initial:
        claims.append(obs(f"Words strongly tied to the start of a {corpus.unit_kind}: "
                          + ", ".join(f"{sh(w)} ({p:.0%} of its {c} uses, {r}x chance)" for w, c, p, r in initial[:5]) + "."))
    if final:
        claims.append(obs(f"Words strongly tied to the end of a {corpus.unit_kind}: "
                          + ", ".join(f"{sh(w)} ({p:.0%} of its {c} uses, {r}x chance)" for w, c, p, r in final[:5]) + "."))

    fset = set(w for w, _ in func)
    fw_first = fw_second = 0
    for u in corpus.units:
        for a, b in zip(u, u[1:]):
            if a in fset and b not in fset:
                fw_first += 1
            elif b in fset and a not in fset:
                fw_second += 1
    order = None
    if fw_first + fw_second >= 20:
        ratio = fw_first / (fw_first + fw_second)
        if ratio >= 0.6:
            order = "head-initial"
            claims.append(hyp(f"Function-word candidates come before a content word {ratio:.0%} of the time, "
                              f"like prepositions and articles in head-initial languages (e.g. SVO English)."))
        elif ratio <= 0.4:
            order = "head-final"
            claims.append(hyp(f"Function-word candidates come after a content word {1 - ratio:.0%} of the time, "
                              f"like postpositions and particles in head-final languages (e.g. SOV Hindi, Japanese)."))
        if order and side:
            agree = (order == "head-final" and side == "suffixing") or (order == "head-initial" and side == "prefixing")
            claims.append(hyp(f"{'This agrees' if agree else 'This does not match the usual tendency'}: "
                              f"suffixing languages tend to be head-final (Greenberg's correlations)."))
    if final and order == "head-final":
        claims.append(hyp("In verb-final languages, words that cluster at the end of a clause are often verbs, "
                          "copulas or sentence particles: " + ", ".join(sh(w) for w, *_ in final[:4]) + "."))

    n_num, ctx = number_context(corpus, nums) if nums else (0, [])
    if nums:
        claims.append(obs("Number-like signs: " + "; ".join(f"{sh(s)} ({c}x, {why})" for s, c, why in nums[:8]) + "."))
        claims.append(obs(f"{n_num} tokens are made only of number-like signs."))
        for w, sidename, c, lift, share in ctx[:5]:
            claims.append(inf(f"{sh(w)} appears right {sidename} a number {c} times ({lift}x chance). Words that "
                              f"hug numbers are often units, commodities or 'total' markers, as on Linear B tablets."))
    else:
        claims.append(obs("No digit-like or tally-like signs detected."))

    names, how = name_candidates(corpus, suf["paradigm_words"])
    if names:
        claims.append(inf(f"Possible proper names ({how}): " + ", ".join(f"{sh(w)} ({c})" for w, c in names[:10]) + "."))

    return {
        "suffixes": [(sh(s), c) for s, c in suf["scores"]],
        "prefixes": [(sh(s), c) for s, c in pre["scores"]],
        "signatures": [([sh(x) for x in k], [sh(s) for s in v[:6]], len(v)) for k, v in suf["signatures"]],
        "affix_type": side, "order_hint": order,
        "function_words": [(sh(w), c) for w, c in func],
        "initial_words": [(sh(w), c, p, r) for w, c, p, r in initial],
        "final_words": [(sh(w), c, p, r) for w, c, p, r in final],
        "numeral_signs": [(sh(s), c, why) for s, c, why in nums],
        "number_context": [(sh(w), sd, c, l, s) for w, sd, c, l, s in ctx],
        "names": [(sh(w), c) for w, c in names],
        "claims": claims,
        "_func": fset, "_nums": set(s for s, _, _ in nums), "_paradigm_words": suf["paradigm_words"],
    }
