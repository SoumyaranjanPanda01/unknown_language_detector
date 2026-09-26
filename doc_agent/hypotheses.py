import json
import random
from collections import Counter

import numpy as np

from .method4 import decode, lexicon_rate, name

HELP = """Hypothesis commands (one per line):
  gloss TOKEN = MEANING near numbers|line_start|line_end|TOKEN2
  role TOKEN function_word|number|name|unit_start|unit_end
  affix -ENDING = MEANING          (or PREFIX- = MEANING)
  values G001=a G002=b ... in LANG  (sign-to-sound mapping, LANG is a code like en, de)
  help | quit"""


def parse_line(line, corpus):
    words = line.strip().split()
    if not words:
        return None
    cmd = words[0].lower()
    if cmd == "gloss":
        rest = " ".join(words[1:])
        tok, _, tail = rest.partition("=")
        meaning, _, anchor = tail.partition(" near ")
        return {"type": "gloss", "token": tok.strip(), "meaning": meaning.strip(), "anchor": anchor.strip() or None}
    if cmd == "role" and len(words) >= 3:
        return {"type": "role", "token": words[1], "role": words[2]}
    if cmd == "affix":
        rest = " ".join(words[1:])
        aff, _, meaning = rest.partition("=")
        return {"type": "affix", "affix": aff.strip(), "meaning": meaning.strip()}
    if cmd == "values":
        rest = " ".join(words[1:])
        pairs, _, lang = rest.partition(" in ")
        mapping = dict(p.split("=", 1) for p in pairs.split() if "=" in p)
        return {"type": "sign_values", "map": mapping, "language": lang.strip() or "en"}
    return {"type": "unknown", "raw": line}


def occurrences(corpus, tok):
    out = []
    for ui, u in enumerate(corpus.units):
        for i, t in enumerate(u):
            if t == tok:
                out.append((ui, i))
    return out


def near(corpus, ui, i, anchor, nums):
    u = corpus.units[ui]
    neigh = [u[j] for j in (i - 1, i + 1) if 0 <= j < len(u)]
    if anchor == "numbers":
        return any(t and all(c in nums for c in t) for t in neigh)
    if anchor == "line_start":
        return i == 0
    if anchor == "line_end":
        return i == len(u) - 1
    target = corpus.parse(anchor)
    return target in neigh


def chance_rate(corpus, anchor, nums, trials=400, seed=0):
    rng = random.Random(seed)
    cells = [(ui, i) for ui, u in enumerate(corpus.units) for i in range(len(u))]
    if not cells:
        return 0
    pick = [cells[rng.randrange(len(cells))] for _ in range(trials)]
    return sum(near(corpus, ui, i, anchor, nums) for ui, i in pick) / trials


def phrase(anchor):
    return {"numbers": "next to a number", "line_start": "at the start of a unit",
            "line_end": "at the end of a unit"}.get(anchor, f"next to {anchor}")


def test_gloss(h, corpus, m3):
    tok = corpus.parse(h["token"])
    occ = occurrences(corpus, tok)
    res = {"hypothesis": h, "label": "HYPOTHESIS", "occurrences": len(occ)}
    if not occ:
        res.update(verdict="UNTESTABLE", evidence=f"token {h['token']} does not occur in the document")
        return res
    if not h.get("anchor"):
        res.update(verdict="UNTESTABLE", evidence=(f"{len(occ)} occurrences found, but a meaning cannot be checked "
                                                   f"without a context rule. Add 'near numbers', 'near line_end' or 'near TOKEN'."))
        return res
    nums = m3["_nums"]
    ok = [o for o in occ if near(corpus, *o, h["anchor"], nums)]
    rate = len(ok) / len(occ)
    base = chance_rate(corpus, h["anchor"], nums)
    bad = [" ".join(corpus.show(t) for t in corpus.units[ui]) for ui, i in occ if (ui, i) not in ok][:3]
    if rate >= 0.7 and rate >= base * 2:
        verdict = "CONSISTENT"
    elif rate <= max(base * 1.2, 0.2):
        verdict = "INCONSISTENT"
    else:
        verdict = "PARTLY CONSISTENT"
    res.update(verdict=verdict, rate=round(rate, 3), chance=round(base, 3),
               evidence=(f"{h['token']} is {phrase(h['anchor'])} in {len(ok)} of {len(occ)} uses ({rate:.0%}); "
                         f"a random position would be {base:.0%}. If it means '{h['meaning']}', this pattern is "
                         f"{'what you would expect' if verdict == 'CONSISTENT' else 'weak or contrary evidence'}."),
               counterexamples=bad)
    return res


def test_role(h, corpus, m3):
    tok = corpus.parse(h["token"])
    occ = occurrences(corpus, tok)
    role = h["role"]
    res = {"hypothesis": h, "label": "HYPOTHESIS", "occurrences": len(occ)}
    if not occ:
        res.update(verdict="UNTESTABLE", evidence=f"token {h['token']} does not occur")
        return res
    n = len(corpus.tokens)
    checks = []
    if role == "function_word":
        freq = len(occ) / n
        spread = len(set(ui * 10 // max(1, len(corpus.units)) for ui, _ in occ)) / 10
        inflected = tok in m3["_paradigm_words"]
        checks = [("frequent (>0.5% of tokens)", freq > 0.005, f"{freq:.2%}"),
                  ("spread across the text", spread >= 0.7, f"in {spread:.0%} of tenths"),
                  ("does not inflect", not inflected, "inflects" if inflected else "no ending variants")]
    elif role == "number":
        checks = [("made of number-like signs", all(c in m3["_nums"] for c in tok),
                   "yes" if all(c in m3["_nums"] for c in tok) else "contains other signs")]
    elif role == "name":
        pos = [ui for ui, _ in occ]
        span = (max(pos) - min(pos)) / max(1, len(corpus.units))
        checks = [("rare (2 to 10 uses)", 2 <= len(occ) <= 10, f"{len(occ)} uses"),
                  ("does not inflect", tok not in m3["_paradigm_words"], ""),
                  ("clustered in one passage", span <= 0.3, f"spans {span:.0%} of the text")]
    elif role in ("unit_start", "unit_end"):
        want_end = role == "unit_end"
        hit = sum(1 for ui, i in occ if (i == len(corpus.units[ui]) - 1 if want_end else i == 0))
        checks = [(f"sits at {corpus.unit_kind} {'end' if want_end else 'start'}", hit / len(occ) >= 0.6,
                   f"{hit}/{len(occ)}")]
    else:
        res.update(verdict="UNTESTABLE", evidence=f"unknown role '{role}'")
        return res
    passed = sum(1 for _, ok, _ in checks if ok)
    verdict = "CONSISTENT" if passed == len(checks) else ("INCONSISTENT" if passed == 0 else "PARTLY CONSISTENT")
    res.update(verdict=verdict, evidence="; ".join(f"{c}: {'pass' if ok else 'fail'} ({d})" for c, ok, d in checks))
    return res


def test_affix(h, corpus, m3):
    raw = h["affix"]
    is_suffix = raw.startswith("-")
    aff = corpus.parse(raw.strip("-"))
    types = Counter(corpus.tokens)
    with_aff = [w for w in types if (w.endswith(aff) if is_suffix else w.startswith(aff)) and len(w) > len(aff) + 1]
    stems = [w[:-len(aff)] if is_suffix else w[len(aff):] for w in with_aff]
    bare = [s for s in stems if s in types]
    alt = [s for s in stems if any(w != s + aff and (w.startswith(s) if is_suffix else w.endswith(s)) for w in types)]
    res = {"hypothesis": h, "label": "HYPOTHESIS", "occurrences": sum(types[w] for w in with_aff)}
    verdict = "CONSISTENT" if len(set(alt)) >= 3 else ("PARTLY CONSISTENT" if alt else "INCONSISTENT")
    res.update(verdict=verdict, evidence=(f"{len(with_aff)} word types carry it; {len(set(alt))} of their stems also "
                                          f"appear with other endings and {len(set(bare))} appear bare. That supports "
                                          f"it being a real affix. Its meaning '{h['meaning']}' cannot be verified "
                                          f"from internal evidence alone."),
               examples=[corpus.show(w) for w in with_aff[:6]])
    return res


def test_values(h, corpus, m3, trials=200):
    lang = h.get("language", "en")
    mapping = {corpus.parse(k): v for k, v in h["map"].items()}
    toks = [t for t in corpus.tokens if all(c in mapping for c in t)]
    coverage = len(toks) / max(1, len(corpus.tokens))
    res = {"hypothesis": {**h, "map": dict(list(h["map"].items())[:40])}, "label": "HYPOTHESIS"}
    if not toks:
        res.update(verdict="UNTESTABLE", evidence="the mapping does not cover any complete word")
        return res
    try:
        rate, hits = lexicon_rate(decode(toks, mapping), lang)
    except Exception:
        res.update(verdict="UNTESTABLE", evidence=f"no lexicon for language '{lang}'")
        return res
    keys, vals = list(mapping), list(mapping.values())
    rng = random.Random(0)
    null = []
    for _ in range(trials):
        v = vals[:]
        rng.shuffle(v)
        null.append(lexicon_rate(decode(toks, dict(zip(keys, v))), lang)[0])
    p = (1 + sum(n >= rate for n in null)) / (trials + 1)
    verdict = "CONSISTENT" if p < 0.01 and rate > 0.1 else ("PARTLY CONSISTENT" if p < 0.05 else "INCONSISTENT")
    res.update(verdict=verdict, rate=round(rate, 4), null_mean=round(float(np.mean(null)), 4), p_value=round(p, 4),
               evidence=(f"The mapping covers {coverage:.0%} of words and turns {rate:.0%} of them into real "
                         f"{name(lang)} words. Random reshuffles of the same values give {np.mean(null):.0%} on average "
                         f"(p = {p:.3f}). Consistent means the mapping beats chance, not that it is right."),
               examples=[w for w, _ in Counter(hits).most_common(10)])
    return res


def test(h, corpus, m3):
    kind = h.get("type")
    if kind == "gloss":
        return test_gloss(h, corpus, m3)
    if kind == "role":
        return test_role(h, corpus, m3)
    if kind == "affix":
        return test_affix(h, corpus, m3)
    if kind == "sign_values":
        return test_values(h, corpus, m3)
    return {"hypothesis": h, "label": "HYPOTHESIS", "verdict": "UNTESTABLE", "evidence": "unrecognised hypothesis type"}


def load_file(path, corpus):
    text = open(path, encoding="utf-8").read()
    if path.endswith(".json"):
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    return [h for h in (parse_line(l, corpus) for l in text.splitlines() if l.strip() and not l.startswith("#")) if h]


def interactive(corpus, m3):
    print(HELP)
    results = []
    while True:
        try:
            line = input("hypothesis> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line in ("quit", "exit", "q"):
            break
        if line == "help":
            print(HELP)
            continue
        h = parse_line(line, corpus)
        r = test(h, corpus, m3)
        results.append(r)
        print(f"[{r['label']}] {r['verdict']}: {r['evidence']}")
        for c in r.get("counterexamples", []):
            print("   counterexample:", c)
        if r.get("examples"):
            print("   examples:", ", ".join(map(str, r["examples"])))
    return results
