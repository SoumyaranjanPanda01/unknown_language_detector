import math
import random
from collections import Counter

import numpy as np

from .claims import obs, inf, hyp
from .corpus import entropy

BLUE, ORANGE, INK, MUTED, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"


def cond_entropy(tokens):
    pairs, firsts = Counter(), Counter()
    for t in tokens:
        seq = "^" + t + "$"
        for a, b in zip(seq, seq[1:]):
            pairs[(a, b)] += 1
            firsts[a] += 1
    total = sum(pairs.values())
    if not total:
        return 0.0
    return -sum(c / total * math.log2(c / firsts[a]) for (a, _), c in pairs.items())


def zipf_fit(counts):
    freqs = np.array(sorted(counts.values(), reverse=True), dtype=float)
    keep = freqs >= 2
    if keep.sum() < 5:
        return None
    r = np.log10(np.arange(1, len(freqs) + 1)[keep])
    f = np.log10(freqs[keep])
    slope, icpt = np.polyfit(r, f, 1)
    pred = slope * r + icpt
    ss_res = ((f - pred) ** 2).sum()
    ss_tot = ((f - f.mean()) ** 2).sum() or 1
    return {"slope": round(-slope, 3), "intercept": round(icpt, 3), "r2": round(1 - ss_res / ss_tot, 3),
            "fitted_ranks": int(keep.sum())}


def shuffled_tokens(tokens, seed):
    rng = random.Random(seed)
    signs = [c for t in tokens for c in t]
    rng.shuffle(signs)
    out, i = [], 0
    for t in tokens:
        out.append("".join(signs[i:i + len(t)]))
        i += len(t)
    return out


def ngrams(tokens, n):
    c = Counter()
    for t in tokens:
        for i in range(len(t) - n + 1):
            c[t[i:i + n]] += 1
    return c


def plot_zipf(word_counts, shuffled_counts, fit, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    real = sorted(word_counts.values(), reverse=True)
    shuf = sorted(shuffled_counts.values(), reverse=True)
    ax.plot(range(1, len(real) + 1), real, color=BLUE, lw=2, label="This document")
    ax.plot(range(1, len(shuf) + 1), shuf, color=ORANGE, lw=2, label="Same signs, order shuffled")
    if fit:
        r = np.arange(1, len(real) + 1)
        ax.plot(r, 10 ** fit["intercept"] * r ** (-fit["slope"]), color=MUTED, lw=1, ls="--",
                label=f"Power-law fit, slope {fit['slope']}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Word rank", color=MUTED)
    ax.set_ylabel("Word frequency", color=MUTED)
    ax.set_title("Rank-frequency curve (Zipf plot)", color=INK, loc="left", fontsize=11)
    ax.grid(True, which="major", color="#e4e3df", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bdbcb6")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def run(corpus, plot_path=None):
    toks = corpus.tokens
    signs = Counter(corpus.signs)
    words = Counter(toks)
    h1 = entropy(signs)
    h_cond = cond_entropy(toks)
    h_word = entropy(words)
    fit = zipf_fit(words)

    shuf_runs = [shuffled_tokens(toks, s) for s in range(5)]
    h_cond_shuf = float(np.mean([cond_entropy(t) for t in shuf_runs]))
    types_shuf = float(np.mean([len(set(t)) for t in shuf_runs]))
    fit_shuf = zipf_fit(Counter(shuf_runs[0]))
    gap = (h_cond_shuf - h_cond) / h_cond_shuf if h_cond_shuf else 0
    ttr = len(words) / max(1, len(toks))

    claims = [
        obs(f"{len(toks)} word tokens, {len(words)} word types (type/token ratio {ttr:.2f}); "
            f"mean word length {np.mean([len(t) for t in toks]):.2f} signs."),
        obs(f"Sign entropy H1 = {h1:.2f} bits (max possible for {len(signs)} signs is "
            f"{math.log2(max(2, len(signs))):.2f}). Conditional entropy of the next sign given the previous one "
            f"= {h_cond:.2f} bits, versus {h_cond_shuf:.2f} bits for the same signs in shuffled order."),
        obs(f"Word entropy = {h_word:.2f} bits."),
    ]
    if fit:
        claims.append(obs(f"Zipf fit on the {fit['fitted_ranks']} words seen at least twice: slope {fit['slope']}, "
                          f"R^2 {fit['r2']}. Natural language usually gives a slope near 1 with a tight straight line."))
    else:
        claims.append(obs("Too few repeated words to fit a Zipf curve."))
    claims.append(obs(f"Shuffled control produces {types_shuf:.0f} word types instead of {len(words)}"
                      + (f", Zipf slope {fit_shuf['slope']}" if fit_shuf else "") + "."))

    zipf_ok = fit and 0.7 <= fit["slope"] <= 1.4 and fit["r2"] >= 0.85
    if gap >= 0.08 and zipf_ok and types_shuf > len(words) * 1.15:
        verdict = "language-like"
        claims.append(inf(f"Sign order is far from random (conditional entropy {gap:.0%} lower than shuffled) and "
                          f"words repeat in a Zipf-like way. This is the fingerprint of natural language, "
                          f"or of a simple cipher or code over a natural language, which looks the same."))
    elif gap < 0.03:
        verdict = "random-like"
        claims.append(inf("Sign order is about as predictable as a shuffled version of itself. "
                          "This looks like random or meaningless sign strings, or a strong cipher."))
    else:
        verdict = "structured, unclear"
        claims.append(inf(f"Sign order has some structure (conditional entropy {gap:.0%} below shuffled) but the "
                          f"word statistics are not a clean Zipf pattern. Possible short text, list or "
                          f"accounting record, non-linguistic notation, or a segmentation problem."))
    claims.append(hyp("Zipf conformity alone does not prove language: random typing with a space key also gives a "
                      "Zipf-like curve (Miller, 1957). The shuffled comparison above is the stronger test."))

    if plot_path:
        plot_zipf(words, Counter(shuf_runs[0]), fit, plot_path)
    return {
        "tokens": len(toks), "types": len(words), "ttr": round(ttr, 4),
        "h1": round(h1, 3), "h_cond": round(h_cond, 3), "h_cond_shuffled": round(h_cond_shuf, 3),
        "structure_gap": round(gap, 4), "h_word": round(h_word, 3), "zipf": fit, "zipf_shuffled": fit_shuf,
        "verdict": verdict,
        "top_words": [(corpus.show(w), c) for w, c in words.most_common(25)],
        "top_bigrams": [(corpus.show(w), c) for w, c in ngrams(toks, 2).most_common(15)],
        "top_trigrams": [(corpus.show(w), c) for w, c in ngrams(toks, 3).most_common(15)],
        "plot": plot_path, "claims": claims,
    }
