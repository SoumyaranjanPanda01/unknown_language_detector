from collections import Counter

from .claims import obs, inf, hyp


def writing_system_guess(n_types, est_types, signs_per_word):
    n = est_types or n_types
    if n <= 40:
        kind = "alphabet or abjad"
        why = f"about {n} distinct signs; alphabets and abjads use roughly 20 to 40"
    elif n <= 100:
        kind = "syllabary (possibly with some logograms)"
        why = (f"about {n} distinct signs; syllabaries sit around 40 to 100 "
               f"(Linear B has about 87 syllabic signs, Cherokee 85)")
    elif n <= 1000:
        kind = "logo-syllabic system"
        why = f"about {n} distinct signs; mixed systems like cuneiform or Maya glyphs sit in the hundreds"
    else:
        kind = "logographic system"
        why = f"over 1000 distinct signs, typical of logographic writing like Chinese"
    if signs_per_word is not None:
        why += f"; average word length is {signs_per_word:.1f} signs"
    return kind, why


def run(corpus, detection, glyph_info=None):
    signs = Counter(corpus.signs)
    total = sum(signs.values())
    n_types = len(signs)
    f1 = sum(1 for c in signs.values() if c == 1)
    f2 = sum(1 for c in signs.values() if c == 2)
    chao1 = n_types + (f1 * f1 / (2 * f2) if f2 else f1 * (f1 - 1) / 2)
    unseen = f1 / total if total else 0
    wl = [len(t) for t in corpus.tokens]
    spw = sum(wl) / len(wl) if wl and corpus.separator != "none" else None

    claims = []
    scripts = detection["script"]["scripts"]
    if glyph_info:
        claims.append(obs(f"Image transcribed by glyph clustering: {glyph_info['n_glyph_tokens']} glyph tokens on "
                          f"{glyph_info['n_lines']} lines grouped into {glyph_info['n_sign_types']} sign types "
                          f"(clustering threshold {glyph_info['cluster_threshold']}). Labels G000, G001... are ranked "
                          f"by frequency. A sign sheet image lets a human check merges and splits."))
    else:
        desc = ", ".join(f"{k.title()} {v:.0%}" for k, v in list(scripts.items())[:4]) or "none"
        claims.append(obs(f"Unicode script make-up of letters: {desc}."))
    claims.append(obs(f"{total} sign tokens, {n_types} distinct signs; {f1} signs occur only once."))
    if total:
        claims.append(inf(f"Estimated full inventory about {chao1:.0f} signs (Chao1). Unseen-sign probability "
                          f"{unseen:.1%}, so {'the inventory is close to complete' if unseen < 0.01 else 'more signs likely exist beyond this sample'}."))
    claims.append(obs(f"Word separator: {corpus.separator}. Units for positional analysis: {corpus.unit_kind}s "
                      f"({len(corpus.units)} units)."))
    for n in corpus.notes:
        claims.append(inf(n))

    single = Counter(t for t in corpus.tokens if len(t) == 1)
    divider_like = [(s, c) for s, c in single.items() if c >= 5 and c / signs[s] >= 0.8 and corpus.separator != "none"]
    if divider_like:
        shown = ", ".join(f"{corpus.show(s)} ({c}x)" for s, c in sorted(divider_like, key=lambda x: -x[1])[:6])
        claims.append(inf(f"Signs that almost always stand alone as a whole word: {shown}. These are punctuation, "
                          f"logograms, numerals or determinatives."))

    if glyph_info:
        if glyph_info["direction"]:
            claims.append(inf(f"Writing direction {glyph_info['direction']}: {glyph_info['direction_reason']}."))
        else:
            claims.append(inf(f"Writing direction not determined: {glyph_info['direction_reason']}."))
        claims.append(hyp("Boustrophedon and vertical layouts were not tested automatically; check the sign sheet "
                          "for mirrored sign pairs that alternate line by line."))
    elif detection["script"]["unicode_direction"]:
        claims.append(obs(f"Direction {detection['script']['unicode_direction']} from the Unicode bidi class of the characters."))
    else:
        claims.append(inf("Direction cannot be read from Private Use characters; it is taken as the order the "
                          "characters were keyed in, which may not match the original."))

    kind, why = writing_system_guess(n_types, round(chao1), spw)
    claims.append(inf(f"Writing system type: {kind} ({why})."))
    return {
        "total_signs": total, "distinct_signs": n_types, "hapax_signs": f1,
        "chao1_inventory": round(chao1, 1), "unseen_mass": round(unseen, 4),
        "signs_per_word": None if spw is None else round(spw, 2),
        "sign_frequencies": [(corpus.show(s), c) for s, c in signs.most_common()],
        "system_type": kind, "claims": claims,
    }
