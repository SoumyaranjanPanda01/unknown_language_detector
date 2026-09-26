import re

import regex
import unicodedata
from collections import Counter
from functools import lru_cache

MULTI_WORD_SCRIPTS = {"CJK", "OLD", "LINEAR", "CANADIAN", "NEW", "TAI", "MEROITIC", "EGYPTIAN", "OL", "KHITAN"}
UNDECIPHERED_SCRIPTS = {"LINEAR A", "CYPRO-MINOAN", "KHITAN SMALL", "PROTO-ELAMITE"}
SCRIPT_LANGUAGE = {
    "ORIYA": ("Odia", "or"), "TAMIL": ("Tamil", "ta"), "TELUGU": ("Telugu", "te"),
    "KANNADA": ("Kannada", "kn"), "MALAYALAM": ("Malayalam", "ml"), "GUJARATI": ("Gujarati", "gu"),
    "GURMUKHI": ("Punjabi", "pa"), "SINHALA": ("Sinhala", "si"), "THAI": ("Thai", "th"),
    "LAO": ("Lao", "lo"), "KHMER": ("Khmer", "km"), "MYANMAR": ("Burmese", "my"),
    "GEORGIAN": ("Georgian", "ka"), "ARMENIAN": ("Armenian", "hy"), "ETHIOPIC": ("Amharic", "am"),
    "TIBETAN": ("Tibetan", "bo"), "HANGUL": ("Korean", "ko"), "HIRAGANA": ("Japanese", "ja"),
    "KATAKANA": ("Japanese", "ja"), "HEBREW": ("Hebrew", "he"), "CHEROKEE": ("Cherokee", "chr"),
    "OL CHIKI": ("Santali", "sat"), "THAANA": ("Dhivehi", "dv"), "MONGOLIAN": ("Mongolian", "mn"),
    "LINEAR B": ("Mycenaean Greek", "gmy"), "CUNEIFORM": ("Akkadian or Sumerian", "akk"),
    "EGYPTIAN HIEROGLYPH": ("Ancient Egyptian", "egy"), "OLD ITALIC": ("Etruscan, Oscan, Umbrian or Old Latin", "ett"),
    "GOTHIC": ("Gothic", "got"), "RUNIC": ("Old Norse or other Germanic", "non"), "OGHAM": ("Primitive Irish", "pgl"),
}
PUA_RANGES = [(0xE000, 0xF8FF), (0xF0000, 0xFFFFD), (0x100000, 0x10FFFD)]


def is_pua(ch):
    o = ord(ch)
    return any(a <= o <= b for a, b in PUA_RANGES)


def script_of(ch):
    if is_pua(ch):
        return "PRIVATE-USE"
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return "UNASSIGNED"
    parts = name.split()
    if parts[0] in MULTI_WORD_SCRIPTS and len(parts) > 1:
        return parts[0] + " " + parts[1]
    return parts[0]


def is_content_char(ch):
    if is_pua(ch):
        return True
    cat = unicodedata.category(ch)
    return cat[0] in ("L", "M", "N", "S", "C") and cat not in ("Cc", "Cf", "Zs")


def script_profile(text):
    counts = Counter()
    rtl = ltr = 0
    for ch in text:
        if ch.isspace():
            continue
        cat = unicodedata.category(ch)
        if is_pua(ch) or cat[0] in ("L", "M") or cat == "Cn":
            counts[script_of(ch)] += 1
            if is_pua(ch):
                continue
            bidi = unicodedata.bidirectional(ch)
            if bidi in ("R", "AL"):
                rtl += 1
            elif bidi == "L":
                ltr += 1
    total = sum(counts.values()) or 1
    shares = {k: round(v / total, 4) for k, v in counts.most_common()}
    unknown = shares.get("PRIVATE-USE", 0) + shares.get("UNASSIGNED", 0)
    if rtl + ltr == 0:
        direction = None
    else:
        direction = "right-to-left" if rtl > ltr else "left-to-right"
    return {"scripts": shares, "letters": total if counts else 0, "unknown_share": round(unknown, 4),
            "unicode_direction": direction}


@lru_cache(maxsize=1)
def lingua_detector():
    try:
        from lingua import LanguageDetectorBuilder
    except ImportError:
        return None
    return LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()


def lingua_guess(text, top=3):
    det = lingua_detector()
    if det is None:
        return []
    sample = text[:20000]
    vals = det.compute_language_confidence_values(sample)
    out = []
    for v in vals[:top]:
        iso = v.language.iso_code_639_1.name.lower()
        out.append({"language": v.language.name.title(), "iso": iso, "confidence": round(v.value, 4)})
    return out


def words_of(text):
    return regex.findall(r"[\p{L}\p{M}]+(?:['’][\p{L}\p{M}]+)?", text.lower())


def vocab_coverage(text, iso):
    import logging
    import warnings
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        return None
    words = words_of(text)[:5000]
    if not words:
        return None
    types = Counter(words)
    logging.getLogger("wordfreq").setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zipf_frequency("the", iso)
            hit = sum(c for w, c in types.items() if zipf_frequency(w, iso) > 1.5)
    except (LookupError, ValueError):
        return None
    return round(hit / sum(types.values()), 4)


def detect_language(text, min_conf=0.45, min_cov=0.5):
    prof = script_profile(text)
    result = {"script": prof, "candidates": [], "coverage": None, "known": False, "reason": ""}
    if prof["letters"] < 20:
        result["reason"] = "too few letters to run language detection"
        return result
    if prof["unknown_share"] > 0.2:
        result["reason"] = "characters sit in Private Use or unassigned Unicode ranges, no known script"
        return result
    dominant = next(iter(prof["scripts"]), None)
    result["dominant_script"] = dominant
    if dominant in UNDECIPHERED_SCRIPTS:
        result["reason"] = f"{dominant.title()} is encoded in Unicode but has never been deciphered"
        return result
    cands = lingua_guess(text)
    result["candidates"] = cands
    top = cands[0] if cands else None
    if dominant in SCRIPT_LANGUAGE and (top is None or top["confidence"] < min_conf):
        name, iso = SCRIPT_LANGUAGE[dominant]
        result.update(known=True, language=name, iso=iso,
                      reason=f"identified from the {dominant.title()} script, which is tied to {name}")
        return result
    if top is None:
        result["reason"] = "no language model matched"
        return result
    cov = vocab_coverage(text, top["iso"])
    result["coverage"] = cov
    if top["confidence"] < min_conf:
        result["reason"] = f"best guess {top['language']} only at confidence {top['confidence']}"
        return result
    if cov is not None and cov < min_cov:
        result["reason"] = (f"script reads like {top['language']} but only {cov:.0%} of words are in its "
                            f"vocabulary, so the letters are known but the language is not")
        return result
    result["known"] = True
    result["language"] = top["language"]
    result["iso"] = top["iso"]
    result["reason"] = ("language model and vocabulary check agree" if cov is not None else
                        "language model only; no vocabulary list exists for this language to double-check it")
    return result
