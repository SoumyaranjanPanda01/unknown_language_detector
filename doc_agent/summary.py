import os
import re

import regex
from collections import Counter

SENT_SPLIT = re.compile(r"(?<=[.!?।॥。！？؟])\s+|\n{2,}")


def split_sentences(text):
    parts = [re.sub(r"\s+", " ", s).strip() for s in SENT_SPLIT.split(text) if s and s.strip()]
    return [s for s in parts if 15 < len(s) < 600]


def word_weight(word, iso):
    try:
        from wordfreq import zipf_frequency
        z = zipf_frequency(word, iso)
    except Exception:
        z = 0.0
    return max(0.0, 7.5 - z)


def keywords(text, iso, n=12):
    words = [w for w in regex.findall(r"[\p{L}\p{M}]{3,}", text.lower())]
    tf = Counter(words)
    scored = {w: c * word_weight(w, iso) for w, c in tf.items() if c >= 2 or len(tf) < 50}
    return [w for w, _ in sorted(scored.items(), key=lambda x: -x[1])[:n]]


def extractive_summary(text, iso, n=5):
    sents = split_sentences(text)
    if not sents:
        return []
    tf = Counter(regex.findall(r"[\p{L}\p{M}]{2,}", text.lower()))
    weights = {w: c * word_weight(w, iso) for w, c in tf.items()}
    top = max(weights.values()) if weights else 1
    scores = []
    for i, s in enumerate(sents):
        ws = regex.findall(r"[\p{L}\p{M}]{2,}", s.lower())
        if not ws:
            continue
        score = sum(weights.get(w, 0) for w in ws) / (top * (len(ws) ** 0.5))
        if i == 0:
            score *= 1.2
        scores.append((score, i, s))
    picked = sorted(sorted(scores, reverse=True)[:n], key=lambda x: x[1])
    return [s for _, _, s in picked]


def llm_available():
    return bool(os.environ.get("SARVAM_API_KEY"))


def llm_report(text, language, out_language, want_translation=False):
    key = os.environ.get("SARVAM_API_KEY")
    if not key:
        return None
    # placeholder: wire the Sarvam chat/translate call here and return
    # {"doc_type": ..., "summary": ..., "key_points": [...], "translation": ...}
    return {"error": "Sarvam API key found but the call is a placeholder, not implemented yet"}


def guess_doc_type(text):
    t = text.lower()
    checks = [
        ("legal or licence text", ["hereby", "licence", "license", "warranty", "liability", "pursuant"]),
        ("letter or email", ["dear ", "regards", "sincerely", "subject:"]),
        ("invoice or financial record", ["invoice", "total", "amount due", "gst", "qty"]),
        ("academic or technical paper", ["abstract", "introduction", "references", "et al"]),
        ("code or configuration", ["def ", "import ", "function", "{", "};"]),
    ]
    best, hits = "general prose", 0
    for name, marks in checks:
        h = sum(t.count(m) > 0 for m in marks)
        if h > hits and h >= 2:
            best, hits = name, h
    return best
