import math
import re
import unicodedata
from collections import Counter

from .glyphs import label

SENT_END = set(".!?।॥。！？؟;·")


def entropy(counter):
    total = sum(counter.values())
    if not total:
        return 0.0
    return -sum(c / total * math.log2(c / total) for c in counter.values() if c)


def is_punct(ch):
    return unicodedata.category(ch)[0] == "P" or ch in SENT_END


def side_entropy(text, ch):
    left, right = Counter(), Counter()
    for i, c in enumerate(text):
        if c == ch:
            if i > 0:
                left[text[i - 1]] += 1
            if i + 1 < len(text):
                right[text[i + 1]] += 1
    return entropy(left) + entropy(right)


def find_char_separator(text):
    body = [c for c in text if c != "\n"]
    counts = Counter(body)
    best = None
    for ch, n in counts.most_common(6):
        if n < 10:
            continue
        doubled = text.count(ch + ch) / n
        score = side_entropy(text, ch)
        if doubled < 0.02 and (best is None or score > best[1]):
            best = (ch, score)
    if best is None:
        return None
    others = [side_entropy(text, c) for c, n in counts.most_common(15) if c != best[0] and n >= 10]
    if others and best[1] > max(others) * 1.05:
        return best[0]
    return None


class Corpus:
    def __init__(self, text, glyph_mode=False):
        self.raw = text
        self.glyph_mode = glyph_mode
        self.notes = []
        body = text.replace("\r", "")
        non_nl = [c for c in body if c != "\n"]
        ws_share = sum(c.isspace() for c in non_nl) / max(1, len(non_nl))
        self.has_case = any(c.isupper() for c in body) and any(c.islower() for c in body)

        if ws_share >= 0.03:
            self.separator = "whitespace"
            sep = None
        else:
            sep = find_char_separator(body)
            if sep:
                self.separator = f"sign {self.show(sep)}"
                self.notes.append(f"no spaces; {self.show(sep)} behaves like a word divider "
                                  f"(varied neighbours on both sides, never doubled)")
            else:
                self.separator = "none"
                self.notes.append("no word separator found; each sign is treated as one token")

        punct_ends = sum(1 for c in body if c in SENT_END or (unicodedata.category(c) == "Po"))
        self.sentence_marks = punct_ends >= 3 and not glyph_mode

        lines = [l for l in body.split("\n") if l.strip()]
        units = []
        if self.sentence_marks:
            joined = " ".join(lines)
            chunks = re.split(r"(?<=[.!?।॥。！？؟])\s+", joined)
            self.unit_kind = "sentence"
        else:
            chunks = lines
            self.unit_kind = "line"
        for ch in chunks:
            toks = self.tokenize(ch, sep)
            if toks:
                units.append(toks)
        if len(units) <= 1 and self.tokens_count(units) > 60:
            flat = units[0] if units else []
            units = [flat[i:i + 12] for i in range(0, len(flat), 12)]
            self.unit_kind = "12-token window"
            self.notes.append("no line or sentence structure; positional stats use arbitrary 12-token windows")
        self.units = units
        self.tokens = [t for u in units for t in u]
        self.signs = [c for t in self.tokens for c in t]

    @staticmethod
    def tokens_count(units):
        return sum(len(u) for u in units)

    def tokenize(self, chunk, sep):
        if self.separator == "whitespace":
            parts = chunk.split()
        elif sep:
            parts = chunk.replace(" ", "").split(sep)
        else:
            parts = [c for c in chunk if not c.isspace()]
        out = []
        for p in parts:
            p = "".join(c for c in p if not is_punct(c) or self.glyph_mode)
            p = p.strip()
            if p:
                out.append(p.lower() if self.has_case else p)
        return out

    def show(self, s):
        if self.glyph_mode or any(0xE000 <= ord(c) <= 0xF8FF for c in s):
            return "-".join(label(c) for c in s)
        return s

    def parse(self, s):
        if re.fullmatch(r"([GP]\d{3})(-[GP]\d{3})*", s):
            base = {"G": 0xF0000, "P": 0xE000}
            return "".join(chr(base[x[0]] + int(x[1:])) for x in s.split("-"))
        return s.lower() if self.has_case else s

    def cased_words(self):
        return re.findall(r"[^\W\d_]+", self.raw)
