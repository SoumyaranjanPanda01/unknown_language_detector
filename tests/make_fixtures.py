import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).parent / "fixtures"
HERE.mkdir(exist_ok=True)
PUA = 0xE000
rng = random.Random(7)

gpl = Path("/usr/share/common-licenses/GPL-3").read_text()
words = gpl.split()[:3500]
english = " ".join(words)
english = english.replace("  ", " ")
lines, cur = [], []
for w in words:
    cur.append(w)
    if len(cur) == 14:
        lines.append(" ".join(cur))
        cur = []
(HERE / "english.txt").write_text("\n".join(lines) + "\nConfig note: api_key = sk-ant-api03-FAKEKEYFAKEKEYFAKEKEY1234\n")

(HERE / "hindi.txt").write_text(
    "भारत एक विशाल देश है और इसकी संस्कृति बहुत पुरानी है। यहाँ अनेक भाषाएँ बोली जाती हैं और हर राज्य की अपनी परंपरा है। "
    "गाँवों में आज भी खेती मुख्य काम है, जबकि शहरों में उद्योग और सेवाएँ तेज़ी से बढ़ रही हैं। "
    "सरकार ने छोटे दुकानदारों के लिए डिजिटल भुगतान को आसान बनाने की योजना शुरू की है। "
    "इस योजना से दुकानदार बिना नकद के ग्राहकों से पैसे ले सकते हैं और उनका हिसाब भी अपने आप बन जाता है। "
    "विशेषज्ञों का मानना है कि अगले पाँच वर्षों में छोटे शहरों में डिजिटल लेनदेन दोगुना हो जाएगा।\n")

letters = sorted(set("abcdefghijklmnopqrstuvwxyz"))
perm = list(range(len(letters)))
rng.shuffle(perm)
cmap = {c: chr(PUA + perm[i]) for i, c in enumerate(letters)}
clean_lines = []
for l in lines:
    t = re.sub(r"[^a-z ]", "", l.lower())
    t = re.sub(r" +", " ", t).strip()
    if t:
        clean_lines.append(t)
cipher = ["".join(cmap.get(c, c) for c in l) for l in clean_lines]
(HERE / "cipher_english.txt").write_text("\n".join(cipher) + "\n")
(HERE / "cipher_key.txt").write_text("\n".join(f"P{perm[i]:03d}={c}" for i, c in enumerate(letters)))

signs = [chr(PUA + 100 + i) for i in range(26)]
rand_lines = []
for _ in range(250):
    toks = ["".join(rng.choice(signs) for _ in range(rng.randint(2, 8))) for _ in range(rng.randint(8, 14))]
    rand_lines.append(" ".join(toks))
(HERE / "random.txt").write_text("\n".join(rand_lines) + "\n")

nouns = ["mashu", "kelor", "tiven", "barug", "solem", "dakri", "pelun", "ghoram", "nuvet", "firan",
         "lomek", "zaru", "hetin", "worak", "silum", "kapet", "yoran", "brisen", "talun", "mekor"]
verbs = ["dagi", "pelo", "suri", "kanta", "bero", "timla", "gova", "reshi"]
cases = ["", "ka", "ne", "ri", "rika"]
posts = ["to", "am"]
names = ["Aruvesh", "Tolimar", "Qenbaro"]
con = []
for i in range(420):
    if rng.random() < 0.22:
        n = rng.randint(1, 9)
        unit = rng.choice(["mashu", "tiven", "barug"])
        con.append(f"{unit} {'|' * n} {rng.choice(['holi', 'holi', 'wen'])}")
        continue
    subj = rng.choice(nouns) + rng.choice(["", "ri"])
    obj = rng.choice(nouns) + rng.choice(["ka", "rika"])
    parts = [subj]
    if rng.random() < 0.5:
        parts.append(rng.choice(nouns) + "ne")
    if rng.random() < 0.3:
        parts += [rng.choice(nouns), rng.choice(posts)]
    if 150 <= i < 190 and rng.random() < 0.6:
        parts.insert(0, names[(i // 13) % 3].lower())
    parts += [obj, rng.choice(verbs) + rng.choice(["sa", "sa", "mi"])]
    con.append(" ".join(parts))
alpha = sorted(set("".join(con).replace(" ", "")))
cmap2 = {c: chr(PUA + 200 + i) for i, c in enumerate(alpha)}
(HERE / "conlang.txt").write_text("\n".join("".join(cmap2.get(c, c) for c in l) for l in con) + "\n")
(HERE / "conlang_plain.txt").write_text("\n".join(con) + "\n")


def glyph_shape(seed):
    r = random.Random(seed)
    pts = [(r.randint(2, 20), r.randint(2, 30)) for _ in range(r.randint(3, 5))]
    extra = r.random() < 0.5
    return pts, extra


shapes = {c: glyph_shape(i * 31 + 5) for i, c in enumerate(letters)}
W, H = 1400, 1900
img = Image.new("L", (W, H), 238)
d = ImageDraw.Draw(img)
y = 60
for l in clean_lines[:48]:
    x = 60
    for w in l.split():
        if x + len(w) * 26 > W - 60:
            y += 50
            x = 60
        for ch in w:
            pts, extra = shapes[ch]
            d.line([(x + px, y + py) for px, py in pts], fill=20, width=3)
            if extra:
                d.ellipse([x + 8, y + 12, x + 14, y + 18], outline=20, width=2)
            x += 26
        x += 22
    y += 50
    if y > H - 80:
        break
arr = np.array(img).astype(float) + np.random.default_rng(1).normal(0, 6, (H, W))
Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(HERE / "manuscript.png")
print("fixtures written to", HERE)
