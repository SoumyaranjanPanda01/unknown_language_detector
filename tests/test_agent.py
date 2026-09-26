import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT))

from doc_agent import analyze
from doc_agent.corpus import Corpus
from doc_agent import hypotheses


def run(name, **kw):
    out = tempfile.mkdtemp()
    result, corpus, _ = analyze(str(FIX / name), out_dir=out, use_llm=False, **kw)
    return result, corpus, Path(out)


def test_known_languages():
    r, _, out = run("english.txt")
    assert r["tier"] == 1 and r["iso"] == "en"
    assert "FAKEKEY" not in (out / "report.md").read_text()
    r, _, _ = run("hindi.txt")
    assert r["tier"] == 1 and r["iso"] == "hi"


def test_random_is_flagged():
    r, _, _ = run("random.txt")
    assert r["tier"] == 2 and r["m2"]["verdict"] == "random-like"
    assert r["m3"]["signatures"] == []


def test_cipher_is_language_like_and_deciphered():
    r, _, out = run("cipher_english.txt", decipher_langs=["en"])
    assert r["m2"]["verdict"] == "language-like"
    d = r["m4"]["decipherment"][0]
    key = dict(l.split("=") for l in (FIX / "cipher_key.txt").read_text().split())
    acc = sum(d["mapping"].get(k) == v for k, v in key.items()) / len(key)
    assert acc > 0.8 and d["hit_rate"] > 3 * d["control_rate"]
    report = (out / "report.md").read_text()
    for section in range(1, 8):
        assert f"## {section}." in report


def test_conlang_structure():
    r, corpus, _ = run("conlang.txt")
    m3 = r["m3"]
    assert m3["affix_type"] == "suffixing"
    assert m3["numeral_signs"], "tally signs should be found"
    assert m3["number_context"], "unit words next to numbers should be found"
    total_word = m3["number_context"][0][0]
    res = hypotheses.test({"type": "gloss", "token": total_word, "meaning": "x", "anchor": "numbers"}, corpus,
                          {**m3, "_nums": {corpus.parse(s) for s, _, _ in m3["numeral_signs"]}})
    assert res["verdict"] == "CONSISTENT"


def test_unknown_language_in_known_script():
    r, _, _ = run("conlang_plain.txt")
    assert r["tier"] == 2


def test_manuscript_image():
    if not (FIX / "manuscript.png").exists():
        return
    r, _, out = run("manuscript.png", decipher_langs=["en"])
    assert r["tier"] == 2 and r["glyphs"]["n_sign_types"] <= 30
    assert (out / "sign_sheet.png").exists()
    assert r["m4"]["decipherment"][0]["hit_rate"] > 0.5


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
