import json
import os
from datetime import datetime
from pathlib import Path

from . import method1, method2, method3, method4, hypotheses
from .claims import obs, inf, hyp
from .corpus import Corpus
from .detect import detect_language
from .glyphs import transcribe, ocr_known
from .loader import load
from .redact import redact
from .summary import extractive_summary, keywords, llm_report, llm_available, guess_doc_type


def condition_notes(doc, text, glyph_info):
    notes = []
    if doc["kind"] == "image":
        import cv2
        import numpy as np
        img = cv2.imread(doc["images"][0], cv2.IMREAD_GRAYSCALE)
        if img is not None:
            h, w = img.shape
            notes.append(obs(f"Image {w}x{h}px, contrast (std of grey levels) {img.std():.0f}. "
                             f"{'Low contrast, faded or damaged surface likely.' if img.std() < 35 else 'Contrast is adequate for glyph extraction.'}"))
        notes.append(inf("Era cannot be measured from pixels. Material, palaeography and provenance need a human "
                         "or lab examination (radiocarbon, ink analysis)."))
    else:
        bad = text.count("�")
        notes.append(obs(f"Digital text, {len(text)} characters, {text.count(chr(10)) + 1} lines"
                         + (f", {bad} unreadable replacement characters" if bad else ", no encoding damage") + "."))
        notes.append(inf("Era is not visible in a digital transcription; the original object would be needed."))
    for n in doc["notes"]:
        notes.append(obs(n))
    return notes


def tier1(doc, text, det, out_language, translate, use_llm):
    iso = det.get("iso", "en")
    llm = llm_report(text, det["language"], out_language, translate) if use_llm and llm_available() else None
    rep = {
        "tier": 1, "language": det["language"], "iso": iso, "detection": det,
        "doc_type": (llm or {}).get("doc_type") or guess_doc_type(text),
        "summary_extract": extractive_summary(text, iso),
        "keywords": keywords(text, iso),
        "llm": llm,
    }
    return rep


def tier2(doc, text, det, out_dir, glyph_info, decipher_langs, hyp_items, iters):
    corpus = Corpus(text, glyph_mode=glyph_info is not None)
    m1 = method1.run(corpus, det, glyph_info)
    m2 = method2.run(corpus, plot_path=str(out_dir / "zipf.png"))
    m3 = method3.run(corpus, m2)
    m4 = method4.run(corpus, det, m1, m2, m3, decipher_langs=decipher_langs, iters=iters)
    tests = [hypotheses.test(h, corpus, m3) for h in (hyp_items or [])]
    return corpus, {"tier": 2, "detection": det, "m1": m1, "m2": m2, "m3": m3, "m4": m4, "hypotheses": tests}


def analyze(path, out_dir="doc_agent_report", out_language="English", force_tier2=False, translate=False,
            decipher_langs=None, hypotheses_path=None, glyph_threshold=0.45, use_llm=True, iters=6000):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = load(path)
    glyph_info = None
    text = doc["text"]
    source = "text layer"
    if doc["kind"] == "image":
        ocr = ocr_known(doc["images"], os.environ.get("DOC_AGENT_OCR_LANGS", "eng"))
        det = detect_language(ocr["text"]) if ocr and ocr["mean_conf"] >= 70 and ocr["n_words"] >= 10 else None
        if det and det["known"] and not force_tier2:
            text, source = ocr["text"], f"OCR (mean confidence {ocr['mean_conf']:.0f})"
        else:
            glyph_info = transcribe(doc["images"], threshold=glyph_threshold, out_sheet=str(out_dir / "sign_sheet.png"))
            text, source = glyph_info["text"], "glyph clustering"
    text, n_secrets = redact(text)
    if n_secrets:
        doc["notes"].append(f"{n_secrets} credential-like strings were found and replaced with placeholders before analysis.")

    det = detect_language(text) if glyph_info is None else {
        "known": False, "reason": "OCR could not read the image as a known script",
        "script": {"scripts": {"UNKNOWN GLYPHS": 1.0}, "letters": glyph_info["n_glyph_tokens"],
                   "unknown_share": 1.0, "unicode_direction": None}, "candidates": []}

    meta = {"file": doc["name"], "format": doc["format"], "bytes": doc["bytes"], "pages": doc["pages"],
            "source": source, "generated": datetime.now().isoformat(timespec="seconds"),
            "condition": condition_notes(doc, text, glyph_info)}

    hyp_items = None
    if det["known"] and not force_tier2:
        result = tier1(doc, text, det, out_language, translate, use_llm)
        corpus = None
    else:
        if hypotheses_path:
            probe = Corpus(text, glyph_mode=glyph_info is not None)
            hyp_items = hypotheses.load_file(hypotheses_path, probe)
        corpus, result = tier2(doc, text, det, out_dir, glyph_info, decipher_langs, hyp_items, iters)
        result["glyphs"] = glyph_info
        result["sample"] = [" ".join(corpus.show(t) for t in u) for u in corpus.units[:3]]
    result["meta"] = meta

    from .report import render
    md = render(result, out_language)
    md, _ = redact(md)
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    safe = json.loads(json.dumps(result, default=lambda o: sorted(o) if isinstance(o, set) else str(o)))
    for k in ("m3",):
        if k in safe:
            for private in ("_func", "_nums", "_paradigm_words"):
                safe[k].pop(private, None)
    js, _ = redact(json.dumps(safe, ensure_ascii=False, indent=2))
    (out_dir / "report.json").write_text(js, encoding="utf-8")
    return result, corpus, out_dir
