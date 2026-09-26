import tempfile
from pathlib import Path

import streamlit as st

from doc_agent import analyze

st.set_page_config(page_title="Doc Agent", layout="wide")
st.title("Doc Agent")
st.caption("Upload a document. Known languages get a summary; unknown scripts get a decipherment report.")

with st.sidebar:
    out_lang = st.text_input("Summary language", "English")
    force = st.checkbox("Force decipherment analysis (Tier 2)")
    decipher = st.text_input("Sign-value search languages (e.g. en,de)", "")
    threshold = st.slider("Glyph clustering threshold (images)", 0.2, 0.8, 0.45, 0.05)
    hyp_text = st.text_area("Hypotheses (one per line, optional)",
                            placeholder="gloss G004-G011 = total near numbers\nrole G002 function_word")

up = st.file_uploader("Document", type=["txt", "md", "pdf", "docx", "png", "jpg", "jpeg", "tif", "tiff", "bmp"])

if up and st.button("Analyze", type="primary"):
    work = Path(tempfile.mkdtemp())
    src = work / up.name
    src.write_bytes(up.getvalue())
    hyp_path = None
    if hyp_text.strip():
        hyp_path = work / "hypotheses.txt"
        hyp_path.write_text(hyp_text)
    langs = [x.strip() for x in decipher.split(",") if x.strip()]
    with st.spinner("Analysing..."):
        result, _, out = analyze(str(src), out_dir=str(work / "report"), out_language=out_lang, force_tier2=force,
                                 decipher_langs=langs, hypotheses_path=str(hyp_path) if hyp_path else None,
                                 glyph_threshold=threshold)
    raw = (out / "report.md").read_text(encoding="utf-8")
    md = raw
    if result["tier"] == 1:
        st.success(f"Tier 1: {result['language']}")
    else:
        st.warning(f"Tier 2: unknown language ({result['m2']['verdict']})")
    for img in ("sign_sheet.png", "zipf.png"):
        md = md.replace(f"![{'Sign sheet' if img == 'sign_sheet.png' else 'Zipf plot'}]({img})", f"[[{img}]]")
    for part in md.split("[["):
        if "]]" in part:
            img, rest = part.split("]]", 1)
            st.image(str(out / img))
            st.markdown(rest)
        else:
            st.markdown(part)
    c1, c2 = st.columns(2)
    c1.download_button("Download report.md", raw.encode(), file_name="report.md")
    c2.download_button("Download report.json", (out / "report.json").read_bytes(), file_name="report.json")
