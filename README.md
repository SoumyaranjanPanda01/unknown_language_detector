# Doc Agent

Upload a document and get a report. Known languages get language ID plus a summary. Unknown scripts get the 4-method decipherment report (script inventory, statistics and Zipf, morphology and positions, comparison and hypotheses), every claim labelled OBSERVED, INFERRED or HYPOTHESIS.

## Run

    pip install -r requirements.txt
    streamlit run app.py

Images need Tesseract installed for OCR (`apt install tesseract-ocr`).

CLI:

    python -m doc_agent analyze FILE --out report_dir [--decipher en,de] [--hypotheses h.txt] [--force-tier2]
    python -m doc_agent hypo FILE --interactive

## Sarvam API

Placeholder only. Set `SARVAM_API_KEY=` (see `.env.example`) and wire the call in `llm_report()` in `doc_agent/summary.py`. Without it Tier 1 uses an extractive summary.

## Tests

    python tests/make_fixtures.py
    python tests/test_agent.py
