# Doc Agent

Upload a document, get a report on it.

- **Known language:** detects the language, writes a short summary, pulls key terms and guesses the document type.
- **Unknown language or script:** switches to decipherment mode and reports what can be measured from the text itself (sign inventory, statistics, word structure, comparisons with known languages), with every claim labelled `OBSERVED`, `INFERRED` or `HYPOTHESIS`.

Everything runs zero-shot on a single document. No training data, no fine-tuning.

## Quick start

```
pip install -r requirements.txt
streamlit run app.py
```

Open the link Streamlit prints, upload a file, click **Analyze**. The report shows on the page and can be downloaded as `report.md` or `report.json`.

Images and scanned PDFs need Tesseract for OCR:

```
sudo apt install tesseract-ocr        # Ubuntu / Debian
brew install tesseract                # macOS
```

The first Tier 2 run builds reference profiles for about 40 languages and caches them in `~/.cache/doc_agent/`. It takes a minute once, then it is instant.

## Supported inputs

`.txt` `.md` `.pdf` `.docx` `.png` `.jpg` `.jpeg` `.tif` `.tiff` `.bmp`

PDFs without a text layer are treated as scanned pages. Images go through OCR first; if OCR cannot read them as a known script, the glyphs are clustered into a sign inventory instead.

## How it decides

| Tier | When | What you get |
|---|---|---|
| 1 | Language model (lingua) is confident and most words are found in that language's vocabulary, or the script belongs to a single language (Odia, Tamil, Thai, Georgian, ...) | Language, summary, key terms, document type |
| 2 | Private-use or unassigned characters, glyph images OCR cannot read, undeciphered Unicode scripts (Linear A, Cypro-Minoan), or known letters whose words are not in any vocabulary | The 7-section decipherment report |

You can force Tier 2 on any document with the sidebar checkbox or `--force-tier2`.

## The Tier 2 report

1. **Document summary**: format, size, condition, what can and cannot be said about era.
2. **Script identification (Method 1)**: sign inventory, estimated full inventory size, word separator, writing direction, likely writing system type (alphabet, syllabary, logographic). For images, a sign sheet shows one example of each sign.
3. **Statistical profile (Method 2)**: sign and word frequencies, Shannon entropy, Zipf curve with fit, and a shuffled-text control so you can see whether the order carries structure or is random.
4. **Structural findings (Method 3)**: productive endings and beginnings, ending sets shared by many stems (the Kober triplet idea), function-word candidates, words tied to line start or end, word-order hints, number-like signs, words that sit next to numbers, possible names.
5. **Comparative analysis (Method 4)**: closest statistical profiles among reference languages, vocabulary look-alikes when the letters are readable, and an optional sign-value search that tries to map unknown signs to letters of a chosen language.
6. **Confidence statement**: what is known, what is inferred, what cannot be resolved without an anchor.
7. **Next steps**: concrete things to try or to provide.

Every finding that could appear by chance is compared against a shuffled version of the same text, so the report does not treat coincidence as a discovery.

## Sidebar options

| Option | What it does |
|---|---|
| Summary language | Language for Tier 1 summaries (used once the Sarvam call is wired) |
| Force decipherment analysis | Runs Tier 2 even on a known language |
| Sign-value search languages | Language codes for the automatic sign-to-letter search, e.g. `en,de`. Only runs for scripts with 45 signs or fewer |
| Glyph clustering threshold | For images. Raise it if one sign got split into two labels, lower it if two different signs got merged |
| Hypotheses | Your own guesses, one per line, tested against the whole text |

## Testing hypotheses

Sign labels look like `G004` (image glyphs) or `P012` (private-use characters). Join them with `-` for a word, e.g. `G004-G011-G002`.

```
gloss G004-G011 = total near numbers       word sits next to numbers more than chance?
gloss G004-G011 = king near line_start     or near line_end, or near another word
role G002 function_word                    frequent, spread out, never inflected?
role G007-G001 name                        rare, uninflected, clustered in one passage?
role G009 number                           made only of number-like signs?
affix -G003 = plural                       does the ending attach to many stems?
values G000=e G001=t G002=a in en          does this mapping produce real English words beyond chance?
```

Each result comes back as `CONSISTENT`, `PARTLY CONSISTENT`, `INCONSISTENT` or `UNTESTABLE`, with the evidence and counterexamples. A consistent result stays a hypothesis; it means the text does not contradict it, not that it is proven.

## Sarvam API (placeholder)

The LLM step is a placeholder. Copy `.env.example` to `.env` or export the variable:

```
SARVAM_API_KEY=
```

Then wire the request in `llm_report()` in `doc_agent/summary.py`. It should return a dict with `doc_type`, `summary`, `key_points` and optionally `translation`. Until then, Tier 1 falls back to an extractive summary (the most informative sentences, quoted in the original language).

The key is read from the environment only. It is never written to reports, and any credential-like strings found inside uploaded documents are replaced with placeholders before analysis.

## Command line

```
python -m doc_agent analyze FILE --out report_dir
python -m doc_agent analyze FILE --force-tier2 --decipher en,de --hypotheses my_guesses.txt
python -m doc_agent hypo FILE --interactive
```

Other flags: `--lang-out`, `--translate`, `--no-llm`, `--iters`, `--glyph-threshold`.

## Project layout

```
app.py                 Streamlit UI
doc_agent/
  agent.py             pipeline: load, detect, pick tier, write report
  loader.py            txt / pdf / docx / image loading
  detect.py            script profile and language detection
  summary.py           Tier 1 summary, key terms, Sarvam placeholder
  glyphs.py            OCR and glyph clustering for images
  corpus.py            tokenising, separators, units
  method1.py           script and sign inventory
  method2.py           frequencies, entropy, Zipf, shuffled control
  method3.py           morphology, positions, numbers, names
  method4.py           reference comparison and sign-value search
  references.py        reference language profiles (wordfreq)
  hypotheses.py        hypothesis tester and interactive mode
  report.py            markdown report
  redact.py            credential redaction
tests/
  make_fixtures.py     builds the test documents
  test_agent.py        end-to-end checks
```

## Tests

```
python tests/make_fixtures.py
python tests/test_agent.py
```

The fixtures are English, Hindi, English written in an invented alphabet, random signs, an invented suffixing language with tally numbers, and a rendered handwritten page with made-up glyphs. The checks confirm that known languages go to Tier 1, random text is flagged as random, the invented language's endings and numbers are found, and the sign-value search recovers the hidden alphabet from both the text and the image.

## Limits

- Structure can be measured without a key; meaning usually cannot. Without a related known language, a bilingual text or pictures, the report says so and stops at hypotheses.
- The sign-value search assumes one sign per letter, so it only runs for 45 signs or fewer. Syllabaries need a partial grid supplied as a `values` hypothesis.
- Glyph clustering works best on clean, evenly spaced writing. Damaged or cursive manuscripts may need a hand-typed transliteration.
- Boustrophedon and vertical layouts are not detected automatically.
- OCR only reads the languages installed for Tesseract (`DOC_AGENT_OCR_LANGS`, default `eng`).
