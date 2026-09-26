from collections import Counter


def line(c):
    return f"- **[{c['label']}]** {c['text']}"


def table(rows, headers):
    if not rows:
        return "_none_"
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(str(x).replace("|", "\\|") for x in r) + " |")
    return "\n".join(out)


def header(r):
    m = r["meta"]
    pages = f", {m['pages']} page(s)" if m["pages"] else ""
    return [f"# Doc Agent report: {m['file']}", "",
            f"Generated {m['generated']}. Source of text: {m['source']}.", "",
            "Labels: **OBSERVED** = measured from the text, **INFERRED** = probable from patterns, "
            "**HYPOTHESIS** = unverified guess.", "",
            "## 1. Document summary", "",
            f"- **[OBSERVED]** Format {m['format']}, {m['bytes']} bytes{pages}."] + [line(c) for c in m["condition"]]


def render_tier1(r, out_language):
    d = r["detection"]
    L = header(r)
    L.append(f"- **[INFERRED]** Document type: {r['doc_type']}.")
    L += ["", "## 2. Language identification", "",
          f"- **[OBSERVED]** Language: **{r['language']}** ({r['iso']}). {d['reason'].capitalize()}."]
    if d["candidates"]:
        L.append("- **[OBSERVED]** Model scores: " + ", ".join(
            f"{c['language']} {c['confidence']:.2f}" for c in d["candidates"]) + ".")
    if d.get("coverage") is not None:
        L.append(f"- **[OBSERVED]** {d['coverage']:.0%} of word tokens are in the {r['language']} vocabulary.")
    scripts = ", ".join(f"{k.title()} {v:.0%}" for k, v in list(d["script"]["scripts"].items())[:3])
    L.append(f"- **[OBSERVED]** Script: {scripts}; direction {d['script']['unicode_direction']}.")
    llm = r.get("llm") or {}
    L += ["", "## 3. Content summary", ""]
    if llm.get("summary"):
        L.append(f"_Summary in {out_language}, written by the configured language model._")
        L += ["", llm["summary"]]
    else:
        if llm.get("error"):
            L.append(f"- **[OBSERVED]** {llm['error']}; fell back to extractive summary.")
        L.append(f"_Extractive summary: the most informative sentences, quoted in the original language. "
                 f"Set SARVAM_API_KEY to get an abstractive summary in {out_language}._")
        L.append("")
        L += [f"> {s}" for s in r["summary_extract"]] or ["_no full sentences found_"]
    L += ["", "## 4. Key points", ""]
    if llm.get("key_points"):
        L += [f"- {p}" for p in llm["key_points"]]
    else:
        L.append("- **[INFERRED]** Key terms (frequent here, rare in general usage): " + ", ".join(r["keywords"]) + ".")
    if llm.get("translation"):
        L += ["", f"## 5. Translation into {out_language}", "", llm["translation"]]
    L += ["", "## Configuration", "",
          "Summaries and translation use an LLM only when a key is configured in the environment:", "",
          "```", "SARVAM_API_KEY=", "```", "",
          "Keys are read from the environment at run time and are never written into reports."]
    return "\n".join(L)


def render_tier2(r):
    d, m1, m2, m3, m4 = r["detection"], r["m1"], r["m2"], r["m3"], r["m4"]
    L = header(r)
    L.append(f"- **[OBSERVED]** Tier 2 triggered: {d['reason']}.")
    if r.get("sample"):
        L += ["", "First units as transcribed:", "", "```"] + [s[:160] for s in r["sample"]] + ["```"]

    L += ["", "## 2. Script identification (Method 1)", ""] + [line(c) for c in m1["claims"]]
    if r.get("glyphs") and r["glyphs"].get("sheet"):
        L += ["", "![Sign sheet](sign_sheet.png)", "",
              "_Sign sheet: one example of each sign cluster. If two labels are the same sign, re-run with a "
              "higher --glyph-threshold; if one label mixes two signs, lower it._"]
    L += ["", "Sign frequencies (top 20):", "", table(m1["sign_frequencies"][:20], ["sign", "count"])]

    L += ["", "## 3. Statistical profile (Method 2)", ""] + [line(c) for c in m2["claims"]]
    L += ["", "![Zipf plot](zipf.png)", ""]
    z = m2["zipf"]
    if z:
        L.append(f"_Zipf curve: the blue line is this document's rank-frequency curve on log-log axes. A straight "
                 f"line with slope near 1 is the natural-language pattern; this one has slope {z['slope']} "
                 f"(R^2 {z['r2']}). The orange line is the same signs shuffled, which is what meaningless strings "
                 f"of the same signs would look like._")
    L += ["", "Most frequent words:", "", table(m2["top_words"][:15], ["word", "count"]),
          "", "Most frequent sign pairs:", "", table(m2["top_bigrams"][:10], ["pair", "count"])]

    L += ["", "## 4. Structural findings (Method 3)", ""] + [line(c) for c in m3["claims"]]
    if m3["signatures"]:
        L += ["", "Ending sets shared by several stems:", "",
              table([(", ".join("-" + x if x else "(bare)" for x in k), len(v) if isinstance(v, list) else v, ", ".join(s))
                     for k, s, v in m3["signatures"][:6]], ["ending set", "stems", "examples"])]

    L += ["", "## 5. Comparative analysis and hypotheses (Method 4)", ""] + [line(c) for c in m4["claims"]]
    for dcp in m4["decipherment"]:
        pairs = list(dcp["mapping"].items())[:30]
        L += ["", f"Sign-value mapping tried against {dcp['lang']} (top 30 signs):", "",
              "```", " ".join(f"{a}={b}" for a, b in pairs), "```"]
    if r["hypotheses"]:
        L += ["", "### Human-supplied hypotheses", ""]
        for t in r["hypotheses"]:
            h = t["hypothesis"]
            desc = {"gloss": lambda: f"{h['token']} = '{h['meaning']}' near {h.get('anchor')}",
                    "role": lambda: f"{h['token']} is a {h['role']}",
                    "affix": lambda: f"{h['affix']} = '{h['meaning']}'",
                    "sign_values": lambda: f"sign values in {h.get('language')} ({len(h['map'])} signs)"}.get(
                h.get("type"), lambda: str(h))()
            L.append(f"- **[HYPOTHESIS] {t['verdict']}**: {desc}. {t['evidence']}")
            for c in t.get("counterexamples", []):
                L.append(f"    - counterexample: `{c[:120]}`")
    L += ["", "The interactive checker tests any hypothesis against the whole text: "
              "`python -m doc_agent hypo FILE --interactive`."]

    counts = Counter(c["label"] for m in (m1, m2, m3, m4) for c in m["claims"])
    strong_lex = any(x["rate"] - x["control"] >= 0.1 and (x["lift"] or 0) >= 1.5 for x in m4["lexical"])
    strong_dec = any(x["hit_rate"] >= 0.2 and x["hit_rate"] >= 2 * max(x["control_rate"], 0.02) for x in m4["decipherment"])
    confirmed = [t for t in r["hypotheses"] if t["verdict"] == "CONSISTENT"]
    L += ["", "## 6. Confidence statement", "",
          f"This report makes {counts['OBSERVED']} observed, {counts['INFERRED']} inferred and "
          f"{counts['HYPOTHESIS']} hypothetical claims.", "",
          f"- **Known (measured):** sign inventory ({m1['distinct_signs']} signs), word segmentation "
          f"({m1.get('total_signs')} signs in {m2['tokens']} words), frequency and entropy profile, "
          f"recurring endings and positions.",
          f"- **Inferred:** the text is {m2['verdict']}; the writing system is most likely {'an' if m1['system_type'][0] in 'aeiou' else 'a'} {m1['system_type']}"
          + (f"; it looks {m3['affix_type']}" if m3["affix_type"] else "")
          + (f" and {m3['order_hint']}" if m3["order_hint"] else "") + "."]
    if strong_dec or strong_lex:
        L.append("- **Candidate anchor found:** at least one comparison beats its shuffled control by a clear "
                 "margin (see section 5). It is still a hypothesis until a human checks it word by word.")
    else:
        L.append("- **Unresolvable without an anchor:** the meaning of words and sentences. No related known "
                 "language, bilingual text or picture context was found, so no translation can be offered. "
                 "Linear A, the Indus script, Rongorongo and the Voynich manuscript are stuck at exactly this point.")
    if confirmed:
        L.append(f"- {len(confirmed)} human hypothesis(es) were consistent with the whole text. Consistent is not proven.")

    L += ["", "## 7. Next steps", ""]
    steps = []
    if r.get("glyphs"):
        steps.append("Check the sign sheet and re-run with a different --glyph-threshold if signs were merged or split; "
                     "or type a clean transliteration (one sign per character) and analyse that file.")
    if m2["tokens"] < 1500:
        steps.append(f"Provide more text. {m2['tokens']} words is small; statistics settle above a few thousand words.")
    steps.append("Provide any related text: another inscription in the same script, a text from the same place and "
                 "period, or a bilingual. A single bilingual line is worth more than every statistic here.")
    steps.append("Provide illustrations, the object's material and find-spot. Pictures next to text (as on Linear B "
                 "tablets with commodity ideograms) anchor word meanings.")
    if m3["number_context"]:
        steps.append(f"Test whether {m3['number_context'][0][0]} names a counted item: "
                     f"`gloss {m3['number_context'][0][0]} = <item> near numbers`.")
    if m3["function_words"]:
        steps.append(f"Test a function-word guess: `role {m3['function_words'][0][0]} function_word`.")
    if not m4["decipherment"] and m1["distinct_signs"] <= 45:
        close = [c[0] for c in m4["closest_profiles"][:2]]
        steps.append(f"Run the sign-value search against the closest profiles ({', '.join(close)}) with "
                     f"--decipher and a language code (for example --decipher en,de).")
    if m1["distinct_signs"] > 45:
        steps.append("Supply a partial syllabic grid as a `values` hypothesis; the checker compares it against "
                     "random reshuffles of the same values.")
    L += [f"{i}. {s}" for i, s in enumerate(steps, 1)]
    L += ["", "No credentials are stored in this report. If an LLM is configured, its key lives only in the "
              "environment (`SARVAM_API_KEY=`)."]
    return "\n".join(L)


def render(r, out_language="English"):
    return render_tier1(r, out_language) if r["tier"] == 1 else render_tier2(r)
