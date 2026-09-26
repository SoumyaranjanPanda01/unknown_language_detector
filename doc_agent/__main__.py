import argparse
import sys

from .agent import analyze
from .corpus import Corpus


def main(argv=None):
    p = argparse.ArgumentParser(prog="doc_agent", description="Read a document; decipher structure if the language is unknown.")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="full report")
    a.add_argument("file")
    a.add_argument("--out", default="doc_agent_report")
    a.add_argument("--lang-out", default="English", help="language for Tier 1 summaries")
    a.add_argument("--translate", action="store_true", help="Tier 1: ask the LLM for a full translation")
    a.add_argument("--no-llm", action="store_true")
    a.add_argument("--force-tier2", action="store_true", help="run decipherment methods even on a known language")
    a.add_argument("--decipher", default="", help="comma-separated language codes for sign-value search, e.g. en,de")
    a.add_argument("--iters", type=int, default=6000)
    a.add_argument("--hypotheses", help=".txt (one command per line) or .json list of hypotheses")
    a.add_argument("--glyph-threshold", type=float, default=0.45)

    h = sub.add_parser("hypo", help="test hypotheses against a document")
    h.add_argument("file")
    h.add_argument("--interactive", action="store_true")
    h.add_argument("--hypotheses")
    h.add_argument("--glyph-threshold", type=float, default=0.45)
    h.add_argument("--out", default="doc_agent_report")

    args = p.parse_args(argv)
    if args.cmd == "analyze":
        langs = [x.strip() for x in args.decipher.split(",") if x.strip()]
        result, _, out = analyze(args.file, out_dir=args.out, out_language=args.lang_out, force_tier2=args.force_tier2,
                                 translate=args.translate, decipher_langs=langs, hypotheses_path=args.hypotheses,
                                 glyph_threshold=args.glyph_threshold, use_llm=not args.no_llm, iters=args.iters)
        tier = result["tier"]
        what = result.get("language") if tier == 1 else result["m2"]["verdict"]
        print(f"Tier {tier} ({what}). Report: {out / 'report.md'}")
        return 0

    from . import hypotheses
    result, corpus, _ = analyze(args.file, out_dir=args.out, force_tier2=True, use_llm=False,
                                glyph_threshold=args.glyph_threshold, hypotheses_path=args.hypotheses)
    for t in result["hypotheses"]:
        print(f"[{t['label']}] {t['verdict']}: {t['evidence']}")
    if args.interactive:
        hypotheses.interactive(corpus, result["m3"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
