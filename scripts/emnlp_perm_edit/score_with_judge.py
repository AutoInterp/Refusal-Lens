"""Re-score saved completions with the LLM-judge refusal scorer.

Replaces the keyword `classify_response` with `refusal_llm_judge.judge_refusal`
(True = REFUSAL). Adds `judge_refusal`/`judge_class` to every record, writes a
new *_judged.json, and prints keyword-vs-judge comparison + the bare flip table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from refusal_llm_judge import judge_refusal  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", type=Path, required=True)
    p.add_argument("--model", default=None, help="judge model id (default: package DEFAULT_MODEL)")
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    kw = dict(batch_size=args.batch_size, on_parse_error=None, return_raw=True)
    if args.model:
        kw["model"] = args.model

    for path in args.inputs:
        d = json.loads(path.read_text())
        recs = d["records"]
        items = [{"prompt": r["prompt_text"], "response": r["response"]} for r in recs]
        print(f"\n[{path.name}] scoring {len(items)} completions with the LLM judge ...")
        labels, raw = judge_refusal(items, **kw)
        n_unparsed = sum(1 for l in labels if l is None)
        for r, l, rawtext in zip(recs, labels, raw):
            r["judge_refusal"] = l  # True=REFUSAL, None=unparseable
            r["judge_class"] = ("UNPARSED" if l is None else ("REFUSE" if l else "COMPLY"))
            r["judge_raw"] = rawtext[:80]
        out = path.with_name(path.stem + "_judged.json")
        out.write_text(json.dumps(d, indent=2))
        print(f"  unparsed: {n_unparsed}/{len(labels)}  -> wrote {out.name}")

        # keyword vs judge agreement
        agree = sum(1 for r in recs if r["judge_class"] == r["classification"])
        flip = sum(1 for r in recs if r["classification"] == "COMPLY" and r["judge_class"] == "REFUSE")
        print(f"  keyword vs judge agreement: {agree}/{len(recs)} "
              f"({100*agree/len(recs):.0f}%); kw=COMPLY but judge=REFUSE: {flip}")


if __name__ == "__main__":
    main()
