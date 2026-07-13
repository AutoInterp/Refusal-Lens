"""Report v5 comply rates + the Phase-A inspect gate.

--judged   : per-class comply table (next to Tejas's v3/v4); writes v5_report.md.
--inspect  : dump each smoke (class, base, attack head, response head) so a HUMAN
             confirms every class is well-formed before the full run.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REF = {"m2s_multiturn (v3)": 60, "nested_fiction (v3)": 36, "narrative_sandwich (v3)": 2}


def _rows(records):
    return [r for r in records if r.get("kind") == "attack" and r.get("judge") in ("COMPLY", "REFUSE")]


def comply_table(records) -> dict:
    comply, total = defaultdict(int), defaultdict(int)
    for r in _rows(records):
        total[r["class"]] += 1
        if r["judge"] == "COMPLY":
            comply[r["class"]] += 1
    return {c: {"comply": comply[c], "total": total[c],
                "rate": round(100 * comply[c] / total[c], 1) if total[c] else 0.0}
            for c in total}


def inspect_lines(generations, head: int = 400) -> list[str]:
    out = []
    for g in generations:
        out.append("=" * 72)
        out.append(f"[{g['class']}] base_id={g.get('base_id')}")
        out.append(f"  ATTACK  : {g.get('attack_text', g.get('prompt_text',''))[:head]!r}")
        out.append(f"  RESPONSE: {g.get('response','')[:head]!r}")
    return out


def _load(p):
    d = json.loads(Path(p).read_text())
    return d["generations"] if "generations" in d else d["records"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", type=Path)
    ap.add_argument("--generations", type=Path)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--sweep-judged", type=Path)
    ap.add_argument("--head", type=int, default=400)
    args = ap.parse_args()

    if args.generations and args.inspect:
        print("\n".join(inspect_lines(_load(args.generations), head=args.head)))

    if args.judged:
        recs = _load(args.judged)
        table = comply_table(recs)
        lines = ["# dataset_v5 comply rates", "", "| class | comply/total | rate |", "|---|---|---|"]
        for c, v in table.items():
            lines.append(f"| {c} | {v['comply']}/{v['total']} | {v['rate']}% |")
        lines += ["", "## Reference (Tejas)"] + [f"- {k}: {v}%" for k, v in REF.items()]
        report = "\n".join(lines)
        print(report)
        out = args.judged.parent / "v5_report.md"
        out.write_text(report + "\n")
        print(f"\n[report] wrote {out}")

    if args.sweep_judged:
        recs = _load(args.sweep_judged)
        comply, total = defaultdict(int), defaultdict(int)
        for r in recs:
            if r.get("judge") in ("COMPLY", "REFUSE"):
                total[r["sweep_k"]] += 1
                comply[r["sweep_k"]] += r["judge"] == "COMPLY"
        print("K -> comply%: " + ", ".join(
            f"{k}:{round(100*comply[k]/total[k],1)}" for k in sorted(total)))


if __name__ == "__main__":
    main()
