"""Stage 08 recovery drilldown — comply-baseline JB cases per ablation.

Filters Stage 08's per-prompt records (08_ablation/ablation_results.json) down
to JB conditions where the baseline complied (so recovery is measurable), then
emits a per-prompt view of which ablations flipped the response back to REFUSE
— alongside the baseline and ablated response text for qualitative review.

The aggregate `recovery_rate` in `ablation_summary.json` is *already* correctly
restricted to baseline-comply denominators (n_recovered_refusal /
n_baseline_comply), so this script does not change any headline metric. What it
adds is per-prompt visibility — which complies got flipped, which didn't, and
what the responses actually said — so you can read the qualitative outcome by
hand and spot patterns that aggregate rates wash out.

Inputs:
    <run-dir>/08_ablation/ablation_results.json

Outputs to <run-dir>/08_ablation/:
    recovery_drilldown.json  — by_class[cls].per_ablation[abl][pos_mode] with
                               per-prompt records sorted by flipped_to_refuse desc
    recovery_drilldown.csv   — flat one-row-per-(prompt × ablation × pos_mode)
                               for spreadsheet review

Usage:
    PYTHONPATH=src python3 scripts/pipeline/08_recovery_drilldown.py \\
        --run-dir <run-dir>

    # Only certain ablations / position modes
    python3 ... --ablations universal_refusal_core,jb_fiction_specific_vs_ctrl
    python3 ... --positions all
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

JB_CLASSES = ("analytical", "cognitive_reframe", "completion", "fiction", "roleplay")
RESPONSE_PREVIEW_CHARS = 160


def parse_args():
    p = argparse.ArgumentParser(description="Stage 08 comply-baseline recovery drilldown")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--ablations", type=str, default=None,
                   help="Comma-separated ablation names to include (default: all in the file).")
    p.add_argument("--positions", type=str, default=None,
                   help="Comma-separated position modes to include (default: all that ran).")
    p.add_argument("--out-stem", type=str, default="recovery_drilldown",
                   help="Output filename stem under 08_ablation/ (default: recovery_drilldown).")
    p.add_argument("--no-csv", action="store_true",
                   help="Skip CSV emission (JSON only).")
    return p.parse_args()


def load_ablation_results(run_dir: Path) -> dict:
    path = run_dir / "08_ablation" / "ablation_results.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Did Stage 08 run?", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text())


def collect_drilldown(
    data: dict,
    abl_filter: set[str] | None,
    pos_filter: set[str] | None,
) -> dict:
    """For each baseline-COMPLY JB case, walk the ablations and record the outcome."""
    by_class: dict[str, dict] = {
        cls: {"n_baseline_comply": 0, "per_ablation": {}} for cls in JB_CLASSES
    }

    for record in data.get("results", []):
        prompt_id = record.get("prompt_id")
        topic = record.get("topic")
        base = record.get("base")
        baseline = record.get("baseline", {})
        ablations = record.get("ablations", {})

        for cls in JB_CLASSES:
            cond = f"jb_{cls}"
            base_entry = baseline.get(cond)
            if not base_entry or base_entry.get("cls") != "COMPLY":
                continue
            by_class[cls]["n_baseline_comply"] += 1
            base_resp = base_entry.get("response", "")
            base_coherent = base_entry.get("coherent", True)

            for abl_name, abl_block in ablations.items():
                if abl_filter and abl_name not in abl_filter:
                    continue
                for pos_mode, per_cond in abl_block.items():
                    if pos_mode == "n_features":
                        continue
                    if pos_filter and pos_mode not in pos_filter:
                        continue
                    abl_entry = per_cond.get(cond)
                    if not abl_entry:
                        continue

                    abl_cls = abl_entry.get("cls")
                    abl_resp = abl_entry.get("response", "")
                    abl_coherent = abl_entry.get("coherent", True)
                    flipped = (abl_cls == "REFUSE")
                    coverage = abl_entry.get("coverage", {})

                    abl_node = by_class[cls]["per_ablation"].setdefault(abl_name, {})
                    pm_node = abl_node.setdefault(pos_mode, {
                        "n_seen": 0,
                        "n_flipped_to_refuse": 0,
                        "n_unchanged_comply": 0,
                        "n_incoherent_ablated": 0,
                        "recovery_rate": 0.0,
                        "coherent_recovery_rate": 0.0,
                        "prompts": [],
                    })
                    pm_node["n_seen"] += 1
                    if flipped:
                        pm_node["n_flipped_to_refuse"] += 1
                    else:
                        pm_node["n_unchanged_comply"] += 1
                    if not abl_coherent:
                        pm_node["n_incoherent_ablated"] += 1
                    pm_node["prompts"].append({
                        "prompt_id": prompt_id,
                        "topic": topic,
                        "base": base,
                        "baseline_cls": "COMPLY",
                        "baseline_coherent": base_coherent,
                        "baseline_response": base_resp,
                        "ablated_cls": abl_cls,
                        "ablated_coherent": abl_coherent,
                        "ablated_response": abl_resp,
                        "flipped_to_refuse": flipped,
                        "coverage": {
                            "frac_in_top_k": coverage.get("frac_in_top_k"),
                            "n_in_top_k": coverage.get("n_in_top_k"),
                            "n_features": coverage.get("n_features"),
                            "low_coverage": coverage.get("low_coverage"),
                            "sum_abs_attribution": coverage.get("sum_abs_attribution"),
                        },
                    })

    for cls_node in by_class.values():
        for abl_node in cls_node["per_ablation"].values():
            for pm_node in abl_node.values():
                seen = pm_node["n_seen"]
                if seen:
                    pm_node["recovery_rate"] = round(
                        pm_node["n_flipped_to_refuse"] / seen, 4
                    )
                    coherent_seen = seen - pm_node["n_incoherent_ablated"]
                    coherent_flips = sum(
                        1 for r in pm_node["prompts"]
                        if r["flipped_to_refuse"] and r["ablated_coherent"]
                    )
                    pm_node["coherent_recovery_rate"] = (
                        round(coherent_flips / coherent_seen, 4)
                        if coherent_seen else 0.0
                    )
                pm_node["prompts"].sort(
                    key=lambda r: (not r["flipped_to_refuse"], r["prompt_id"])
                )
    return by_class


def emit_csv(by_class: dict, out_path: Path) -> None:
    fields = [
        "jb_class", "ablation", "pos_mode", "prompt_id", "topic",
        "baseline_cls", "ablated_cls", "flipped_to_refuse",
        "ablated_coherent", "coverage_frac", "low_coverage",
        "baseline_response_preview", "ablated_response_preview",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cls, cls_node in by_class.items():
            for abl_name, abl_node in cls_node["per_ablation"].items():
                for pos_mode, pm_node in abl_node.items():
                    for r in pm_node["prompts"]:
                        cov = r.get("coverage", {})
                        w.writerow({
                            "jb_class": cls,
                            "ablation": abl_name,
                            "pos_mode": pos_mode,
                            "prompt_id": r["prompt_id"],
                            "topic": r.get("topic", ""),
                            "baseline_cls": r["baseline_cls"],
                            "ablated_cls": r["ablated_cls"],
                            "flipped_to_refuse": r["flipped_to_refuse"],
                            "ablated_coherent": r["ablated_coherent"],
                            "coverage_frac": cov.get("frac_in_top_k"),
                            "low_coverage": cov.get("low_coverage"),
                            "baseline_response_preview": (r["baseline_response"] or "")[:RESPONSE_PREVIEW_CHARS].replace("\n", " "),
                            "ablated_response_preview": (r["ablated_response"] or "")[:RESPONSE_PREVIEW_CHARS].replace("\n", " "),
                        })


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    abl_filter = set(args.ablations.split(",")) if args.ablations else None
    pos_filter = set(args.positions.split(",")) if args.positions else None

    data = load_ablation_results(run_dir)
    by_class = collect_drilldown(data, abl_filter, pos_filter)

    n_records = sum(
        len(pm["prompts"])
        for cls in by_class.values()
        for abl in cls["per_ablation"].values()
        for pm in abl.values()
    )
    n_complies = sum(cls["n_baseline_comply"] for cls in by_class.values())
    ablations_seen = sorted({a for cls in by_class.values() for a in cls["per_ablation"]})
    pos_modes_seen = sorted({
        pm for cls in by_class.values()
        for abl in cls["per_ablation"].values()
        for pm in abl
    })

    out = {
        "metadata": {
            "source": str(run_dir / "08_ablation" / "ablation_results.json"),
            "filter": "baseline.cls == 'COMPLY' on jb_* conditions only",
            "n_baseline_comply_total": n_complies,
            "n_drilldown_records": n_records,
            "ablations_included": ablations_seen,
            "position_modes_included": pos_modes_seen,
            "jb_classes": list(JB_CLASSES),
        },
        "by_class": by_class,
    }

    out_dir = run_dir / "08_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.out_stem}.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {json_path}  "
          f"({n_records} records across {n_complies} comply baselines, "
          f"{len(ablations_seen)} ablation(s) × {len(pos_modes_seen)} pos_mode(s))")

    if not args.no_csv:
        csv_path = out_dir / f"{args.out_stem}.csv"
        emit_csv(by_class, csv_path)
        print(f"Wrote {csv_path}")

    print()
    print("Recovery rates per (jb_class × ablation × pos_mode):")
    print(f"  {'jb_class':>20s} × {'ablation':<42s} [{'mode':>7s}]   "
          f"recovery   coherent_recovery   flipped/seen")
    for cls, cls_node in by_class.items():
        cb = cls_node["n_baseline_comply"]
        if not cb:
            print(f"  {cls:>20s}   (0 baseline complies — recovery not measurable)")
            continue
        for abl_name, abl_node in cls_node["per_ablation"].items():
            for pos_mode, pm_node in abl_node.items():
                rr = pm_node["recovery_rate"]
                cr = pm_node["coherent_recovery_rate"]
                fl = pm_node["n_flipped_to_refuse"]
                seen = pm_node["n_seen"]
                print(
                    f"  {cls:>20s} × {abl_name:<42s} [{pos_mode:>7s}]   "
                    f"{rr:>6.1%}     {cr:>6.1%}             {fl}/{seen}"
                )

    print()
    print("DONE!")


if __name__ == "__main__":
    main()
