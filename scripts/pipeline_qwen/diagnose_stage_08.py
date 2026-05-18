"""
Diagnostic for Stage 08 ablation outputs.

Reads `<run-dir>/08_ablation/ablation_results.json` and prints:
  - per-(ablation × jb_class) recovery counts (coherent and total)
  - bare break counts
  - flip-direction histogram
  - baseline-vs-ablated response similarity statistics
  - activation-audit summary (if present in 08_ablation/activation_audit.json)

Useful for sanity-checking a smoke run before committing GPU time to a full
50-prompt run, and for re-reading old runs without rerunning the model.

Usage:
    PYTHONPATH=src python3 scripts/pipeline/diagnose_stage_08.py \\
        --run-dir data/results/pipeline_runs/run_20260422_015552
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: F401

JB_CLASSES = ("jb_roleplay", "jb_fiction", "jb_analytical", "jb_completion", "jb_cognitive_reframe")


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose a Stage 08 ablation run")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--positions", default="all",
                   help="Which positions-mode block to inspect (default: all)")
    p.add_argument("--show-snapshots", action="store_true",
                   help="Print baseline vs ablated text snippets for the first prompt")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = run_dir / "08_ablation"
    if not out_dir.exists():
        sys.exit(f"08_ablation/ not found at {out_dir}")

    R = json.loads((out_dir / "ablation_results.json").read_text())
    pos_mode = args.positions
    n_results = len(R["results"])
    print(f"=== Stage 08 diagnostic: {run_dir.name} ({n_results} prompts, positions={pos_mode}) ===")
    print()

    # --- baseline COMPLY denominators per JB class ---
    per_class_comply = defaultdict(int)
    bare_refuse = 0
    for row in R["results"]:
        for cond in JB_CLASSES:
            if row["baseline"].get(cond, {}).get("cls") == "COMPLY":
                per_class_comply[cond] += 1
        if row["baseline"].get("bare", {}).get("cls") == "REFUSE":
            bare_refuse += 1
    print(f"Baseline JB COMPLY counts: {dict(per_class_comply)}  total={sum(per_class_comply.values())}")
    print(f"Baseline bare REFUSE: {bare_refuse}/{n_results}")
    print()

    # --- per-ablation × per-class recovery + bare break ---
    abl_names = list(R["results"][0]["ablations"].keys())
    print(f"=== Per-ablation × per-class recovery (positions={pos_mode}) ===")
    print(f"{'ablation':<42} {'class':<22} {'denom':>6} {'rec':>10} {'coh_rec':>10}")
    for ab_name in abl_names:
        for cls in JB_CLASSES:
            denom = recovered = coh = 0
            for row in R["results"]:
                bl = row["baseline"].get(cls)
                ab = row["ablations"][ab_name].get(pos_mode, {}).get(cls)
                if not bl or not ab:
                    continue
                if bl["cls"] == "COMPLY":
                    denom += 1
                    if ab["cls"] == "REFUSE":
                        recovered += 1
                        if ab.get("coherent"):
                            coh += 1
            rate = recovered / denom * 100 if denom else 0
            crate = coh / denom * 100 if denom else 0
            print(f"  {ab_name[:40]:<42} {cls:<22} {denom:>6} {recovered}/{denom} ({rate:>4.0f}%) {coh}/{denom} ({crate:>3.0f}%)")
        print()

    print(f"=== Per-ablation bare REFUSE → COMPLY (break) ===")
    for ab_name in abl_names:
        denom = broke = coh = 0
        for row in R["results"]:
            bl = row["baseline"].get("bare")
            ab = row["ablations"][ab_name].get(pos_mode, {}).get("bare")
            if not bl or not ab:
                continue
            if bl["cls"] == "REFUSE":
                denom += 1
                if ab["cls"] == "COMPLY":
                    broke += 1
                    if ab.get("coherent"):
                        coh += 1
        rate = broke / denom * 100 if denom else 0
        print(f"  {ab_name[:42]:<44} bare break: {broke}/{denom} ({rate:.0f}%)  coherent: {coh}/{denom}")
    print()

    # --- flip direction histogram ---
    flip_counter = Counter()
    flip_coh = {"both_coherent": 0, "ablated_incoherent": 0, "baseline_incoherent": 0}
    sim_buckets = {"identical (≥0.95)": 0, "similar (0.6–0.95)": 0, "different (<0.6)": 0}
    n_total_cells = 0
    for row in R["results"]:
        for cond, bl in row["baseline"].items():
            for ab_name in abl_names:
                ab = row["ablations"][ab_name].get(pos_mode, {}).get(cond)
                if not ab:
                    continue
                n_total_cells += 1
                # similarity
                s = difflib.SequenceMatcher(None, bl.get("response", ""), ab.get("response", "")).ratio()
                if s >= 0.95:
                    sim_buckets["identical (≥0.95)"] += 1
                elif s >= 0.6:
                    sim_buckets["similar (0.6–0.95)"] += 1
                else:
                    sim_buckets["different (<0.6)"] += 1
                # flip
                if bl["cls"] != ab["cls"]:
                    flip_counter[(bl["cls"], ab["cls"])] += 1
                    if not ab.get("coherent", True):
                        flip_coh["ablated_incoherent"] += 1
                    elif not bl.get("coherent", True):
                        flip_coh["baseline_incoherent"] += 1
                    else:
                        flip_coh["both_coherent"] += 1

    total_flips = sum(flip_counter.values())
    print(f"=== Flip direction histogram ({total_flips} flips across {n_total_cells} cells) ===")
    for (a, b), n in sorted(flip_counter.items()):
        print(f"  {a:>8} → {b:<8}: {n}")
    print(f"  coherence: {flip_coh}")
    print()

    print(f"=== Baseline-vs-ablated response similarity ({n_total_cells} cells) ===")
    for label, n in sim_buckets.items():
        pct = n / n_total_cells * 100 if n_total_cells else 0
        print(f"  {label:<22}: {n:>5}  ({pct:>5.1f}%)")
    if sim_buckets["identical (≥0.95)"] > 0.5 * n_total_cells:
        print("  ⚠️  >50% of cells are nearly identical — intervention may not be landing.")
    print()

    # --- activation audit summary (if present) ---
    audit_path = out_dir / "activation_audit.json"
    if audit_path.exists():
        A = json.loads(audit_path.read_text())
        print("=== Activation audit (from Stage 02 attribution data) ===")
        print(f"{'ablation':<42} {'class group':<10} {'top-50 hit':>12} {'mean |attr|':>12}")
        for ab_name, audit in A.get("per_ablation", {}).items():
            for grp, stats in audit["by_class_group"].items():
                print(f"  {ab_name[:40]:<42} {grp:<10} "
                      f"{stats['mean_top50_hit_rate']*100:>10.2f}% "
                      f"{stats['mean_attr_per_prompt']:>12.5f}")
            print()
    else:
        print("(no activation_audit.json; rerun Stage 08 main with the updated script)")
    print()

    # --- response snapshots ---
    if args.show_snapshots and R["results"]:
        print("=== Response snapshots (prompt 0) ===")
        row = R["results"][0]
        for cond in ("bare", "jb_fiction", "jb_analytical"):
            bl = row["baseline"].get(cond)
            if not bl:
                continue
            print(f"\n[{cond}] baseline ({bl['cls']}):")
            print(f"   {bl['response'][:200]}")
            for ab_name in abl_names:
                ab = row["ablations"][ab_name].get(pos_mode, {}).get(cond)
                if not ab:
                    continue
                s = difflib.SequenceMatcher(None, bl["response"], ab["response"]).ratio()
                marker = " ← FLIPPED" if bl["cls"] != ab["cls"] else ""
                print(f"\n[{cond}] via {ab_name} ({ab['cls']}, sim={s:.2f}){marker}:")
                print(f"   {ab['response'][:200]}")


if __name__ == "__main__":
    main()
