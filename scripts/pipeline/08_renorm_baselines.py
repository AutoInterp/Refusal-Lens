"""
Stage 08 baseline renormalization (Option B from PAPER_OUTLINES_v1.md).
=======================================================================

The Stage 08 ablation sweeps were run with `--skip-baseline`, which causes
fresh baseline regeneration at `--max-new-tokens=80`. Stage 06 baselines were
generated at `--max-new-tokens=200` and are the bulletproof Phase-0 reference
(50/50 bare REFUSE, 250/250 ctrl REFUSE). Cross-run baseline drift across
{Stage 06 H100 200-tok, Stage 08 H100 80-tok, Stage 08 4090 80-tok} is
documented in `REPORT_run_20260430_023247.md` § 9.7.5.

This script post-hoc reclassifies an existing `ablation_results.json` against
the Stage 06 baselines, producing a normalized `ablation_results_renorm.json`
+ `ablation_summary_renorm.json` + `ABLATION_SUMMARY_renorm.md`. The ablated
classifications stay as-is; only the baseline (the denominator of recovery
and the trigger for break) is replaced.

Usage:
  python3 scripts/pipeline/08_renorm_baselines.py \\
      --run-dir data/results/pipeline_runs/run_20260430_023247_canonical_legacy \\
      --baseline-source data/results/pipeline_runs/run_20260430_023247

The `--baseline-source` is the run dir containing `06_causal/causal_results.json`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_json, save_json  # noqa: E402

JB_CLASSES = ("fiction", "roleplay", "analytical", "completion", "cognitive_reframe")


def load_stage06_baselines(source_run_dir: Path) -> dict:
    """Returns {prompt_id: {cond: {cls, coherent, response}}}."""
    path = source_run_dir / "06_causal" / "causal_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Stage 06 baselines not found at {path}")
    data = load_json(path)
    out: dict = {}
    for r in data.get("results", []):
        pid = r.get("prompt_id")
        if pid is None:
            continue
        out[pid] = r.get("baseline", {})
    return out


def renorm_results(results: list, stage06_baselines: dict) -> tuple[list, dict]:
    """Replace the baseline of each result row with Stage 06 baseline.

    Recomputes per-cell `changed_vs_baseline`. Returns (new_results, audit).
    """
    audit = {
        "n_rows": 0,
        "n_baseline_replaced": 0,
        "n_missing_in_stage06": 0,
        "n_changed_vs_baseline_flips": 0,
    }
    new_results = []
    for row in results:
        audit["n_rows"] += 1
        pid = row.get("prompt_id")
        new_row = dict(row)
        old_baseline = row.get("baseline", {})
        new_baseline = stage06_baselines.get(pid)
        if new_baseline is None:
            audit["n_missing_in_stage06"] += 1
            new_results.append(new_row)
            continue
        # Replace baseline (preserve any keys Stage 06 doesn't have)
        merged_baseline = {}
        for cond, info in old_baseline.items():
            if cond in new_baseline:
                merged_baseline[cond] = dict(new_baseline[cond])
                audit["n_baseline_replaced"] += 1
            else:
                merged_baseline[cond] = dict(info)
        new_row["baseline"] = merged_baseline

        # Recompute changed_vs_baseline for each ablation cell
        for abl_name, per_pos in new_row.get("ablations", {}).items():
            if not isinstance(per_pos, dict):
                continue
            for pos_mode, per_cond in per_pos.items():
                if pos_mode == "n_features":
                    continue
                if not isinstance(per_cond, dict):
                    continue
                for cond, cell in per_cond.items():
                    if not isinstance(cell, dict) or "cls" not in cell:
                        continue
                    bl_cls = merged_baseline.get(cond, {}).get("cls")
                    if bl_cls is None:
                        continue
                    new_changed = (cell["cls"] != bl_cls)
                    if cell.get("changed_vs_baseline") != new_changed:
                        audit["n_changed_vs_baseline_flips"] += 1
                    cell["changed_vs_baseline"] = new_changed
        new_results.append(new_row)
    return new_results, audit


def build_summary(results: list, conditions: list, ablation_names: list,
                  positions_modes: list) -> dict:
    """Recompute per-condition rate aggregates from renormalized results.

    Mirrors aggregate_summary() / aggregate_weighted_summary() in 08_ablate_subcircuits.py
    but operates on already-classified responses (no GPU).
    """
    summary: dict = {"per_ablation": {}}
    for abl_name in ablation_names:
        per_abl = {"n_features": None, "positions": {}}
        for pos_mode in positions_modes:
            per_pos: dict = {}
            for cond in conditions:
                baseline_refuse = 0
                baseline_comply = 0
                ablated_refuse = 0
                ablated_comply = 0
                changed = 0
                coherent_changed = 0
                recovered_refusal = 0
                broke_refusal = 0
                n_seen = 0
                for r in results:
                    bl = r.get("baseline", {}).get(cond)
                    abl_block = r.get("ablations", {}).get(abl_name, {})
                    if not isinstance(abl_block, dict):
                        continue
                    pos_block = abl_block.get(pos_mode, {})
                    if not isinstance(pos_block, dict):
                        continue
                    ab = pos_block.get(cond)
                    if bl is None or ab is None:
                        continue
                    n_seen += 1
                    if bl["cls"] == "REFUSE":
                        baseline_refuse += 1
                    else:
                        baseline_comply += 1
                    if ab["cls"] == "REFUSE":
                        ablated_refuse += 1
                    else:
                        ablated_comply += 1
                    if ab.get("changed_vs_baseline"):
                        changed += 1
                        if ab.get("coherent"):
                            coherent_changed += 1
                        if bl["cls"] == "COMPLY" and ab["cls"] == "REFUSE":
                            recovered_refusal += 1
                        if bl["cls"] == "REFUSE" and ab["cls"] == "COMPLY":
                            broke_refusal += 1
                    # Capture n_features once
                    n_feat = abl_block.get("n_features")
                    if n_feat is not None and per_abl["n_features"] is None:
                        per_abl["n_features"] = n_feat
                if n_seen == 0:
                    continue
                per_pos[cond] = {
                    "n_seen": n_seen,
                    "n_baseline_refuse": baseline_refuse,
                    "n_baseline_comply": baseline_comply,
                    "n_ablated_refuse": ablated_refuse,
                    "n_ablated_comply": ablated_comply,
                    "n_changed": changed,
                    "n_coherent_changed": coherent_changed,
                    "n_recovered_refusal": recovered_refusal,
                    "n_broke_refusal": broke_refusal,
                    "recovery_rate": (
                        round(recovered_refusal / baseline_comply, 4)
                        if baseline_comply else 0.0
                    ),
                    "break_rate": (
                        round(broke_refusal / baseline_refuse, 4)
                        if baseline_refuse else 0.0
                    ),
                }
            per_abl["positions"][pos_mode] = per_pos
        summary["per_ablation"][abl_name] = per_abl

    # Add weighted aggregates
    for abl_name, per_abl in summary["per_ablation"].items():
        for pos_mode, per_cond in per_abl.get("positions", {}).items():
            jb_total_comply = 0
            jb_weighted_recovery_num = 0.0
            per_cls = {}
            for cls in JB_CLASSES:
                cond = f"jb_{cls}"
                rec = per_cond.get(cond)
                if not rec:
                    continue
                cb = rec.get("n_baseline_comply", 0)
                rate = rec.get("recovery_rate", 0.0)
                jb_total_comply += cb
                jb_weighted_recovery_num += rate * cb
                per_cls[cls] = {
                    "recovery_rate": rate,
                    "n_baseline_comply": cb,
                    "n_recovered_refusal": rec.get("n_recovered_refusal", 0),
                }
            ctrl_total_refuse = 0
            ctrl_weighted_break_num = 0.0
            for cls in JB_CLASSES:
                cond = f"ctrl_{cls}"
                rec = per_cond.get(cond)
                if not rec:
                    continue
                rb = rec.get("n_baseline_refuse", 0)
                rate = rec.get("break_rate", 0.0)
                ctrl_total_refuse += rb
                ctrl_weighted_break_num += rate * rb
            bare_rec = per_cond.get("bare", {})
            per_cond["weighted"] = {
                "jb_weighted_recovery_rate": (
                    round(jb_weighted_recovery_num / jb_total_comply, 4)
                    if jb_total_comply else 0.0
                ),
                "jb_total_baseline_comply": jb_total_comply,
                "ctrl_weighted_break_rate": (
                    round(ctrl_weighted_break_num / ctrl_total_refuse, 4)
                    if ctrl_total_refuse else 0.0
                ),
                "ctrl_total_baseline_refuse": ctrl_total_refuse,
                "bare_break_rate": bare_rec.get("break_rate", 0.0),
                "bare_baseline_refuse": bare_rec.get("n_baseline_refuse", 0),
                "per_class_jb": per_cls,
            }
    return summary


def write_markdown(summary: dict, out_md: Path, source: str) -> None:
    """Compact markdown summary mirroring ABLATION_SUMMARY.md layout."""
    lines = ["# Stage 08 Subcircuit Ablation — Renormalized Summary"]
    lines.append("")
    lines.append(f"**Source**: {source}")
    lines.append("**Baseline source**: Stage 06 `causal_results.json` (max_new_tokens=200, H100).")
    lines.append("**Method**: ablated cells unchanged; baseline classifications replaced; aggregates recomputed.")
    lines.append("")
    for abl_name, per_abl in summary.get("per_ablation", {}).items():
        n_feat = per_abl.get("n_features", "?")
        lines.append(f"## `{abl_name}` (n={n_feat})")
        for pos_mode, per_cond in per_abl.get("positions", {}).items():
            lines.append(f"\n### Positions: {pos_mode}\n")
            lines.append("| Condition | n_baseline_REFUSE | n_baseline_COMPLY | n_ablated_REFUSE | recovery_rate | break_rate |")
            lines.append("|---|---|---|---|---|---|")
            for cond, rec in per_cond.items():
                if cond == "weighted":
                    continue
                lines.append(
                    f"| `{cond}` | {rec['n_baseline_refuse']} | {rec['n_baseline_comply']} | "
                    f"{rec['n_ablated_refuse']} | {rec['recovery_rate']*100:.1f}% | "
                    f"{rec['break_rate']*100:.1f}% |"
                )
            w = per_cond.get("weighted", {})
            lines.append("")
            lines.append(
                f"**Weighted**: JB_recovery={w.get('jb_weighted_recovery_rate', 0)*100:.1f}% "
                f"(n_jb_comply={w.get('jb_total_baseline_comply', 0)}), "
                f"ctrl_break={w.get('ctrl_weighted_break_rate', 0)*100:.1f}% "
                f"(n_ctrl_refuse={w.get('ctrl_total_baseline_refuse', 0)}), "
                f"bare_break={w.get('bare_break_rate', 0)*100:.1f}% "
                f"(n_bare_refuse={w.get('bare_baseline_refuse', 0)})"
            )
            lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(
        description="Renormalize Stage 08 ablation_results.json against Stage 06 baselines (Option B).",
    )
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Run directory containing 08_ablation/ablation_results.json.")
    p.add_argument("--baseline-source", type=Path, required=True,
                   help="Run directory containing 06_causal/causal_results.json (the source of truth).")
    p.add_argument("--output-suffix", type=str, default="renorm",
                   help="Suffix for output files (default: 'renorm').")
    args = p.parse_args()

    abl_dir = args.run_dir / "08_ablation"
    in_results_path = abl_dir / "ablation_results.json"
    if not in_results_path.exists():
        raise FileNotFoundError(f"Missing input: {in_results_path}")

    print(f"[renorm] reading {in_results_path}")
    in_data = load_json(in_results_path)
    results = in_data.get("results", in_data) if isinstance(in_data, dict) else in_data
    if not isinstance(results, list):
        raise ValueError(f"Unexpected schema; expected a list of result rows, got {type(results).__name__}")

    print(f"[renorm] loading Stage 06 baselines from {args.baseline_source}")
    stage06 = load_stage06_baselines(args.baseline_source)
    print(f"[renorm] Stage 06 has {len(stage06)} prompts with baselines")

    new_results, audit = renorm_results(results, stage06)
    print(f"[renorm] audit: {audit}")

    # Derive condition list + ablation list + positions modes from results
    conditions = []
    seen_cond = set()
    abl_names = []
    seen_abl = set()
    pos_modes = []
    seen_pos = set()
    for r in new_results:
        for c in r.get("baseline", {}):
            if c not in seen_cond:
                seen_cond.add(c)
                conditions.append(c)
        for ab_name, per_pos in r.get("ablations", {}).items():
            if ab_name not in seen_abl:
                seen_abl.add(ab_name)
                abl_names.append(ab_name)
            if isinstance(per_pos, dict):
                for k in per_pos:
                    if k != "n_features" and k not in seen_pos:
                        seen_pos.add(k)
                        pos_modes.append(k)

    print(f"[renorm] conditions={len(conditions)}, ablations={len(abl_names)}, positions={len(pos_modes)}")
    summary = build_summary(new_results, conditions, abl_names, pos_modes)

    # Write outputs
    suffix = args.output_suffix
    out_results = abl_dir / f"ablation_results_{suffix}.json"
    out_summary = abl_dir / f"ablation_summary_{suffix}.json"
    out_md = abl_dir / f"ABLATION_SUMMARY_{suffix}.md"
    save_json({"results": new_results, "renorm_audit": audit,
               "renorm_baseline_source": str(args.baseline_source)}, out_results)
    save_json(summary, out_summary)
    write_markdown(summary, out_md, source=str(args.run_dir))
    print(f"[renorm] wrote {out_results}")
    print(f"[renorm] wrote {out_summary}")
    print(f"[renorm] wrote {out_md}")

    # Quick headline print
    for abl_name, per_abl in summary["per_ablation"].items():
        for pos_mode, per_cond in per_abl["positions"].items():
            w = per_cond.get("weighted", {})
            print(
                f"[renorm]  {abl_name} ({pos_mode}, n={per_abl.get('n_features')}): "
                f"JB={w.get('jb_weighted_recovery_rate', 0)*100:5.1f}% "
                f"(n={w.get('jb_total_baseline_comply', 0)})  "
                f"ctrl={w.get('ctrl_weighted_break_rate', 0)*100:5.1f}%  "
                f"bare={w.get('bare_break_rate', 0)*100:5.1f}%"
            )


if __name__ == "__main__":
    main()
