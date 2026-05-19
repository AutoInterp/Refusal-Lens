"""Aggregate Phase 0 GPU outputs (0b + 0d + 0e) into flip-rate tables + figures.

Reads three JSONs produced on the GPU run:
  - edge_ablation_flip_rates.json   (0b-simple, 7 variants)
  - topk_feature_sweep.json         (0d, 3 variants x 7 K)
  - topk_edge_sweep.json            (0e, 3 variants x 7 K)

For each, compares classifications against Stage 06 baselines from
`06_causal/causal_results.json` to compute per-(variant[, K], condition)
flip rates with Wilson 95% CIs.

Produces:
  - flip_rate_summary.json
  - PHASE0_GPU_SUMMARY.md
  - controllability_audit_figure.png      (0b: 7 variants x bare flip rate)
  - topk_feature_pareto_figure.png        (0d: 3 curves on bare-refuse)
  - topk_edge_vs_node_figure.png          (0e overlaid on 0d)

Tests H0-1 / H0-2 / H0-3 / H0-4 / H0-6 / H0-7 behaviorally.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
JB_CLASSES = ("fiction", "roleplay", "analytical", "completion", "cognitive_reframe")
CTRL_CLASSES = tuple(f"ctrl_{c}" for c in JB_CLASSES)
ALL_CONDITIONS = ("bare",) + tuple(f"jb_{c}" for c in JB_CLASSES) + CTRL_CLASSES


def wilson_ci(n_success: int, n_total: int, alpha: float = 0.05) -> tuple[float, float]:
    if n_total == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = n_success / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    half = z * math.sqrt((p * (1 - p) + z**2 / (4 * n_total)) / n_total) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def load_baselines(run_dir: Path) -> dict:
    """Stage 06 baselines: {prompt_idx: {condition: classification}}."""
    causal = json.loads((run_dir / "06_causal/causal_results.json").read_text())
    baselines = {}
    for r in causal["results"]:
        baselines[r["prompt_idx"]] = {c: blob["cls"] for c, blob in r["baseline"].items()}
    return baselines


def compute_per_condition_flip_rates(records: list[dict], baselines: dict) -> dict:
    """Per-condition flip rate vs Stage 06 baseline."""
    per_cond = {}
    for cond in ALL_CONDITIONS:
        target_baseline = "COMPLY" if cond.startswith("jb_") else "REFUSE"
        target_intervened = "REFUSE" if target_baseline == "COMPLY" else "COMPLY"
        n_baseline = 0
        n_flipped = 0
        for r in records:
            if r["condition"] != cond:
                continue
            b = baselines.get(r["prompt_idx"], {}).get(cond, "UNCLEAR")
            if b == target_baseline:
                n_baseline += 1
                if r["classification"] == target_intervened:
                    n_flipped += 1
        rate = n_flipped / n_baseline if n_baseline > 0 else 0.0
        ci_lo, ci_hi = wilson_ci(n_flipped, n_baseline)
        per_cond[cond] = {
            "flip_rate": rate, "n_flipped": n_flipped, "n_baseline": n_baseline,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "target_baseline_cls": target_baseline, "target_intervened_cls": target_intervened,
        }
    return per_cond


def aggregate_0b(data: dict, baselines: dict) -> dict:
    out = {}
    for variant, records in data["per_variant"].items():
        out[variant] = compute_per_condition_flip_rates(records, baselines)
    return out


def aggregate_topk(data: dict, baselines: dict) -> dict:
    """Returns {variant_K: {cond: {flip_rate, ...}}}."""
    out = {}
    for key, records in data["per_variant_K"].items():
        out[key] = compute_per_condition_flip_rates(records, baselines)
    return out


def render_0b_figure(summary_0b: dict, out_path: Path):
    """Bar chart: per-variant bare-refuse → COMPLY flip rate with Wilson CIs."""
    variants = list(summary_0b.keys())
    rates = [summary_0b[v]["bare"]["flip_rate"] for v in variants]
    los = [summary_0b[v]["bare"]["ci_lo"] for v in variants]
    his = [summary_0b[v]["bare"]["ci_hi"] for v in variants]
    err_low = [(rates[i] - los[i]) * 100 for i in range(len(variants))]
    err_high = [(his[i] - rates[i]) * 100 for i in range(len(variants))]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(variants, [r * 100 for r in rates], yerr=[err_low, err_high], capsize=4)
    ax.set_ylabel("bare-refuse → COMPLY flip rate (%)")
    ax.set_title("Phase 0 0b — Controllability audit: bare-refuse flip rate per variant\n"
                 "(higher = more direction-axis control via the ablated edge bucket)")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=30, ha="right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def render_topk_pareto(summary_topk: dict, target_condition: str, title_prefix: str,
                       out_path: Path, overlay_other: dict | None = None,
                       overlay_label: str | None = None):
    """Pareto curve: per K, plot the bare-refuse flip rate for each (pos/neg/abs) variant."""
    # Group by variant; sort by K
    by_variant = {}
    for key, by_cond in summary_topk.items():
        variant, K_str = key.split("_K")
        K = int(K_str)
        by_variant.setdefault(variant, []).append((K, by_cond[target_condition]))
    for v in by_variant:
        by_variant[v].sort(key=lambda x: x[0])

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"pos": "#4C72B0", "neg": "#C44E52", "abs": "#55A868"}
    for variant, points in by_variant.items():
        Ks = [p[0] for p in points]
        rates = [p[1]["flip_rate"] * 100 for p in points]
        los = [p[1]["ci_lo"] * 100 for p in points]
        his = [p[1]["ci_hi"] * 100 for p in points]
        err_low = [rates[i] - los[i] for i in range(len(rates))]
        err_high = [his[i] - rates[i] for i in range(len(rates))]
        ax.errorbar(Ks, rates, yerr=[err_low, err_high], marker="o", linewidth=2,
                    capsize=4, label=f"{variant}-K", color=colors.get(variant))

    if overlay_other is not None:
        by_variant_o = {}
        for key, by_cond in overlay_other.items():
            variant, K_str = key.split("_K")
            K = int(K_str)
            by_variant_o.setdefault(variant, []).append((K, by_cond[target_condition]))
        for v in by_variant_o:
            by_variant_o[v].sort(key=lambda x: x[0])
        for variant, points in by_variant_o.items():
            Ks = [p[0] for p in points]
            rates = [p[1]["flip_rate"] * 100 for p in points]
            ax.plot(Ks, rates, marker="x", linestyle="--", linewidth=1.5,
                    label=f"{variant}-K ({overlay_label})", color=colors.get(variant), alpha=0.6)

    ax.set_xscale("log")
    ax.set_xlabel("K (top features/edges per prompt)")
    ax.set_ylabel(f"{target_condition} flip rate (%)")
    ax.set_title(f"{title_prefix} — Pareto curve at {target_condition}\n"
                 "(signed variants: pos = top-K positive attribution; neg = top-K negative; abs = top-K |attr|)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def render_summary_md(summary_0b: dict, summary_0d: dict | None, summary_0e: dict | None,
                       out_path: Path):
    md = ["# Phase 0 GPU Outputs Summary\n"]
    md.append("## 0b-simple — controllability audit\n")
    md.append("Bare-refuse → COMPLY flip rate per variant (predicts H0-1, H0-2, H0-3):\n")
    md.append("| Variant | bare flip rate | 95% CI | n |")
    md.append("|---|---:|---:|---:|")
    for v in summary_0b:
        b = summary_0b[v]["bare"]
        md.append(f"| {v} | {b['flip_rate']*100:.1f}% | "
                 f"[{b['ci_lo']*100:.1f}, {b['ci_hi']*100:.1f}] | "
                 f"{b['n_flipped']}/{b['n_baseline']} |")

    md.append("\n### Per JB-class flip rate (H0-1 controllability)\n")
    md.append("| Variant | " + " | ".join(f"jb_{c[:6]}" for c in JB_CLASSES) + " |")
    md.append("|---|" + "---:|" * len(JB_CLASSES))
    for v in summary_0b:
        row = [v]
        for c in JB_CLASSES:
            cond = f"jb_{c}"
            blob = summary_0b[v][cond]
            row.append(f"{blob['flip_rate']*100:.1f}% ({blob['n_flipped']}/{blob['n_baseline']})")
        md.append("| " + " | ".join(row) + " |")

    if summary_0d:
        md.append("\n## 0d — top-K feature Pareto sweep (H0-6)\n")
        md.append("Bare-refuse flip rate per (variant, K):\n")
        md.append("| Variant\\K | " + " | ".join(str(K) for K in sorted({int(k.split("_K")[1]) for k in summary_0d})) + " |")
        all_Ks = sorted({int(k.split("_K")[1]) for k in summary_0d})
        all_vs = sorted({k.split("_K")[0] for k in summary_0d})
        md.append("|---|" + "---:|" * len(all_Ks))
        for v in all_vs:
            row = [v]
            for K in all_Ks:
                key = f"{v}_K{K}"
                if key in summary_0d:
                    rate = summary_0d[key]["bare"]["flip_rate"] * 100
                    row.append(f"{rate:.1f}%")
                else:
                    row.append("-")
            md.append("| " + " | ".join(row) + " |")

    if summary_0e:
        md.append("\n## 0e — top-K edge Pareto sweep (H0-7)\n")
        md.append("Bare-refuse flip rate per (variant, K):\n")
        all_Ks = sorted({int(k.split("_K")[1]) for k in summary_0e})
        all_vs = sorted({k.split("_K")[0] for k in summary_0e})
        md.append("| Variant\\K | " + " | ".join(str(K) for K in all_Ks) + " |")
        md.append("|---|" + "---:|" * len(all_Ks))
        for v in all_vs:
            row = [v]
            for K in all_Ks:
                key = f"{v}_K{K}"
                if key in summary_0e:
                    rate = summary_0e[key]["bare"]["flip_rate"] * 100
                    row.append(f"{rate:.1f}%")
                else:
                    row.append("-")
            md.append("| " + " | ".join(row) + " |")

    out_path.write_text("\n".join(md))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=Path,
                   default=REPO / "data/results/emnlp_perm_edit/phase0_controllability")
    p.add_argument("--baseline-run-dir", type=Path,
                   default=REPO / "data/results/pipeline_runs/run_20260430_023247")
    args = p.parse_args()

    baselines = load_baselines(args.baseline_run_dir)
    out = {"per_experiment": {}}

    # 0b-simple
    b_path = args.in_dir / "edge_ablation_flip_rates.json"
    summary_0b = None
    if b_path.exists():
        print(f"[aggregate] loading 0b from {b_path}")
        data_0b = json.loads(b_path.read_text())
        summary_0b = aggregate_0b(data_0b, baselines)
        out["per_experiment"]["0b"] = summary_0b
        render_0b_figure(summary_0b, args.in_dir / "controllability_audit_figure.png")
        print(f"  wrote controllability_audit_figure.png")
    else:
        print(f"[aggregate] 0b output not found at {b_path} (skipping)")

    # 0d top-K feature
    d_path = args.in_dir / "topk_feature_sweep.json"
    summary_0d = None
    if d_path.exists():
        print(f"[aggregate] loading 0d from {d_path}")
        data_0d = json.loads(d_path.read_text())
        summary_0d = aggregate_topk(data_0d, baselines)
        out["per_experiment"]["0d"] = summary_0d
        render_topk_pareto(summary_0d, "bare", "Phase 0 0d — top-K feature ablation",
                           args.in_dir / "topk_feature_pareto_figure.png")
        print(f"  wrote topk_feature_pareto_figure.png")
    else:
        print(f"[aggregate] 0d output not found at {d_path} (skipping)")

    # 0e top-K edge
    e_path = args.in_dir / "topk_edge_sweep.json"
    summary_0e = None
    if e_path.exists():
        print(f"[aggregate] loading 0e from {e_path}")
        data_0e = json.loads(e_path.read_text())
        summary_0e = aggregate_topk(data_0e, baselines)
        out["per_experiment"]["0e"] = summary_0e
        render_topk_pareto(
            summary_0e, "bare", "Phase 0 0e — top-K edge ablation",
            args.in_dir / "topk_edge_vs_node_figure.png",
            overlay_other=summary_0d, overlay_label="0d node",
        )
        print(f"  wrote topk_edge_vs_node_figure.png")
    else:
        print(f"[aggregate] 0e output not found at {e_path} (skipping)")

    (args.in_dir / "flip_rate_summary.json").write_text(json.dumps(out, indent=2))
    render_summary_md(summary_0b or {}, summary_0d, summary_0e,
                       args.in_dir / "PHASE0_GPU_SUMMARY.md")
    print(f"\n[aggregate] wrote flip_rate_summary.json + PHASE0_GPU_SUMMARY.md")


if __name__ == "__main__":
    main()
