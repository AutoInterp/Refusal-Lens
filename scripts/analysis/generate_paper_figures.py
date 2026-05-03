"""
Generate paper figures for Outline E from existing Stage 08 sweep results.
==========================================================================
Pulls renormalized (Stage 06 baseline-aligned) ablation summaries from:
  - run_20260430_023247                            (orig 5 ablations, renormed)
  - run_20260430_023247_canonical_{legacy, k100f20, k50f50}  (canonical sweep, renormed)
  - run_20260430_023247_full_{k100f20, k50f50}     (Tier 2: 8 subcircuits × 2 configs)
  - run_20260430_023247_topN                       (Tier 1: per-prompt top-N + random)
  - run_20260430_023247/06_causal/causal_summary.json  (direction intervention 100/98/100)

Outputs to ./figures/:
  F3_per_prompt_vs_corpus_union.png  — Pillar 2: 6 vs 88 vs 1 across constructions
  F4_L13_F427_spotlight.png          — Pillar 4: single-feature spotlight
  F5_recovery_vs_features_pareto.png — Pillar 3: the Pareto curve (the headline)
  F6_construction_rule_robustness.png — Pillar 2 robustness across all subcircuits

  recovery_table.csv                  — every ablation × config × headline numbers
  recovery_with_ci.json               — Wilson 95% CIs on every recovery rate

Designed to be re-runnable as more sweeps land. Skips missing inputs gracefully.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "data" / "results" / "pipeline_runs"
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load_summary(run_dir: Path, prefer_renorm: bool = True) -> dict | None:
    """Load ablation_summary[_renorm].json, falling back to non-renormed if missing."""
    if prefer_renorm:
        path = run_dir / "08_ablation" / "ablation_summary_renorm.json"
        if path.exists():
            return json.load(open(path))
    path = run_dir / "08_ablation" / "ablation_summary.json"
    if path.exists():
        return json.load(open(path))
    return None


def load_causal_summary(run_dir: Path) -> dict | None:
    path = run_dir / "06_causal" / "causal_summary.json"
    if path.exists():
        return json.load(open(path))
    return None


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for binomial proportion. Returns (lo, hi) in [0,1]."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) / n) + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def collect_all_ablations() -> list[dict]:
    """Walk all sister run dirs and collect every (ablation × config × position) row."""
    sources = [
        ("orig", RESULTS / "run_20260430_023247", "orig 5-ablations"),
        ("canonical_legacy", RESULTS / "run_20260430_023247_canonical_legacy", "canonical legacy 88"),
        ("canonical_k100f20", RESULTS / "run_20260430_023247_canonical_k100f20", "canonical k100_f20"),
        ("canonical_k50f50", RESULTS / "run_20260430_023247_canonical_k50f50", "canonical k50_f50"),
        ("full_k100f20", RESULTS / "run_20260430_023247_full_k100f20", "full sweep k100_f20"),
        ("full_k50f50", RESULTS / "run_20260430_023247_full_k50f50", "full sweep k50_f50"),
        ("topN", RESULTS / "run_20260430_023247_topN", "per-prompt top-N + random"),
    ]
    rows = []
    for tag, run_dir, descr in sources:
        s = load_summary(run_dir)
        if s is None:
            print(f"[skip] {tag}: no ablation_summary found at {run_dir}/08_ablation/")
            continue
        print(f"[load] {tag}: {len(s.get('per_ablation', {}))} ablations from {run_dir.name}")
        for abl_name, per_abl in s.get("per_ablation", {}).items():
            n_feat = per_abl.get("n_features")
            for pos_mode, per_cond in per_abl.get("positions", {}).items():
                w = per_cond.get("weighted", {})
                if not w:
                    continue
                jb_n = w.get("jb_total_baseline_comply", 0)
                jb_rate = w.get("jb_weighted_recovery_rate", 0.0)
                jb_k = round(jb_rate * jb_n)
                jb_lo, jb_hi = wilson_ci(jb_k, jb_n)
                ctrl_n = w.get("ctrl_total_baseline_refuse", 0)
                ctrl_rate = w.get("ctrl_weighted_break_rate", 0.0)
                ctrl_k = round(ctrl_rate * ctrl_n)
                ctrl_lo, ctrl_hi = wilson_ci(ctrl_k, ctrl_n)
                bare_n = w.get("bare_baseline_refuse", 0)
                bare_rate = w.get("bare_break_rate", 0.0)
                bare_k = round(bare_rate * bare_n)
                bare_lo, bare_hi = wilson_ci(bare_k, bare_n)
                rows.append({
                    "source": tag,
                    "source_descr": descr,
                    "ablation": abl_name,
                    "n_features": n_feat,
                    "position_mode": pos_mode,
                    "jb_recovery_rate": jb_rate,
                    "jb_n": jb_n,
                    "jb_ci_lo": jb_lo,
                    "jb_ci_hi": jb_hi,
                    "ctrl_break_rate": ctrl_rate,
                    "ctrl_n": ctrl_n,
                    "ctrl_ci_lo": ctrl_lo,
                    "ctrl_ci_hi": ctrl_hi,
                    "bare_break_rate": bare_rate,
                    "bare_n": bare_n,
                    "bare_ci_lo": bare_lo,
                    "bare_ci_hi": bare_hi,
                    "per_class_jb": w.get("per_class_jb", {}),
                })
    return rows


def write_table(rows: list[dict]) -> None:
    """Write a CSV table of every row."""
    out = FIG_DIR / "recovery_table.csv"
    cols = ["source", "ablation", "n_features", "position_mode",
            "jb_recovery_rate", "jb_n", "jb_ci_lo", "jb_ci_hi",
            "ctrl_break_rate", "ctrl_n", "ctrl_ci_lo", "ctrl_ci_hi",
            "bare_break_rate", "bare_n", "bare_ci_lo", "bare_ci_hi"]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(f"{r.get(c, '')}" for c in cols))
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[wrote] {out}")


def write_ci_json(rows: list[dict]) -> None:
    out = FIG_DIR / "recovery_with_ci.json"
    json.dump(rows, open(out, "w"), indent=2)
    print(f"[wrote] {out}")


def fig5_recovery_vs_features(rows: list[dict], direction_rate: float = 1.0) -> None:
    """The Pareto curve. x = log(n_features), y = JB recovery, with direction line at top."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[skip] matplotlib unavailable")
        return

    # Bucket by source category
    cat_colors = {
        "corpus_union": "#888888",
        "per_prompt_k100_f20": "#2C7FB8",
        "per_prompt_k50_f50": "#0570B0",
        "per_prompt_top_N": "#41AB5D",
        "random_control": "#E31A1C",
        "orig": "#FB6A4A",
        "direction": "#A60026",
    }

    def categorize(r):
        s, name = r["source"], r["ablation"]
        if name.startswith("per_prompt_top_"):
            return "per_prompt_top_N"
        if name.startswith("per_prompt_random_"):
            return "random_control"
        if "k100f20" in s or "k100_f20" in s:
            return "per_prompt_k100_f20"
        if "k50f50" in s or "k50_f50" in s:
            return "per_prompt_k50_f50"
        if "legacy" in s:
            return "corpus_union"
        if s == "orig":
            return "orig"
        return "corpus_union"

    fig, ax = plt.subplots(figsize=(9, 6))
    # Direction line
    ax.axhline(y=direction_rate, color=cat_colors["direction"], linestyle="--", linewidth=2,
               label=f"Direction intervention ({direction_rate*100:.0f}%)", zorder=10)

    # Points
    for r in rows:
        if r["position_mode"] != "all":
            continue
        if not r["n_features"] or r["n_features"] <= 0:
            continue
        cat = categorize(r)
        color = cat_colors.get(cat, "#666666")
        x = r["n_features"]
        y = r["jb_recovery_rate"]
        # Vertical CI bars
        lo = r["jb_ci_lo"]
        hi = r["jb_ci_hi"]
        ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]], fmt="o", color=color,
                    capsize=3, alpha=0.85, markersize=8)
        # Annotate Pareto pts
        ax.annotate(r["ablation"][:18], (x, y), fontsize=7, alpha=0.6,
                    xytext=(5, 3), textcoords="offset points")

    # Legend
    seen_cats = set()
    for r in rows:
        cat = categorize(r)
        if cat in seen_cats:
            continue
        seen_cats.add(cat)
        ax.scatter([], [], color=cat_colors[cat], label=cat.replace("_", " "), s=80)

    ax.set_xscale("log")
    ax.set_xlabel("n features ablated (log scale)", fontsize=12)
    ax.set_ylabel("JB-comply → REFUSE recovery rate (weighted)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Recovery-vs-features Pareto: ablations plateau far below direction-level potency",
                 fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", fontsize=10)

    out = FIG_DIR / "F5_recovery_vs_features_pareto.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[wrote] {out}")


def fig3_per_prompt_vs_corpus(rows: list[dict]) -> None:
    """Bar chart: canonical_pro_refusal at 88 vs 6 vs 1 features (renormalized)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    # Pull canonical_pro_refusal rows
    target = [r for r in rows if r["ablation"] == "canonical_pro_refusal" and r["position_mode"] == "all"]
    target.sort(key=lambda r: r["n_features"] or 0)
    if not target:
        print("[skip] no canonical_pro_refusal rows for F3")
        return

    labels = [f"{r['n_features']}f\n({r['source'].replace('canonical_', '')})" for r in target]
    jb = [r["jb_recovery_rate"] * 100 for r in target]
    jb_lo = [(r["jb_recovery_rate"] - r["jb_ci_lo"]) * 100 for r in target]
    jb_hi = [(r["jb_ci_hi"] - r["jb_recovery_rate"]) * 100 for r in target]
    ctrl = [r["ctrl_break_rate"] * 100 for r in target]
    bare = [r["bare_break_rate"] * 100 for r in target]

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - w, jb, w, label="JB recovery (↑ is good)", color="#2C7FB8",
           yerr=[jb_lo, jb_hi], capsize=4)
    ax.bar(x, ctrl, w, label="ctrl break (↓ is good)", color="#E31A1C")
    ax.bar(x + w, bare, w, label="bare break (↓ is good)", color="#FB6A4A")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate (%)")
    ax.set_title("`canonical_pro_refusal` across construction rules\n"
                 "Per-prompt 6 features beats corpus-union 88 features on every axis",
                 fontsize=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    out = FIG_DIR / "F3_per_prompt_vs_corpus_union.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[wrote] {out}")


def fig6_construction_rule_robustness(rows: list[dict]) -> None:
    """Scatter: construction rule (x) × subcircuit (y), color = recovery."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    # Group by (subcircuit, source) for `all` mode
    # Sources we care about: orig, canonical_*, full_*
    keep_subs = ["universal_refusal_core", "ctrl_shared_refusal", "canonical_pro_refusal",
                 "jb_fiction_specific_vs_ctrl", "jb_analytical_specific_vs_ctrl",
                 "jb_cognitive_reframe_specific_vs_ctrl", "jb_completion_specific_vs_ctrl",
                 "jb_roleplay_specific_vs_ctrl", "anti_refusal_amplifiers"]
    config_order = ["orig", "canonical_legacy", "full_k100f20", "canonical_k100f20",
                    "full_k50f50", "canonical_k50f50"]
    config_label = {"orig": "orig (corpus, K=50)", "canonical_legacy": "canonical legacy (88)",
                    "full_k100f20": "k100_f20 sweep", "canonical_k100f20": "k100_f20 canonical",
                    "full_k50f50": "k50_f50 sweep", "canonical_k50f50": "k50_f50 canonical"}

    grid = {}  # {(sub, config): rate}
    n_grid = {}  # {(sub, config): n_features}
    for r in rows:
        if r["position_mode"] != "all":
            continue
        if r["ablation"] not in keep_subs:
            continue
        if r["source"] not in config_order:
            continue
        grid[(r["ablation"], r["source"])] = r["jb_recovery_rate"]
        n_grid[(r["ablation"], r["source"])] = r["n_features"]

    if not grid:
        print("[skip] no rows for F6")
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    sub_y = {s: i for i, s in enumerate(reversed(keep_subs))}
    cfg_x = {c: i for i, c in enumerate(config_order)}

    for (sub, cfg), rate in grid.items():
        x = cfg_x[cfg]
        y = sub_y[sub]
        n = n_grid[(sub, cfg)]
        if n is None or n <= 0:
            continue
        size = max(40, min(400, (n or 0) * 4))
        sc = ax.scatter(x, y, s=size, c=[rate], cmap="RdYlGn", vmin=0, vmax=0.5,
                        edgecolors="black", alpha=0.85)
        ax.text(x, y, f"{rate*100:.0f}%\nn={n}", ha="center", va="center", fontsize=8)

    ax.set_xticks(list(cfg_x.values()))
    ax.set_xticklabels([config_label[c] for c in config_order], rotation=30, ha="right")
    ax.set_yticks(list(sub_y.values()))
    ax.set_yticklabels([s.replace("_", " ") for s in reversed(keep_subs)])
    ax.set_title("JB recovery rate across (subcircuit × construction rule)\n"
                 "Marker size = n_features; color = recovery rate",
                 fontsize=12)
    plt.colorbar(sc, ax=ax, label="JB recovery rate")
    ax.grid(True, alpha=0.3)

    out = FIG_DIR / "F6_construction_rule_robustness.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[wrote] {out}")


def fig4_l13_f427_spotlight(rows: list[dict]) -> None:
    """L13:F427 single-feature spotlight: per-class flip rate, comparison to canonical."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    JB = ["fiction", "roleplay", "analytical", "cognitive_reframe", "completion"]

    # canonical_pro_refusal at k50_f50 IS L13:F427 alone
    single = next((r for r in rows
                   if r["ablation"] == "canonical_pro_refusal"
                   and "k50f50" in r["source"]
                   and r["position_mode"] == "all"), None)
    six = next((r for r in rows
                if r["ablation"] == "canonical_pro_refusal"
                and "k100f20" in r["source"]
                and r["position_mode"] == "all"), None)

    if single is None or six is None:
        print("[skip] missing canonical_pro_refusal rows for F4")
        return

    single_rates = [single["per_class_jb"].get(c, {}).get("recovery_rate", 0.0) for c in JB]
    six_rates = [six["per_class_jb"].get(c, {}).get("recovery_rate", 0.0) for c in JB]

    x = np.arange(len(JB))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, [r*100 for r in single_rates], w,
           label=f"L13:F427 alone (1 feat, weighted={single['jb_recovery_rate']*100:.1f}%)",
           color="#A60026")
    ax.bar(x + w/2, [r*100 for r in six_rates], w,
           label=f"6-feature canonical (k100_f20, weighted={six['jb_recovery_rate']*100:.1f}%)",
           color="#2C7FB8")
    ax.set_xticks(x)
    ax.set_xticklabels([f"jb_{c}" for c in JB], rotation=20)
    ax.set_ylabel("JB-comply → REFUSE recovery rate (%)")
    ax.set_title("L13:F427 single-feature spotlight\n"
                 "Single feature recovers a meaningful fraction; 6-feature canonical adds compositional expression",
                 fontsize=11)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    out = FIG_DIR / "F4_L13_F427_spotlight.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[wrote] {out}")


def main():
    print("[generate_paper_figures] collecting data...")
    rows = collect_all_ablations()
    print(f"[generate_paper_figures] {len(rows)} (ablation × config) rows total")

    write_table(rows)
    write_ci_json(rows)

    # Direction intervention rate (Stage 06 weighted) — load from causal_summary if available
    causal = load_causal_summary(RESULTS / "run_20260430_023247")
    direction_rate = 1.0
    if causal:
        # Stage 06 schema: summary has L15_pro_refusal_add → flip_rate, etc.
        summary = causal.get("summary", causal)
        for key, block in summary.items():
            if "pro_refusal_add" in key:
                rate = block.get("flip_rate")
                if rate is not None:
                    direction_rate = rate
                    print(f"[generate_paper_figures] using direction_rate from {key}: {direction_rate}")
                    break

    fig5_recovery_vs_features(rows, direction_rate)
    fig3_per_prompt_vs_corpus(rows)
    fig4_l13_f427_spotlight(rows)
    fig6_construction_rule_robustness(rows)

    print("\n[generate_paper_figures] DONE — figures in ./figures/")


if __name__ == "__main__":
    main()
