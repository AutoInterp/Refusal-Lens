"""
Generate paper figures from Stage 02b geometry + Stage 08 ablation sweep results.
================================================================================
F1 (geometry, ICML v2 narrative) reads:
  - run_20260430_023247/02b_stats/residuals_L15_per_cond.pt
  - run_20260430_023247/01_direction/refusal_direction.pt
  - run_20260430_023247/02b_stats/direction_alignment.json

F3-F6 (ablation sweep, supplementary) read renormalised (Stage 06 baseline-aligned)
ablation summaries from:
  - run_20260430_023247                            (orig 5 ablations, renormed)
  - run_20260430_023247_canonical_{legacy, k100f20, k50f50}  (canonical sweep, renormed)
  - run_20260430_023247_full_{k100f20, k50f50}     (Tier 2: 8 subcircuits × 2 configs)
  - run_20260430_023247_topN                       (Tier 1: per-prompt top-N + random)
  - run_20260430_023247/06_causal/causal_summary.json  (direction intervention 100/98/100)

Outputs to ./figures/ (PDF + PNG for each):
  F1_per_class_geometry              — ICML v2 lead figure: per-class JB displacement
  F3_per_prompt_vs_corpus_union.png  — supplementary Pillar 2: 6 vs 88 vs 1 across constructions
  F4_L13_F427_spotlight.png          — supplementary Pillar 4: single-feature spotlight
  F5_recovery_vs_features_pareto.png — supplementary Pillar 3: the Pareto curve
  F6_construction_rule_robustness.png — supplementary Pillar 2 robustness across subcircuits

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

# ICML v2 paper canonical run (geometry + causal data come from here).
GEOMETRY_RUN = "run_20260430_023247"

# Per-class palette used by F1 (and any future per-class figure).
JB_CLASSES = ["fiction", "roleplay", "analytical", "completion", "cognitive_reframe"]
JB_CLASS_COLORS = {
    "fiction":           "#D62728",  # red
    "roleplay":          "#FF7F0E",  # orange
    "analytical":        "#2CA02C",  # green
    "completion":        "#9467BD",  # purple
    "cognitive_reframe": "#1F77B4",  # blue
}
JB_CLASS_LABEL = {
    "fiction": "fiction",
    "roleplay": "roleplay",
    "analytical": "analytical",
    "completion": "completion",
    "cognitive_reframe": "cog. reframe",
}


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

    fig, ax = plt.subplots(figsize=(13, 7.5))
    # Direction line
    ax.axhline(y=direction_rate, color=cat_colors["direction"], linestyle="--", linewidth=2,
               label=f"Direction intervention ({direction_rate*100:.0f}%)", zorder=10)

    # Collect plottable points first
    pts = []  # list of (x, y, lo, hi, cat, color, name)
    for r in rows:
        if r["position_mode"] != "all":
            continue
        if not r["n_features"] or r["n_features"] <= 0:
            continue
        cat = categorize(r)
        color = cat_colors.get(cat, "#666666")
        pts.append((r["n_features"], r["jb_recovery_rate"], r["jb_ci_lo"], r["jb_ci_hi"],
                    cat, color, r["ablation"]))

    # Draw error bars
    for x, y, lo, hi, cat, color, name in pts:
        ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]], fmt="o", color=color,
                    capsize=3, alpha=0.85, markersize=8)

    # Pareto frontier: maximize y for any n <= x. Only annotate frontier + a few extremes.
    sorted_pts = sorted(pts, key=lambda p: (p[0], -p[1]))
    frontier = []
    best_y = -1.0
    for p in sorted_pts:
        if p[1] > best_y:
            frontier.append(p)
            best_y = p[1]
    # Also force-label the highest-y point overall and the largest-n point
    must_label = set(id(p) for p in frontier)
    if pts:
        must_label.add(id(max(pts, key=lambda p: p[1])))
        must_label.add(id(max(pts, key=lambda p: p[0])))
        must_label.add(id(min(pts, key=lambda p: p[0])))

    # Place labels with leader lines, alternating above/below to reduce overlap
    label_pts = [p for p in pts if id(p) in must_label]
    label_pts.sort(key=lambda p: p[0])
    for i, (x, y, lo, hi, cat, color, name) in enumerate(label_pts):
        # Alternate vertical placement so neighbors don't stack
        dy = 18 if (i % 2 == 0) else -22
        dx = 6 if (i % 2 == 0) else -6
        ha = "left" if dx > 0 else "right"
        ax.annotate(name, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, alpha=0.85, ha=ha,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.6, alpha=0.85),
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.6, alpha=0.6))

    # Legend
    seen_cats = set()
    for _, _, _, _, cat, _, _ in pts:
        if cat in seen_cats:
            continue
        seen_cats.add(cat)
        ax.scatter([], [], color=cat_colors[cat], label=cat.replace("_", " "), s=80)

    ax.set_xscale("log")
    ax.set_xlabel("n features ablated (log scale)", fontsize=12)
    ax.set_ylabel("JB-comply → REFUSE recovery rate (weighted)", fontsize=12)
    ax.set_ylim(-0.05, 1.10)
    ax.set_title("Recovery-vs-features Pareto: ablations plateau far below direction-level potency",
                 fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", fontsize=10, framealpha=0.9)

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

    fig, ax = plt.subplots(figsize=(12, 7.5))
    sub_y = {s: i for i, s in enumerate(reversed(keep_subs))}
    cfg_x = {c: i for i, c in enumerate(config_order)}

    # Cap marker size so big-n cells don't swallow neighbours; use sqrt scaling.
    import math as _math
    sc = None
    for (sub, cfg), rate in grid.items():
        x = cfg_x[cfg]
        y = sub_y[sub]
        n = n_grid[(sub, cfg)]
        if n is None or n <= 0:
            continue
        size = max(60, min(260, 30 * _math.sqrt(n)))
        sc = ax.scatter(x, y, s=size, c=[rate], cmap="RdYlGn", vmin=0, vmax=0.5,
                        edgecolors="black", linewidths=0.6, alpha=0.9, zorder=3)
        # Two-line label placed above the marker so it never clashes with the disk.
        ax.annotate(
            f"{rate*100:.0f}%  (n={n})",
            xy=(x, y), xytext=(0, 14), textcoords="offset points",
            ha="center", va="bottom", fontsize=8.5, zorder=4,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#888888",
                      lw=0.4, alpha=0.85),
        )

    # Add some breathing room around the grid
    ax.set_xlim(-0.6, len(config_order) - 0.4)
    ax.set_ylim(-0.6, len(keep_subs) - 0.4)
    ax.set_xticks(list(cfg_x.values()))
    ax.set_xticklabels([config_label[c] for c in config_order], rotation=25, ha="right",
                       fontsize=10)
    ax.set_yticks(list(sub_y.values()))
    ax.set_yticklabels([s.replace("_", " ") for s in reversed(keep_subs)], fontsize=10)
    ax.set_title("JB recovery rate across (subcircuit × construction rule)\n"
                 "Marker size ∝ √n_features; colour = recovery rate; label shows %  (n=…)",
                 fontsize=12)
    if sc is not None:
        plt.colorbar(sc, ax=ax, label="JB recovery rate", pad=0.02)
    ax.grid(True, alpha=0.25, zorder=1)

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


# ==========================================================================
# F1 — Per-class jailbreak displacement geometry (ICML v2 lead figure)
# ==========================================================================

def fig1_per_class_geometry(run_dir: Path) -> None:
    """F1: 2D projection of per-class JB displacement at L15, pos -2.

    Plot plane:
        x = projection on r̂  (refusal direction; harmful at 0, harmless at -1)
        y = projection on PC1 of orthogonal residuals across the 5 jb classes
    All quantities normalised to ‖r̂‖ for unitless axes.

    Layers (back-to-front):
        1. dashed grey reference axis  bare → harmless
        2. faint per-prompt clouds for each jb_class (50 prompts each, alpha 0.18)
           — visually addresses the "single direction per class" concern
        3. ctrl-class centroids as hollow markers (should sit near bare)
        4. jb-class arrows + bold centroids with cosine + magnitude annotations
        5. bare and harmless reference centroids (large X markers)

    Outputs PDF (vector, for paper inclusion) + PNG (for previews).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import torch
    except ImportError:
        print("[skip] F1: matplotlib/numpy/torch unavailable")
        return

    res_path = run_dir / "02b_stats" / "residuals_L15_per_cond.pt"
    dir_path = run_dir / "01_direction" / "refusal_direction.pt"
    align_path = run_dir / "02b_stats" / "direction_alignment.json"
    for p in (res_path, dir_path, align_path):
        if not p.exists():
            print(f"[skip] F1: missing input {p}")
            return

    residuals = torch.load(res_path, map_location="cpu", weights_only=False)
    direction_data = torch.load(dir_path, map_location="cpu", weights_only=False)
    align = json.load(open(align_path))

    # residuals tensors are stored as (n_prompts=50, n_positions=3, d_model=2560)
    # with positions in target order [-5, -3, -2]. Decision token = pos -2 = index 2.
    POS_IDX = 2
    target_positions = align.get("metadata", {}).get("target_positions")
    if target_positions and target_positions[POS_IDX] != -2:
        print(f"[skip] F1: unexpected position layout {target_positions}; "
              "expected pos -2 at index 2")
        return

    r_hat = direction_data["direction_pos-2_layer15"].numpy().astype(np.float64)
    r_hat_norm = float(align["metadata"]["r_hat_norm"])

    # Per-prompt activations & per-class centroids at pos -2.
    per_prompt = {cond: residuals[cond][:, POS_IDX, :].numpy().astype(np.float64)
                  for cond in residuals}
    centroids = {cond: per_prompt[cond].mean(0) for cond in per_prompt}
    c_bare = centroids["bare"]

    # Build the orthogonal axis: PC1 of *centered* {jb_class_centroid − bare_centroid}_⊥r̂.
    # Centering before SVD removes the shared orthogonal component (which projects all 5
    # classes to the same y); the resulting PC1 is the direction of *maximal between-class
    # spread*, so each class lands at a distinct y-coord.
    disp_jb = np.stack([centroids[f"jb_{c}"] - c_bare for c in JB_CLASSES])  # (5, d)
    proj_r = disp_jb @ r_hat                                                  # (5,)
    orth = disp_jb - proj_r[:, None] * r_hat[None, :]                         # (5, d)
    orth_centered = orth - orth.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(orth_centered, full_matrices=False)
    v_orth = vt[0]
    # Sanity: v_orth must be orthogonal to r_hat by construction.
    assert abs(float(v_orth @ r_hat)) < 1e-6, "v_orth not orthogonal to r_hat"
    # Sign convention: orient so 'fiction' is in the upper half — purely visual.
    if (disp_jb[JB_CLASSES.index("fiction")] @ v_orth) < 0:
        v_orth = -v_orth

    def to_2d(arr: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
        """Project (..., d) array of activations into (x, y) in r̂-units, relative to bare.

        x is projection onto the *harmless* direction (-r̂), so jailbreak displacements
        and the harmless centroid both have x > 0 (rightward = away from refusal).
        """
        d = arr - c_bare
        x = -(d @ r_hat) / r_hat_norm   # projection on -r̂; harmless = +1
        y = (d @ v_orth) / r_hat_norm
        return x, y

    # ---- Plot --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.0, 5.6))

    # 1) Reference axis  bare(0,0) → harmless(1,0) along -r̂.
    # Annotation is *above* the dashed line with a white bbox so it stays legible even
    # when JB clouds / arrows pass over it.
    ax.plot([0, 1], [0, 0], ls="--", color="#888888", lw=1.2, alpha=0.75, zorder=1)
    ax.annotate(
        r"$-\hat{r}$  axis (harmful $\to$ harmless)",
        xy=(0.5, 0), xytext=(0, 8), textcoords="offset points",
        ha="center", va="bottom", fontsize=8.5, color="#555555", style="italic",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.85),
        zorder=4,
    )

    # 2) Per-prompt clouds for jb conditions (50 dots / class — shows within-class spread)
    for c in JB_CLASSES:
        color = JB_CLASS_COLORS[c]
        xs, ys = to_2d(per_prompt[f"jb_{c}"])
        ax.scatter(xs, ys, c=color, s=22, alpha=0.30, zorder=2,
                   edgecolors="none", rasterized=True)

    # 3) Ctrl centroids as hollow markers (should sit close to bare, validating that
    #    length-matched benign prefixes don't move the residual along -r̂).
    for c in JB_CLASSES:
        color = JB_CLASS_COLORS[c]
        xc, yc = to_2d(centroids[f"ctrl_{c}"][None, :])
        ax.scatter(float(xc[0]), float(yc[0]), s=70, facecolors="white",
                   edgecolors=color, linewidth=1.4, alpha=0.85, marker="o",
                   zorder=3)

    # 4) JB arrows + bold centroids
    jb_pts = {}
    for c in JB_CLASSES:
        color = JB_CLASS_COLORS[c]
        xc, yc = to_2d(centroids[f"jb_{c}"][None, :])
        xc, yc = float(xc[0]), float(yc[0])
        jb_pts[c] = (xc, yc)
        ax.annotate(
            "", xy=(xc, yc), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0, alpha=0.95,
                            mutation_scale=16, shrinkA=4, shrinkB=2),
            zorder=4,
        )
        ax.scatter([xc], [yc], s=140, c=color, edgecolors="white",
                   linewidth=1.5, zorder=5)

    # 5) Reference centroids: bare(0,0) and synthesised harmless(1,0)
    ax.scatter([0], [0], s=180, c="#1A1A1A", edgecolors="white",
               linewidth=1.5, marker="X", zorder=6)
    ax.annotate("Harmful\n(bare)", xy=(0, 0), xytext=(-10, -2),
                textcoords="offset points", fontsize=10, fontweight="bold",
                ha="right", va="center")
    ax.scatter([1], [0], s=180, c="#1F8A5C", edgecolors="white",
               linewidth=1.5, marker="X", zorder=6)
    ax.annotate("Harmless\ncentroid", xy=(1, 0), xytext=(10, -2),
                textcoords="offset points", fontsize=10, fontweight="bold",
                ha="left", va="center")

    # Per-class label placement: stacked callout column on the right.
    # No leader lines — colour coding alone matches each label box to its arrow / cloud.
    # Labels are tightened (y_top/y_bot ±0.32) so the column fits inside the plot bounds.
    xs_centroids = [jb_pts[c][0] for c in JB_CLASSES]
    ys_centroids = [jb_pts[c][1] for c in JB_CLASSES]
    plot_xmax = max(xs_centroids + [1.0]) + 0.20  # right edge of plot data
    label_x = plot_xmax + 0.30                     # callout column x in data coords
    sorted_cls = sorted(JB_CLASSES, key=lambda c: -jb_pts[c][1])
    y_top, y_bot = 0.32, -0.32
    n = len(sorted_cls)
    for i, c in enumerate(sorted_cls):
        meta = align["per_class"][c]["pos_minus_2"]
        cos_v = float(meta["cos_neg_r_hat_r_jb"])
        mag_v = float(meta["mag_ratio_r_jb"])
        label = (f"{JB_CLASS_LABEL[c]}\n"
                 f"cos={cos_v:.2f},  $\\|r_{{jb}}\\|/\\|\\hat r\\|$={mag_v:.2f}")
        ly = y_top - i * (y_top - y_bot) / (n - 1)
        ax.text(
            label_x, ly, label,
            fontsize=9, color=JB_CLASS_COLORS[c], fontweight="bold",
            ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.30", fc="white",
                      ec=JB_CLASS_COLORS[c], lw=0.9, alpha=0.95),
            zorder=7,
        )

    # Legend (placed below the plot to avoid overlapping with classes)
    ctrl_handle = plt.Line2D([], [], marker="o", color="white", linestyle="",
                             markerfacecolor="white", markeredgecolor="#444444",
                             markersize=8, markeredgewidth=1.4,
                             label="ctrl centroids (length-matched, no JB semantic)")
    cloud_handle = plt.Line2D([], [], marker="o", color="white", linestyle="",
                              markerfacecolor="#888888", markeredgecolor="none",
                              markersize=5, alpha=0.55,
                              label="individual JB prompts (50 / class)")
    ax.legend(handles=[ctrl_handle, cloud_handle], loc="lower left",
              fontsize=8.5, framealpha=0.92, handletextpad=0.6, borderpad=0.6,
              bbox_to_anchor=(0.0, 0.0))

    # Axes & limits — extend xlim to include the callout column.
    ax.axhline(y=0, color="gray", alpha=0.18, lw=0.5)
    ax.axvline(x=0, color="gray", alpha=0.18, lw=0.5)
    ax.set_xlim(-0.30, label_x + 0.55)
    ax.set_ylim(min(ys_centroids + [-0.05]) - 0.25,
                max(ys_centroids + [+0.05]) + 0.30)
    ax.set_xlabel(r"Projection on $-\hat{r}$  (units of $\|\hat{r}\|$;  harmless direction $\to$)",
                  fontsize=11)
    ax.set_ylabel(r"Class-separating orthogonal axis  ($\|\hat{r}\|$ units)",
                  fontsize=11)
    ax.set_title(
        r"Per-class jailbreak displacement at $\ell=15$, position $-2$",
        fontsize=12, pad=8,
    )
    ax.grid(True, alpha=0.18)
    # Use auto aspect — cosines / magnitudes are reported in the labels themselves, so
    # geometric exactness in the rendering is not required and gives us more chart area.
    ax.set_aspect("auto")

    fig.tight_layout()
    out_pdf = FIG_DIR / "F1_per_class_geometry.pdf"
    out_png = FIG_DIR / "F1_per_class_geometry.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[wrote] {out_pdf}")
    print(f"[wrote] {out_png}")


# ==========================================================================
# F1B — Appendix: raw vs semantic per-class cosines (controlled-dataset payoff)
# ==========================================================================

def fig1b_raw_vs_semantic_cosines(run_dir: Path) -> None:
    """F1B (appendix): cos(-r̂, r_jb) raw vs semantic per class.

    Raw direction:      r_jb       = mean(h_jb_c)  − mean(h_bare)        (Ball convention)
    Semantic direction: r_jb_sem   = mean(h_jb_c)  − mean(h_ctrl_c)     (controlled)

    The raw direction conflates the jailbreak semantic with the prefix-length
    effect (any prefix shifts residuals slightly even without JB semantic).
    The controlled dataset's signature contribution is the ability to disentangle
    these two via the matched ctrl prefixes. Plotting both side-by-side surfaces
    the asymmetry (some classes' apparent harmless-shift is largely prefix-driven).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[skip] F1B: matplotlib/numpy unavailable")
        return

    align_path = run_dir / "02b_stats" / "direction_alignment.json"
    if not align_path.exists():
        print(f"[skip] F1B: missing {align_path}")
        return
    align = json.load(open(align_path))

    raw_cos = []
    sem_cos = []
    for c in JB_CLASSES:
        meta = align["per_class"][c]["pos_minus_2"]
        raw_cos.append(float(meta["cos_neg_r_hat_r_jb"]))
        sem_cos.append(float(meta["cos_neg_r_hat_r_jb_sem"]))

    x = np.arange(len(JB_CLASSES))
    w = 0.38

    fig, ax = plt.subplots(figsize=(8.4, 4.6))

    raw_bars = ax.bar(x - w/2, raw_cos, w,
                      color=[JB_CLASS_COLORS[c] for c in JB_CLASSES],
                      edgecolor="black", linewidth=0.5,
                      label=r"raw  $r_{jb} = h_{jb} - h_{\mathrm{bare}}$")
    sem_bars = ax.bar(x + w/2, sem_cos, w,
                      color=[JB_CLASS_COLORS[c] for c in JB_CLASSES],
                      edgecolor="black", linewidth=0.5, hatch="///",
                      alpha=0.85,
                      label=r"semantic  $r_{jb}^{\mathrm{sem}} = h_{jb} - h_{\mathrm{ctrl}}$")

    # Bar value labels
    for b, v in zip(raw_bars, raw_cos):
        ax.annotate(f"{v:+.2f}", xy=(b.get_x() + b.get_width()/2, v),
                    xytext=(0, 3 if v >= 0 else -10), textcoords="offset points",
                    ha="center", fontsize=8.5, fontweight="bold")
    for b, v in zip(sem_bars, sem_cos):
        ax.annotate(f"{v:+.2f}", xy=(b.get_x() + b.get_width()/2, v),
                    xytext=(0, 3 if v >= 0 else -10), textcoords="offset points",
                    ha="center", fontsize=8.5)

    ax.axhline(y=0, color="black", lw=0.6)
    ax.axhline(y=1, color="#1F8A5C", lw=0.8, ls=":", alpha=0.6,
               label=r"perfect alignment with $-\hat r$ (=1)")

    ax.set_xticks(x)
    ax.set_xticklabels([JB_CLASS_LABEL[c] for c in JB_CLASSES], fontsize=10)
    ax.set_ylabel(r"$\cos(-\hat r,\ r_{jb})$", fontsize=11)
    ax.set_ylim(-0.95, 1.10)
    ax.set_title(
        r"Raw vs semantic per-class jailbreak alignment with $-\hat r$"
        "  (L15, pos $-2$)\n"
        r"Semantic = controlled for prefix-length effect via the matched ctrl prompts",
        fontsize=10.5, pad=8,
    )
    ax.legend(loc="lower left", fontsize=9, framealpha=0.92)
    ax.grid(True, alpha=0.20, axis="y")

    fig.tight_layout()
    out_pdf = FIG_DIR / "F1B_raw_vs_semantic_cosines.pdf"
    out_png = FIG_DIR / "F1B_raw_vs_semantic_cosines.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[wrote] {out_pdf}")
    print(f"[wrote] {out_png}")


def main():
    print("[generate_paper_figures] collecting data...")

    # F1 — geometry figure (the v2 ICML lead figure). Independent of the ablation rows.
    fig1_per_class_geometry(RESULTS / GEOMETRY_RUN)
    # F1B — appendix: raw vs semantic cosines, surfacing the controlled-dataset payoff.
    fig1b_raw_vs_semantic_cosines(RESULTS / GEOMETRY_RUN)

    rows = collect_all_ablations()
    print(f"[generate_paper_figures] {len(rows)} (ablation × config) rows total")

    write_table(rows)
    write_ci_json(rows)

    # Direction intervention rate (Stage 06 weighted) — load from causal_summary if available
    causal = load_causal_summary(RESULTS / GEOMETRY_RUN)
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
