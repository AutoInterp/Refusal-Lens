"""
Stage 02b: Statistical Analysis + Plots
=======================================
Computes paired statistics and generates publication-quality plots from the
Stage 02 attribution results produced by the multi+single two-graph scheme.

For each prompt × condition, Stage 02 now emits both a multi-target graph
(targets = L15 at positions [-5, -3, -2] — the template anchors) and a
single-target baseline graph (target = L15 @ pos=-2 only). This stage
computes paired stats for every combination of:

    modes        = {multi, single}                       (graph target set)
    comparisons  = {vs_bare, vs_ctrl, ctrl_vs_bare}      (what to compare)
    jb classes   = {roleplay, fiction, analytical,
                    completion, cognitive_reframe}

    vs_bare      = bare             ↔ jb_<class>   (legacy comparison)
    vs_ctrl      = ctrl_<class>     ↔ jb_<class>   (token-matched; cleanest)
    ctrl_vs_bare = bare             ↔ ctrl_<class> (sanity: ctrl ≈ bare?)

Statistics per block: Wilcoxon signed-rank, paired t-test, Cohen's d,
bootstrap 95% CI on Δnet, dual-mechanism decomposition (dPos/dNeg).

Plots emitted for multi mode (the headline); single mode is captured in the
JSON + markdown report but only the core distribution plot is rendered for it.

Inputs:  02_attribution/attribution_results.json
         02_attribution/feature_comparison_aggregate.json
         01_direction/direction_metadata.json  (for separation + cosine plots)
Outputs: 02b_stats/statistical_analysis.json
         02b_stats/EXPERIMENT_SUMMARY.md
         02b_stats/*.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: F401
from utils import save_json, load_json, get_stage_dir


# Class set under the controlled dataset. Matches Stage 02.
CONTROLLED_CLASSES = ("roleplay", "fiction", "analytical", "completion", "cognitive_reframe")

# Graph modes. "multi" is the headline (template-anchored 3-position target);
# "single" is the causally-verified pos=-2 baseline.
MODES = ("multi", "single")

# (comparison_key, baseline_cond_template, treatment_cond_template).
# {cls} is expanded against CONTROLLED_CLASSES per-class.
COMPARISONS = (
    ("vs_bare",      "bare",        "jb_{cls}"),
    ("vs_ctrl",      "ctrl_{cls}",  "jb_{cls}"),
    ("ctrl_vs_bare", "bare",        "ctrl_{cls}"),
)


# ======================================================================
# Schema helpers
# ======================================================================


def _get_metrics(row: dict, cond_name: str, mode: str) -> tuple[float, float, float] | None:
    """Pull (net, pos_sum, neg_sum) for a prompt row's condition/mode.

    Handles three historical shapes:
      - Post-refactor:  conditions[cond_name].graphs[mode].{net,pos_sum,neg_sum}
      - Pre-refactor:   conditions[cond_name].{net,pos_sum,neg_sum}  (single-target)
      - Pre-ctrl:       "jb_<cls>" didn't exist — the JB condition was keyed
                        as "<cls>" directly. This fallback lets the new
                        vs_bare comparison still work on pre-ctrl data.

    Returns None when the condition is missing, errored, or not applicable
    to legacy data (e.g. "ctrl_<cls>" on a pre-ctrl run).
    """
    conds = row.get("conditions", row)
    cond = conds.get(cond_name)
    # Legacy condition-name fallback: "jb_fiction" → "fiction".
    # (Pre-ctrl runs used the bare class name as the JB condition key.)
    if cond is None and cond_name.startswith("jb_"):
        cond = conds.get(cond_name.removeprefix("jb_"))
    if not isinstance(cond, dict):
        return None
    # New schema: dig into .graphs[mode]
    if "graphs" in cond:
        g = cond["graphs"].get(mode)
        if not isinstance(g, dict) or "error" in g:
            return None
        return (
            float(g.get("net", 0.0)),
            float(g.get("pos_sum", 0.0)),
            float(g.get("neg_sum", 0.0)),
        )
    # Legacy flat condition. Pre-refactor there was only one graph per
    # condition (single-target), so these metrics apply only to mode="single".
    # For mode="multi" on legacy data, return None — the caller detects that
    # mode isn't present and skips it.
    if mode != "single":
        return None
    if "error" in cond or "net" not in cond:
        return None
    return (
        float(cond.get("net", 0.0)),
        float(cond.get("pos_sum", 0.0)),
        float(cond.get("neg_sum", 0.0)),
    )


def _detect_modes(results: list[dict]) -> list[str]:
    """Return graph modes present in the data. Legacy flat schema (no
    ``graphs`` key anywhere) is treated as single-target → ["single"]."""
    seen: set[str] = set()
    for row in results:
        for cond in row.get("conditions", row).values():
            if isinstance(cond, dict) and "graphs" in cond:
                seen.update(cond["graphs"].keys())
    if not seen:
        return ["single"]
    # Preserve canonical order: multi, single, then any custom names alphabetically.
    ordered = [m for m in MODES if m in seen]
    ordered.extend(sorted(seen - set(MODES)))
    return ordered


def _gather_pairs(
    results: list[dict], a_cond: str, b_cond: str, mode: str,
) -> dict[str, np.ndarray]:
    """Collect aligned arrays of (net, pos_sum, neg_sum) for pairs where both
    conditions are present and error-free. Returns a dict of numpy arrays."""
    a_nets, b_nets, a_pos, b_pos, a_neg, b_neg = [], [], [], [], [], []
    for row in results:
        av = _get_metrics(row, a_cond, mode)
        bv = _get_metrics(row, b_cond, mode)
        if av is None or bv is None:
            continue
        a_nets.append(av[0]); a_pos.append(av[1]); a_neg.append(av[2])
        b_nets.append(bv[0]); b_pos.append(bv[1]); b_neg.append(bv[2])
    return {
        "a_nets": np.array(a_nets), "b_nets": np.array(b_nets),
        "a_pos":  np.array(a_pos),  "b_pos":  np.array(b_pos),
        "a_neg":  np.array(a_neg),  "b_neg":  np.array(b_neg),
    }


# ======================================================================
# Paired stats
# ======================================================================


def _paired_stats(
    pairs: dict[str, np.ndarray], n_bootstrap: int, rng: np.random.RandomState,
) -> dict | None:
    """Compute paired-sample statistics comparing treatment (b) to baseline (a).

    Returns None if fewer than 3 pairs (insufficient for statistical tests).

    Convention: positive `mean_delta` = treatment has larger net attribution
    than baseline, i.e. more refusal signal. For `vs_bare` and `vs_ctrl` where
    the baseline refuses and JB may not, expect negative mean_delta for most
    classes.
    """
    from scipy import stats as scipy_stats

    a = pairs["a_nets"]; b = pairs["b_nets"]
    n = len(a)
    if n < 3:
        return None

    deltas = b - a  # treatment minus baseline

    # Wilcoxon signed-rank — non-parametric
    try:
        w_stat, w_pval = scipy_stats.wilcoxon(a, b)
    except ValueError:
        w_stat, w_pval = float("nan"), float("nan")

    # Paired t-test
    t_stat, t_pval = scipy_stats.ttest_rel(a, b)

    # Cohen's d (paired)
    d_mean = float(np.mean(deltas))
    d_std = float(np.std(deltas, ddof=1))
    cohens_d = d_mean / d_std if d_std > 0 else 0.0

    # Bootstrap 95% CI on mean delta
    boot_means = np.array([
        rng.choice(deltas, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))

    # Dual-mechanism decomposition (change in pro-refusal, anti-refusal halves)
    a_pos_mean = float(np.mean(pairs["a_pos"]))
    b_pos_mean = float(np.mean(pairs["b_pos"]))
    a_neg_mean = float(np.mean(pairs["a_neg"]))
    b_neg_mean = float(np.mean(pairs["b_neg"]))
    d_pos = b_pos_mean - a_pos_mean
    d_neg = b_neg_mean - a_neg_mean

    return {
        "n_pairs": int(n),
        "a_mean_net": float(np.mean(a)), "a_std_net": float(np.std(a)),
        "b_mean_net": float(np.mean(b)), "b_std_net": float(np.std(b)),
        "mean_delta": d_mean,
        "std_delta": d_std,
        "pct_change": (d_mean / np.mean(a) * 100) if np.mean(a) != 0 else 0.0,
        "wilcoxon_stat": float(w_stat), "wilcoxon_pval": float(w_pval),
        "ttest_stat": float(t_stat), "ttest_pval": float(t_pval),
        "cohens_d": float(cohens_d),
        "ci_95_low": ci_low, "ci_95_high": ci_high,
        "n_treatment_lower": int((b < a).sum()),
        "a_mean_pos": a_pos_mean, "b_mean_pos": b_pos_mean,
        "a_mean_neg": a_neg_mean, "b_mean_neg": b_neg_mean,
        "d_pos": d_pos, "d_neg": d_neg,
        "d_pos_pct": (d_pos / a_pos_mean * 100) if a_pos_mean != 0 else 0.0,
        "d_neg_pct": (d_neg / abs(a_neg_mean) * 100) if a_neg_mean != 0 else 0.0,
    }


def _dominance_label(d_pos: float, d_neg: float) -> str:
    """Classify the dominant mechanism based on relative pro/anti deltas."""
    if abs(d_pos) > 2 * abs(d_neg):
        return "Dampening-dominant" if d_pos < 0 else "Pro-refusal recruitment"
    if abs(d_neg) > 2 * abs(d_pos):
        return "Amplification-dominant" if d_neg > 0 else "Anti-suppression"
    return "Balanced"


def _sig_marker(pval: float) -> str:
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return ""


# ======================================================================
# Runner
# ======================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 02b statistical analysis + plots")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--n-bootstrap", type=int, default=10000,
        help="Bootstrap resamples for 95% CI on mean delta.",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation (stats JSON + markdown only).",
    )
    return parser.parse_args()


def _compute_all_stats(
    results: list[dict], n_bootstrap: int, modes: list[str],
) -> dict[str, dict[str, dict[str, dict]]]:
    """Iterate modes × comparisons × classes and compute paired stats for each.
    Missing/insufficient combinations are omitted rather than stubbed out.
    Deterministic bootstrap via a shared seeded RandomState.
    """
    rng = np.random.RandomState(42)
    out: dict = {}
    for mode in modes:
        per_mode: dict = {}
        for comp_name, a_tmpl, b_tmpl in COMPARISONS:
            per_comp: dict = {}
            for cls in CONTROLLED_CLASSES:
                a_cond = a_tmpl.format(cls=cls)
                b_cond = b_tmpl.format(cls=cls)
                pairs = _gather_pairs(results, a_cond, b_cond, mode)
                stats = _paired_stats(pairs, n_bootstrap, rng)
                if stats is None:
                    continue
                stats["baseline_cond"] = a_cond
                stats["treatment_cond"] = b_cond
                stats["dominance"] = _dominance_label(stats["d_pos"], stats["d_neg"])
                per_comp[cls] = stats
            if per_comp:
                per_mode[comp_name] = per_comp
        if per_mode:
            out[mode] = per_mode
    return out


# ======================================================================
# Reporting (print tables + markdown)
# ======================================================================


def _print_stats_table(stats_block: dict, title: str) -> None:
    """Print one comparison × 5-class stats table to stdout."""
    if not stats_block:
        return
    print(f"\n  {title}")
    print(
        f"  {'Class':>18} | {'N':>3} | {'Base':>7} | {'Treat':>7} | "
        f"{'Delta':>7} | {'%Chg':>7} | {'p (W)':>10} | {'d':>7} | {'95% CI':>18}"
    )
    print(f"  {'-'*18}-+-{'-'*3}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-"
          f"{'-'*7}-+-{'-'*10}-+-{'-'*7}-+-{'-'*18}")
    for cls in CONTROLLED_CLASSES:
        if cls not in stats_block:
            continue
        s = stats_block[cls]
        sig = _sig_marker(s["wilcoxon_pval"]) or "ns"
        print(
            f"  {cls:>18} | {s['n_pairs']:3d} | "
            f"{s['a_mean_net']:+7.1f} | {s['b_mean_net']:+7.1f} | "
            f"{s['mean_delta']:+7.1f} | {s['pct_change']:+6.1f}% | "
            f"{s['wilcoxon_pval']:10.4g}{sig:>3} | "
            f"{s['cohens_d']:+7.2f} | "
            f"[{s['ci_95_low']:+.1f}, {s['ci_95_high']:+.1f}]"
        )


def _print_dual_mechanism_table(stats_block: dict, title: str) -> None:
    if not stats_block:
        return
    print(f"\n  {title} — Dual Mechanism Decomposition")
    print(
        f"  {'Class':>18} | {'dPos':>10} | {'dNeg':>10} | {'Net':>8} | Dominant"
    )
    print(f"  {'-'*18}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*25}")
    for cls in CONTROLLED_CLASSES:
        if cls not in stats_block:
            continue
        s = stats_block[cls]
        print(
            f"  {cls:>18} | {s['d_pos']:+10.1f} | {s['d_neg']:+10.1f} | "
            f"{s['mean_delta']:+8.1f} | {s['dominance']}"
        )


def _markdown_stats_table(stats_block: dict, title: str) -> list[str]:
    if not stats_block:
        return []
    lines = [
        f"",
        f"### {title}",
        f"",
        f"| Class | N | Baseline | Treatment | ΔNet | % Change | p (Wilcoxon) | Cohen's d | 95% CI | Dominant |",
        f"|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cls in CONTROLLED_CLASSES:
        if cls not in stats_block:
            continue
        s = stats_block[cls]
        sig = _sig_marker(s["wilcoxon_pval"])
        lines.append(
            f"| **{cls}** | {s['n_pairs']} | "
            f"{s['a_mean_net']:+.1f} | {s['b_mean_net']:+.1f} | "
            f"{s['mean_delta']:+.1f} | {s['pct_change']:+.1f}% | "
            f"{s['wilcoxon_pval']:.4g}{sig} | "
            f"{s['cohens_d']:+.2f} | "
            f"[{s['ci_95_low']:+.1f}, {s['ci_95_high']:+.1f}] | "
            f"{s['dominance']} |"
        )
    return lines


# ======================================================================
# Plots — multi mode only (single mode lives in JSON / markdown for now)
# ======================================================================


def _plot_class_comparison(
    results: list[dict], out_dir: Path, mode: str = "multi",
    filename_suffix: str = "",
) -> Path | None:
    """Grouped bar chart: mean net + pos_sum + neg_sum for each of the 11
    conditions (bare, 5 ctrl_<cls>, 5 jb_<cls>), under the given graph mode."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_conds = ["bare"]
    for cls in CONTROLLED_CLASSES:
        all_conds.append(f"ctrl_{cls}")
        all_conds.append(f"jb_{cls}")

    means, stds, pos_means, neg_means = [], [], [], []
    n_per = []
    for cond in all_conds:
        nets, poss, negs = [], [], []
        for row in results:
            m = _get_metrics(row, cond, mode)
            if m is None:
                continue
            nets.append(m[0]); poss.append(m[1]); negs.append(m[2])
        n_per.append(len(nets))
        means.append(np.mean(nets) if nets else 0.0)
        stds.append(np.std(nets) if nets else 0.0)
        pos_means.append(np.mean(poss) if poss else 0.0)
        neg_means.append(np.mean(negs) if negs else 0.0)

    if max(n_per) == 0:
        return None

    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(all_conds))
    w = 0.26
    ax.bar(x - w, pos_means, w, label="pos_sum (pro-refusal)", color="#d9534f", alpha=0.85)
    ax.bar(x,     means,     w, label="net",                   color="#5cb85c",
           alpha=0.85, yerr=stds, capsize=3)
    ax.bar(x + w, neg_means, w, label="neg_sum (anti-refusal)", color="#5bc0de", alpha=0.85)
    # Color-code tick labels: bare=black, ctrl=gray, jb=red
    ax.set_xticks(x)
    labels = ["bare"]
    for cls in CONTROLLED_CLASSES:
        labels.append(f"ctrl\n{cls}")
        labels.append(f"jb\n{cls}")
    ax.set_xticklabels(labels, fontsize=8)
    for tick, cond in zip(ax.get_xticklabels(), all_conds):
        if cond.startswith("ctrl_"):
            tick.set_color("#555")
        elif cond.startswith("jb_"):
            tick.set_color("#b32")
    ax.set_ylabel("Attribution sum")
    ax.set_title(
        f"Net + Split Attribution by Condition  —  mode={mode}  "
        f"(n={max(n_per)} prompts per condition)"
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(loc="upper right")
    plt.tight_layout()
    path = out_dir / f"class_comparison{filename_suffix}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _plot_effect_sizes(
    stats_by_mode: dict, out_dir: Path, mode: str = "multi",
    filename_suffix: str = "",
) -> Path | None:
    """Horizontal bar chart of Cohen's d for each comparison × class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if mode not in stats_by_mode:
        return None
    per_mode = stats_by_mode[mode]
    if not per_mode:
        return None

    comp_order = [c for c, _, _ in COMPARISONS if c in per_mode]
    fig, axes = plt.subplots(1, len(comp_order), figsize=(5 * len(comp_order), 5),
                             sharey=True)
    if len(comp_order) == 1:
        axes = [axes]

    for ax, comp_name in zip(axes, comp_order):
        block = per_mode.get(comp_name, {})
        cls_list = [c for c in CONTROLLED_CLASSES if c in block]
        ds = [block[c]["cohens_d"] for c in cls_list]
        colors = ["#d9534f" if d < 0 else "#5bc0de" for d in ds]
        ax.barh(range(len(cls_list)), ds, color=colors, alpha=0.8)
        ax.set_yticks(range(len(cls_list)))
        ax.set_yticklabels(cls_list, fontsize=9)
        ax.set_xlabel("Cohen's d")
        ax.set_title(f"{comp_name} ({mode})")
        ax.axvline(0, color="black", linewidth=0.5)
        for thresh, style in [(-0.2, ":"), (-0.5, "--"), (-0.8, "-")]:
            ax.axvline(thresh,  color="gray", linewidth=0.4, linestyle=style)
            ax.axvline(-thresh, color="gray", linewidth=0.4, linestyle=style)

    fig.suptitle(f"Effect Sizes  —  mode={mode}", fontsize=13)
    plt.tight_layout()
    path = out_dir / f"effect_sizes{filename_suffix}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _plot_distribution_by_class(
    results: list[dict], out_dir: Path, mode: str = "multi",
    filename_suffix: str = "",
) -> Path | None:
    """Violin+box+scatter of net attribution for each of the 11 conditions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_conds = ["bare"]
    for cls in CONTROLLED_CLASSES:
        all_conds.append(f"ctrl_{cls}")
        all_conds.append(f"jb_{cls}")

    data = []
    for cond in all_conds:
        nets = []
        for row in results:
            m = _get_metrics(row, cond, mode)
            if m is not None:
                nets.append(m[0])
        data.append(nets)
    if all(len(d) == 0 for d in data):
        return None

    fig, ax = plt.subplots(figsize=(16, 7))
    positions = np.arange(1, len(all_conds) + 1)
    # Only violinplot non-empty data
    non_empty_idx = [i for i, d in enumerate(data) if len(d) >= 2]
    non_empty_data = [data[i] for i in non_empty_idx]
    non_empty_pos = [positions[i] for i in non_empty_idx]
    if non_empty_data:
        parts = ax.violinplot(
            non_empty_data, positions=non_empty_pos,
            showmeans=False, showmedians=False, showextrema=False,
        )
        for i, pc in enumerate(parts["bodies"]):
            cond = all_conds[non_empty_idx[i]]
            if cond == "bare":
                color = "#7f7f7f"
            elif cond.startswith("ctrl_"):
                color = "#5bc0de"
            else:
                color = "#d9534f"
            pc.set_facecolor(color)
            pc.set_alpha(0.45)
            pc.set_edgecolor("black")
            pc.set_linewidth(0.6)

    ax.boxplot(
        data, positions=positions, widths=0.18, patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(facecolor="white", alpha=0.85, edgecolor="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(marker="o", markersize=3, alpha=0.4, markerfacecolor="gray"),
    )

    jitter_rng = np.random.RandomState(42)
    for pos, vals in zip(positions, data):
        if not vals:
            continue
        jitter = jitter_rng.uniform(-0.08, 0.08, len(vals))
        ax.scatter(
            np.full(len(vals), pos) + jitter, vals,
            alpha=0.35, s=10, color="#333", linewidths=0,
        )

    bare_mean = float(np.mean(data[0])) if data[0] else 0.0
    ax.axhline(bare_mean, color="gray", linestyle="--", alpha=0.6,
               label=f"bare mean ({bare_mean:+.1f})")
    ax.axhline(0, color="black", linewidth=0.5)

    labels = ["bare"] + [
        f"{kind}\n{cls}" for cls in CONTROLLED_CLASSES for kind in ("ctrl", "jb")
    ]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    for tick, cond in zip(ax.get_xticklabels(), all_conds):
        if cond.startswith("ctrl_"):
            tick.set_color("#555")
        elif cond.startswith("jb_"):
            tick.set_color("#b32")
    ax.set_ylabel("Net attribution")
    ax.set_title(f"Net-Attribution Distribution by Condition  —  mode={mode}")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / f"distribution_by_class{filename_suffix}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _plot_separation_by_layer(direction_meta: dict, out_dir: Path) -> Path | None:
    if not direction_meta or "layers" not in direction_meta:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers_dict = direction_meta["layers"]
    layer_ids = sorted(int(k) for k in layers_dict.keys())
    separations = [layers_dict[str(lid)]["separation"] for lid in layer_ids]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(layer_ids, separations, marker="o", linewidth=2,
            color="#2c3e50", label="separation")

    best_sep = direction_meta.get("best_separation_layer")
    best_causal = direction_meta.get("best_causal_layer")
    if best_sep is not None:
        ax.axvline(best_sep, color="#d9534f", linestyle="--", alpha=0.7,
                   label=f"best separation (L{best_sep})")
    if best_causal is not None:
        ax.axvline(best_causal, color="#5cb85c", linestyle="--", alpha=0.7,
                   label=f"best causal (L{best_causal})")

    if 33 in layer_ids and layers_dict.get("33", {}).get("separation", 0) < 1000:
        ax.annotate(
            "L33: pre-RMSNorm artifact",
            xy=(33, layers_dict["33"]["separation"]),
            xytext=(27, max(separations) * 0.35),
            arrowprops=dict(arrowstyle="->", color="gray"),
            fontsize=9, color="gray",
        )
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Separation  (|μ(harmful) − μ(harmless)|)")
    ax.set_title("Refusal Direction Separation by Layer (pos=-2)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / "separation_by_layer.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _plot_cosine_heatmap(direction_meta: dict, out_dir: Path) -> Path | None:
    if not direction_meta or "cosine_matrix" not in direction_meta:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.array(direction_meta["cosine_matrix"])
    n = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(11, 9))
    vmax = max(abs(matrix.min()), abs(matrix.max()))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    for kl in [15, 25, 32]:
        if 0 <= kl < n:
            ax.axvline(kl, color="black", linewidth=0.8, alpha=0.4)
            ax.axhline(kl, color="black", linewidth=0.8, alpha=0.4)

    ax.set_xticks(np.arange(0, n, 2))
    ax.set_yticks(np.arange(0, n, 2))
    ax.set_xticklabels([f"L{i}" for i in range(0, n, 2)], fontsize=8)
    ax.set_yticklabels([f"L{i}" for i in range(0, n, 2)], fontsize=8)
    ax.set_xlabel("Layer j")
    ax.set_ylabel("Layer i")
    ax.set_title("Per-Layer Direction Similarity: cos(r̂_i, r̂_j)")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity")
    plt.tight_layout()
    path = out_dir / "cosine_heatmap.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ======================================================================
# Markdown report
# ======================================================================


def _build_report(
    results: list[dict], stats_by_mode: dict, direction_meta: dict | None,
) -> str:
    lines = [
        f"# Stage 02b Statistical Analysis",
        f"",
        f"- **Prompts**: {len(results)}",
        f"- **Model**: {config.MODEL_NAME}",
        f"- **Target**: L{config.MEASUREMENT_LAYER} (causal). Two graph modes:",
        f"  - **multi** — targets the template anchors [-5, -3, -2] "
        f"(`<end_of_turn>`, `<start_of_turn>`, `model`)",
        f"  - **single** — target pos=-2 only (verified causal position)",
        f"",
        f"Comparisons are run for each mode:",
        f"- `vs_bare`: bare ↔ jb_<class> — legacy JB-effect delta",
        f"- `vs_ctrl`: ctrl_<class> ↔ jb_<class> — token-matched, isolates JB semantics",
        f"- `ctrl_vs_bare`: bare ↔ ctrl_<class> — sanity (ctrl should track bare)",
        f"",
    ]

    for mode in MODES:
        if mode not in stats_by_mode:
            continue
        lines.append(f"## Mode: `{mode}`")
        for comp_name, _, _ in COMPARISONS:
            block = stats_by_mode[mode].get(comp_name)
            if not block:
                continue
            lines.extend(_markdown_stats_table(block, f"{mode} · {comp_name}"))

    if direction_meta:
        lines.extend([
            f"",
            f"## Direction (Stage 01) Summary",
            f"",
            f"- Best separation layer: **L{direction_meta.get('best_separation_layer')}** "
            f"(magnitude {direction_meta.get('layers', {}).get(str(direction_meta.get('best_separation_layer')), {}).get('separation', 'n/a')})",
            f"- Best causal layer: **L{direction_meta.get('best_causal_layer')}** "
            f"(used for attribution)",
        ])

    return "\n".join(lines) + "\n"


# ======================================================================
# Main
# ======================================================================


def main():
    args = parse_args()
    run_dir = args.run_dir
    out_dir = get_stage_dir(run_dir, "02b_stats")

    print("=" * 60)
    print("STAGE 02b: Statistical Analysis + Plots")
    print("=" * 60)

    # Load Stage 02 attribution results
    attr_path = run_dir / "02_attribution" / "attribution_results.json"
    if not attr_path.exists():
        print(f"  ERROR: {attr_path} not found. Run Stage 02 first.")
        sys.exit(1)
    raw = load_json(attr_path)
    results = raw if isinstance(raw, list) else raw["results"]
    meta = raw.get("metadata", {}) if isinstance(raw, dict) else {}
    print(f"  Loaded {len(results)} prompt results")
    if meta.get("modes"):
        print(f"  Attribution modes: {meta['modes']}")

    direction_meta_path = run_dir / "01_direction" / "direction_metadata.json"
    direction_meta = load_json(direction_meta_path) if direction_meta_path.exists() else None

    # Detect which graph modes are present in the data. Legacy (single-graph)
    # runs produce only "single". New two-graph runs produce both.
    modes = _detect_modes(results)
    print(f"  Detected modes: {modes}")

    # ---- Stats ----
    print("\n  Computing paired statistics across (mode × comparison × class)...")
    stats_by_mode = _compute_all_stats(results, args.n_bootstrap, modes)

    for mode in modes:
        if mode not in stats_by_mode:
            continue
        for comp_name, _, _ in COMPARISONS:
            block = stats_by_mode[mode].get(comp_name)
            if block:
                _print_stats_table(block, f"[{mode}] {comp_name}")
    for mode in modes:
        if mode not in stats_by_mode:
            continue
        vs_ctrl = stats_by_mode[mode].get("vs_ctrl")
        if vs_ctrl:
            _print_dual_mechanism_table(vs_ctrl, f"[{mode}] vs_ctrl")

    save_json(stats_by_mode, out_dir / "statistical_analysis.json")
    print(f"\n  Saved statistical_analysis.json")

    # ---- Plots ----
    no_plots = getattr(args, "no_plots", False)
    if not no_plots:
        print("\n  Generating plots...")
        # When only one mode is present (legacy or --skip-*-graph runs),
        # emit plots without a suffix so downstream consumers and existing
        # test assertions keep working. When both modes present, suffix
        # each file with _multi / _single for disambiguation.
        single_mode_run = len(modes) == 1
        for mode in modes:
            suffix = "" if single_mode_run else f"_{mode}"
            for plot_fn in (
                _plot_class_comparison,
                _plot_distribution_by_class,
            ):
                p = plot_fn(results, out_dir, mode=mode, filename_suffix=suffix)
                if p:
                    print(f"    Saved {p.name}")
            p = _plot_effect_sizes(stats_by_mode, out_dir, mode=mode, filename_suffix=suffix)
            if p:
                print(f"    Saved {p.name}")
        if direction_meta:
            for fn in (_plot_separation_by_layer, _plot_cosine_heatmap):
                p = fn(direction_meta, out_dir)
                if p:
                    print(f"    Saved {p.name}")

    # ---- Markdown report ----
    print("\n  Writing EXPERIMENT_SUMMARY.md...")
    report = _build_report(results, stats_by_mode, direction_meta)
    with open(out_dir / "EXPERIMENT_SUMMARY.md", "w") as f:
        f.write(report)

    print(f"\n  All outputs saved to {out_dir}/")
    print("DONE!")


if __name__ == "__main__":
    main()
