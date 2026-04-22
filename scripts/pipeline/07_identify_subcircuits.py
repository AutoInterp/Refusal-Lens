"""
Stage 07: Identify Functional Subcircuits (rule-based)
=======================================================
Pure set-logic subcircuit identification from Stage 04 outputs.
No GPU, no ML fitting — all derivations from conditions_seen and
feature_class_sets buckets.

Reads:
- 04_labels/feature_labels.json        (per-feature conditions_seen, layer, attribution)
- 04_labels/feature_class_sets.json    (bucket → feature → classes map +
                                        per_condition_top50 block from Task 7)

Writes to 07_subcircuits/:
- subcircuits.json              — canonical definitions + feature lists + per-subcircuit stats
- subcircuits_summary.json      — sizes + pairwise overlap matrix + jb_vs_ctrl_contrast
- subcircuits_treemap.png       — visual size breakdown
- subcircuits_by_layer.png      — stacked-bar-by-layer (one row per subcircuit)
- subcircuits_overlap.png       — pairwise overlap heatmap
- jb_vs_ctrl_contrast.png       — Task 10 NEW: per-class recruitment contrast
                                  (fraction of JB machinery that's genuinely
                                  JB-semantic vs. prefix-induced)
- jb_specific_by_layer.png      — Task 10 NEW: layer distribution of jb_specific_vs_ctrl
- SUBCIRCUITS_REPORT.md         — human-readable report

Subcircuits (original):
universal_refusal_core            bare + all 5 JB
canonical_pro_refusal             all 5 JB, no bare
sign_flip_convergent              sign_flipped in ≥3 JB classes
dampening_specialists             dampened in ≥3 JB classes
anti_refusal_amplifiers           amplified_anti in ≥3 JB classes
late_wave_layer24_32              any feature in L24–L32 (cross-cuts above)
{class}_exclusive (×5)            exactly one JB class, no bare

Ctrl-aware (new Apr 22 — require feature_class_sets.per_condition_top50 from Task 7):
ctrl_shared_refusal               in bare AND all 5 ctrl_*_top50 but NOT in all 5 jb_*_top50
ctrl_only                         in all 5 ctrl_*_top50 but not bare or any jb_*_top50
jb_{cls}_specific_vs_ctrl (×5)    in jb_{cls}_top50 but NOT in ctrl_{cls}_top50
                                  ("the cleanest JB-semantic subcircuit" per Georg)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: F401  (imported for side-effect / consistency)
from utils import get_stage_dir, load_json, save_json


JB_CLASSES = frozenset({"analytical", "cognitive_reframe", "completion", "fiction", "roleplay"})
ALL_CONDS = JB_CLASSES | {"bare"}

CONVERGENT_MIN_CLASSES = 3
LATE_WAVE_LO, LATE_WAVE_HI = 24, 32


def parse_args():
    p = argparse.ArgumentParser(description="Rule-based subcircuit identification")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--convergent-min", type=int, default=CONVERGENT_MIN_CLASSES)
    p.add_argument("--late-wave-start", type=int, default=LATE_WAVE_LO)
    p.add_argument("--late-wave-end", type=int, default=LATE_WAVE_HI)
    return p.parse_args()


# ------------------------------- rules ----------------------------------

def _jb_classes_seen(conds: set[str]) -> set[str]:
    """Extract JB classes from a conditions_seen set.

    Accepts both the new full-condition schema ('jb_fiction', 'ctrl_fiction')
    and the legacy flat schema where raw class names appeared directly ('fiction').
    Returns strict class tokens (no jb_ prefix) so callers can compare against
    JB_CLASSES.
    """
    jb = set()
    for c in conds:
        if c.startswith("jb_"):
            jb.add(c[3:])
        elif c in JB_CLASSES:
            jb.add(c)
    return jb


def _classify_bucket_classes(classes: list[str]) -> set[str]:
    """Map a `classes` list from a comparison bucket back to raw JB-class tokens.

    Class-sets under the new schema tag with 'jb_{cls}' / 'ctrl_{cls}' — only
    the jb_ tags count as 'JB-class seen' for convergent rules (ctrl-only tags
    correspond to ctrl_vs_bare, which isn't a JB comparison). Legacy flat tags
    (plain 'fiction') are kept.
    """
    out = set()
    for c in classes:
        if c.startswith("jb_"):
            out.add(c[3:])
        elif c in JB_CLASSES:
            out.add(c)
    return out


def build_universal_core(feature_labels: dict) -> list[str]:
    return [
        k for k, v in feature_labels.items()
        if "bare" in v.get("conditions_seen", [])
        and _jb_classes_seen(set(v.get("conditions_seen", []))) >= JB_CLASSES
    ]


def build_canonical_pro_refusal(feature_labels: dict) -> list[str]:
    return [
        k for k, v in feature_labels.items()
        if _jb_classes_seen(set(v.get("conditions_seen", []))) >= JB_CLASSES
        and "bare" not in v.get("conditions_seen", [])
    ]


def build_class_exclusive(feature_labels: dict) -> dict[str, list[str]]:
    exclusive: dict[str, list[str]] = {c: [] for c in JB_CLASSES}
    for k, v in feature_labels.items():
        conds = set(v.get("conditions_seen", []))
        if "bare" in conds:
            continue
        jb = _jb_classes_seen(conds)
        if len(jb) == 1:
            exclusive[next(iter(jb))].append(k)
    return exclusive


def build_convergent_bucket(class_sets: dict, bucket: str, min_classes: int) -> list[str]:
    features = class_sets.get("by_bucket", {}).get(bucket, {}).get("features", {})
    return [
        k for k, classes in features.items()
        if len(_classify_bucket_classes(classes)) >= min_classes
    ]


def build_late_wave(feature_labels: dict, lo: int, hi: int) -> list[str]:
    return [k for k, v in feature_labels.items() if lo <= v.get("layer", -1) <= hi]


# --------------------- ctrl-aware rules (Task 10, Apr 22) -----------------

def _per_cond_sets(class_sets: dict) -> dict[str, set[str]]:
    """Extract per-condition top-50 sets from class_sets; empty dict if absent."""
    raw = class_sets.get("per_condition_top50", {})
    return {cond: set(keys) for cond, keys in raw.items()}


def has_ctrl_data(class_sets: dict) -> bool:
    """True iff per_condition_top50 has entries for bare + all 5 jb_* + all 5 ctrl_*.

    Legacy L32 run JSONs pre-Task-7 won't have this block; skip ctrl-aware rules.
    """
    per = _per_cond_sets(class_sets)
    need = ["bare"] + [f"jb_{c}" for c in JB_CLASSES] + [f"ctrl_{c}" for c in JB_CLASSES]
    return all(k in per for k in need)


def build_ctrl_shared_refusal(class_sets: dict) -> list[str]:
    """Features in bare AND all 5 ctrl top-50 but NOT in all 5 jb top-50.

    'Format-robust refusal spine' — features the refusal circuit uses regardless
    of whether the prefix is a jailbreak or a matched benign control. These are
    NOT jailbreak-semantic; they're the prefix-invariant core that persists
    under ctrl-prefixes (same length/structure as JB, benign semantics).
    """
    per = _per_cond_sets(class_sets)
    if not has_ctrl_data(class_sets):
        return []
    all_ctrl = set.intersection(*(per[f"ctrl_{c}"] for c in JB_CLASSES))
    all_jb = set.intersection(*(per[f"jb_{c}"] for c in JB_CLASSES))
    return sorted((per["bare"] & all_ctrl) - all_jb)


def build_ctrl_only(class_sets: dict) -> list[str]:
    """Features in all 5 ctrl top-50 but NOT in bare and NOT in any jb top-50.

    Usually tiny. If non-empty, it's surprising: the benign ctrl-prefix recruits
    features that neither bare-harmful nor any jailbreak uses. These often map
    to benign-content semantics triggered by the prefix text itself.
    """
    per = _per_cond_sets(class_sets)
    if not has_ctrl_data(class_sets):
        return []
    all_ctrl = set.intersection(*(per[f"ctrl_{c}"] for c in JB_CLASSES))
    any_jb = set.union(*(per[f"jb_{c}"] for c in JB_CLASSES))
    return sorted(all_ctrl - per["bare"] - any_jb)


def build_jb_specific_vs_ctrl(class_sets: dict) -> dict[str, list[str]]:
    """Per-class: features in jb_{cls} top-50 but NOT in ctrl_{cls} top-50.

    The cleanest JB-semantic subcircuit per class. With prefix length/structure
    held constant by ctrl_{cls}, what remains is features the JB *semantic*
    content genuinely recruits. Complements canonical_pro_refusal by isolating
    per-class mechanism instead of cross-class intersection.
    """
    per = _per_cond_sets(class_sets)
    out: dict[str, list[str]] = {}
    for cls in JB_CLASSES:
        jb_k, ctrl_k = f"jb_{cls}", f"ctrl_{cls}"
        if jb_k not in per or ctrl_k not in per:
            out[cls] = []
            continue
        out[cls] = sorted(per[jb_k] - per[ctrl_k])
    return out


def compute_jb_vs_ctrl_contrast(class_sets: dict) -> dict[str, dict]:
    """Per-class recruitment contrast — the headline novel metric for Georg.

    For each jailbreak class, quantify how much of the feature recruitment is
    genuinely JB-semantic vs. a prefix-inflation artifact. Old L32 data couldn't
    compute this (no ctrl). Output per class:

        jb_top50:         |jb_{cls}_top50 corpus-union|
        ctrl_top50:       |ctrl_{cls}_top50|
        intersection:     |jb ∩ ctrl|          (prefix-induced)
        jb_specific:      |jb - ctrl|          (true JB-semantic machinery)
        ctrl_specific:    |ctrl - jb|          (benign-prefix-induced)
        jb_specific_frac: |jb - ctrl| / |jb|   (fraction genuinely JB-semantic)
        overlap_frac:     |jb ∩ ctrl| / |jb|   (fraction prefix-driven)

    Reading: jb_specific_frac close to 1.0 → JB recruits mechanisms the benign
    prefix does NOT (strong JB-semantic signal). Close to 0.0 → JB's effect
    is mostly an artifact of having a long prefix at all.
    """
    per = _per_cond_sets(class_sets)
    out: dict[str, dict] = {}
    for cls in JB_CLASSES:
        jb = per.get(f"jb_{cls}", set())
        ctrl = per.get(f"ctrl_{cls}", set())
        inter = jb & ctrl
        jb_only = jb - ctrl
        ctrl_only = ctrl - jb
        n_jb = len(jb)
        out[cls] = {
            "jb_top50": n_jb,
            "ctrl_top50": len(ctrl),
            "intersection": len(inter),
            "jb_specific": len(jb_only),
            "ctrl_specific": len(ctrl_only),
            "jb_specific_frac": round(len(jb_only) / n_jb, 3) if n_jb else 0.0,
            "overlap_frac": round(len(inter) / n_jb, 3) if n_jb else 0.0,
        }
    return out


# -------------------------- summarization -------------------------------

def summarize_subcircuit(name: str, feature_keys: list[str],
                         feature_labels: dict, n_layers: int = 34) -> dict:
    by_layer = [0] * n_layers
    freqs = []
    for k in feature_keys:
        v = feature_labels.get(k, {})
        layer = v.get("layer")
        if layer is not None and 0 <= layer < n_layers:
            by_layer[layer] += 1
        f = v.get("activation_frequency")
        if f is not None:
            freqs.append(f)

    top = sorted(
        feature_keys,
        key=lambda k: feature_labels.get(k, {}).get("max_abs_attribution", 0.0),
        reverse=True,
    )[:5]
    top_features = []
    for k in top:
        v = feature_labels.get(k, {})
        top_features.append({
            "key": k,
            "layer": v.get("layer"),
            "max_abs_attribution": round(v.get("max_abs_attribution", 0.0), 4),
            "top_logits": (v.get("top_logits") or [])[:3],
        })

    n_occupied = sum(1 for c in by_layer if c > 0)
    peak_layer = max(range(n_layers), key=lambda i: by_layer[i]) if any(by_layer) else None
    return {
        "name": name,
        "size": len(feature_keys),
        "features": feature_keys,
        "by_layer": by_layer,
        "peak_layer": peak_layer,
        "peak_count": by_layer[peak_layer] if peak_layer is not None else 0,
        "n_layers_occupied": n_occupied,
        "mean_activation_frequency": round(sum(freqs) / len(freqs), 6) if freqs else None,
        "top_features": top_features,
    }


def build_overlap_matrix(subcircuits: dict[str, list[str]]) -> dict:
    names = list(subcircuits.keys())
    sets = {n: set(subcircuits[n]) for n in names}
    n = len(names)
    inter = [[0] * n for _ in range(n)]
    norm = [[0.0] * n for _ in range(n)]
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            ab = sets[a] & sets[b]
            inter[i][j] = len(ab)
            m = min(len(sets[a]), len(sets[b]))
            norm[i][j] = len(ab) / m if m else 0.0
    return {"names": names, "intersection": inter, "normalized": norm}


# ------------------------------ plots -----------------------------------

def plot_treemap(summary: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = sorted(summary.items(), key=lambda kv: kv[1]["size"], reverse=True)
    items = [(k, v) for k, v in items if v["size"] > 0]
    try:
        import squarify
        sizes = [v["size"] for _, v in items]
        labels = [f"{k}\nn={v['size']}" for k, v in items]
        colors = plt.cm.tab20([i / max(len(items), 1) for i in range(len(items))])
        fig, ax = plt.subplots(figsize=(14, 8))
        squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.85, ax=ax,
                      text_kwargs={"fontsize": 9})
        ax.axis("off")
        ax.set_title(f"Rule-Based Subcircuits  (total features touched = {sum(sizes)})")
    except ImportError:
        names = [k for k, _ in items]
        sizes = [v["size"] for _, v in items]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(names, sizes, color="#4a90e2", edgecolor="black")
        for i, s in enumerate(sizes):
            ax.text(s, i, f" {s}", va="center")
        ax.set_xlabel("Number of features")
        ax.set_title("Subcircuit sizes (rule-based)  [squarify unavailable → bar chart]")
        ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_by_layer(summary: dict, out_path: Path, n_layers: int = 34) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    items = [(k, v) for k, v in summary.items() if v["size"] > 0]
    n = len(items)
    fig, axes = plt.subplots(n, 1, figsize=(14, max(9, 1.3 * n)), sharex=True)
    if n == 1:
        axes = [axes]
    x = np.arange(n_layers)
    for ax, (name, v) in zip(axes, items):
        ax.bar(x, v["by_layer"], color="#4a90e2", edgecolor="black", linewidth=0.3)
        ax.axvspan(24, 32, alpha=0.08, color="red", zorder=0)
        ax.set_ylabel("n", fontsize=9)
        peak = v["peak_layer"]
        peak_str = f"peak L{peak}={v['peak_count']}" if peak is not None else "empty"
        ax.set_title(f"{name}  (n={v['size']}, {peak_str})", fontsize=10, loc="left")
        ax.grid(axis="y", alpha=0.3)
    axes[-1].set_xlabel("Layer index")
    axes[-1].set_xticks(np.arange(0, n_layers, 2))
    fig.suptitle("Subcircuit Layer Distributions  (shaded: L24–L32 late-wave band)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_overlap(overlap: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = overlap["names"]
    mat = np.array(overlap["normalized"])
    fig, ax = plt.subplots(figsize=(max(9, 0.9 * len(names)), max(7, 0.7 * len(names))))
    im = ax.imshow(mat, cmap="magma_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            v = mat[i][j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.55 else "black", fontsize=8)
    ax.set_title("Subcircuit Pairwise Overlap  |A∩B| / min(|A|, |B|)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("fraction of smaller set in intersection")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_jb_vs_ctrl_contrast(contrast: dict, out_path: Path) -> None:
    """Per-class stacked bars: intersection (prefix-driven) vs jb_specific vs ctrl_specific.

    This is the mentor-facing headline figure — at a glance, which JB classes
    have the most genuinely-JB-semantic machinery (tall blue) vs which are
    mostly prefix-driven (tall yellow). Adds jb_specific_frac annotation on
    each bar.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not contrast:
        return
    classes = sorted(contrast.keys())
    inter = [contrast[c]["intersection"] for c in classes]
    jb_only = [contrast[c]["jb_specific"] for c in classes]
    ctrl_only = [contrast[c]["ctrl_specific"] for c in classes]
    jb_frac = [contrast[c]["jb_specific_frac"] for c in classes]

    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(11, 6))
    # Stack: ctrl-specific (grey, down), intersection (yellow, middle), jb-specific (blue, up)
    # Jb total = intersection + jb_specific (plotted above 0)
    # Ctrl total = -(intersection + ctrl_specific) to appear below 0
    ax.bar(x, jb_only, color="#1f77b4", label="JB-specific (genuinely JB-semantic)")
    ax.bar(x, inter, bottom=jb_only, color="#e0c060",
           label="Shared JB ∩ ctrl (prefix-induced)")
    ax.bar(x, [-v for v in inter], color="#e0c060")  # mirror below zero
    ax.bar(x, [-v for v in ctrl_only], bottom=[-v for v in inter],
           color="#888888", label="Ctrl-specific (benign-prefix-only)")

    # JB-specific fraction annotations above each bar
    for i, c in enumerate(classes):
        top = jb_only[i] + inter[i]
        ax.text(i, top + 1.5, f"JB-specific\n{jb_frac[i]*100:.0f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.set_ylabel("features in top-50 union")
    ax.set_title(
        "Per-class JB-vs-Ctrl recruitment contrast\n"
        "(top: features in jb_{cls}_top50 — blue = jb-only, yellow = shared; "
        "bottom: features in ctrl_{cls}_top50 — yellow = shared, grey = ctrl-only)"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_jb_specific_by_layer(jb_specific: dict, feature_labels: dict,
                              out_path: Path, n_layers: int = 34) -> None:
    """Stacked bar: per-layer count of jb_{cls}_specific_vs_ctrl features, per class.

    Does JB-semantic machinery concentrate in specific layers? Compare against
    the L24–L32 late-wave band. If peaks are the same, the late-wave finding
    held up after controlling for prefix; if they shift, the old band was
    partially a prefix-length artifact.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    classes = sorted(jb_specific.keys())
    colors = {
        "analytical": "#1f77b4",
        "cognitive_reframe": "#ff7f0e",
        "completion": "#2ca02c",
        "fiction": "#d62728",
        "roleplay": "#9467bd",
    }
    x = np.arange(n_layers)
    by_layer = {cls: [0] * n_layers for cls in classes}
    for cls in classes:
        for k in jb_specific[cls]:
            layer = feature_labels.get(k, {}).get("layer")
            if layer is not None and 0 <= layer < n_layers:
                by_layer[cls][layer] += 1

    fig, ax = plt.subplots(figsize=(14, 6))
    bottom = np.zeros(n_layers, dtype=int)
    for cls in classes:
        ax.bar(x, by_layer[cls], bottom=bottom, color=colors.get(cls, "#666"),
               label=cls, edgecolor="black", linewidth=0.2)
        bottom = bottom + np.array(by_layer[cls])
    ax.axvspan(24, 32, alpha=0.08, color="red", zorder=0)
    ax.set_xticks(np.arange(0, n_layers, 2))
    ax.set_xlabel("Layer index")
    ax.set_ylabel("features (stacked across classes)")
    total = sum(len(jb_specific[c]) for c in classes)
    ax.set_title(
        f"JB-specific-vs-ctrl features by layer  (n={total} across 5 JB classes; "
        f"shaded: L24–L32 late-wave band)"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ----------------------------- report -----------------------------------

DEFINITIONS = {
    "universal_refusal_core": (
        "Features seen in **bare + all 5 JB classes**. The canonical refusal core — "
        "present in both harmful-alone and every jailbreak. Ablation control baseline."
    ),
    "canonical_pro_refusal": (
        "Features seen in **all 5 JB classes but NOT bare**. Recruited specifically under "
        "jailbreak — interpretable as a shared JB-suppression / pro-refusal response."
    ),
    "sign_flip_convergent": (
        "Features in the **sign_flipped** bucket of ≥{min} JB classes. Robustly reverse "
        "attribution sign under JB — the highest-confidence mechanism-change features."
    ),
    "dampening_specialists": (
        "Features in the **dampened** bucket of ≥{min} JB classes. Pro-refusal features "
        "whose contribution to the refusal direction weakens across most JB types."
    ),
    "anti_refusal_amplifiers": (
        "Features in the **amplified_anti** bucket of ≥{min} JB classes. Anti-refusal "
        "features that grow in magnitude across most JB types — the bypass signal."
    ),
    "late_wave_layer24_32": (
        "All tagged features in layers **{lo}–{hi}** — the JB-impact band identified in A8. "
        "Layer-based cross-cut; overlaps other subcircuits."
    ),
    "ctrl_shared_refusal": (
        "Features in **bare ∩ all 5 ctrl_*_top50** but **NOT in all 5 jb_*_top50**. "
        "The prefix-invariant refusal spine: machinery the refusal circuit uses "
        "regardless of whether the prefix carries JB-semantics or matched benign content. "
        "These are NOT JB-semantic — they define the baseline that survives a long-prefix "
        "perturbation without JB intent."
    ),
    "ctrl_only": (
        "Features in **all 5 ctrl_*_top50** but not in bare or any jb_*_top50. "
        "Usually tiny; if non-empty, it signals that matched benign prefixes recruit "
        "features neither bare-harmful nor any jailbreak uses — typically benign-content "
        "semantic features triggered by the ctrl prefix text itself."
    ),
}
CLASS_EXCLUSIVE_DEF = (
    "Features seen in **only** the `{cls}` JB class (no bare, no other JB). "
    "Candidates for `{cls}`-specific jailbreak mechanism."
)
JB_SPECIFIC_DEF = (
    "Features in **jb_{cls}_top50 − ctrl_{cls}_top50**. The cleanest JB-semantic "
    "subcircuit for `{cls}`: after controlling for prefix length/structure via the "
    "matched benign ctrl prefix, what remains is features the JB *semantic* content "
    "genuinely recruits. Complements canonical_pro_refusal (which finds cross-class "
    "intersection) by isolating per-class mechanism."
)


def write_report(summary: dict, overlap: dict, out_path: Path,
                 convergent_min: int, late_lo: int, late_hi: int,
                 contrast: dict | None = None) -> None:
    ordered = sorted(summary.keys(), key=lambda k: -summary[k]["size"])
    lines = [
        "# Subcircuits Report (Rule-Based)",
        "",
        "Each subcircuit is defined by a precise set-logic rule over the features observed "
        "across bare + 5 JB classes (original rules) and bare + 5 jb_* + 5 ctrl_* conditions "
        "(new Apr 22 ctrl-aware rules). No ML fitting — fully interpretable.",
        "",
        "## Summary table",
        "",
        "| Subcircuit | Size | Peak layer | n_layers occupied | Mean act. freq. |",
        "|---|---|---|---|---|",
    ]
    for name in ordered:
        v = summary[name]
        freq = v.get("mean_activation_frequency")
        freq_str = f"{freq:.4f}" if freq is not None else "N/A"
        peak = v["peak_layer"]
        peak_str = f"L{peak} (×{v['peak_count']})" if peak is not None else "—"
        lines.append(
            f"| `{name}` | {v['size']} | {peak_str} | {v['n_layers_occupied']} | {freq_str} |"
        )

    # JB-vs-ctrl contrast table (new headline section)
    if contrast:
        lines.extend([
            "",
            "## JB-vs-Ctrl recruitment contrast (NEW — Task 10)",
            "",
            "For each JB class, how much of the corpus-level top-50 recruitment is "
            "**genuinely JB-semantic** vs. **prefix-induced** (also triggered by the "
            "matched benign ctrl prefix)? Old L32 data could not compute this; it's the "
            "headline new finding enabled by the 11-condition ctrl-balanced dataset.",
            "",
            "| Class | \\|jb_top50\\| | \\|ctrl_top50\\| | Intersection | JB-specific | Ctrl-specific | **JB-specific %** | Overlap % |",
            "|---|---|---|---|---|---|---|---|",
        ])
        for cls in sorted(contrast.keys()):
            c = contrast[cls]
            lines.append(
                f"| `{cls}` | {c['jb_top50']} | {c['ctrl_top50']} | {c['intersection']} | "
                f"{c['jb_specific']} | {c['ctrl_specific']} | "
                f"**{c['jb_specific_frac']*100:.0f}%** | {c['overlap_frac']*100:.0f}% |"
            )
        lines.extend([
            "",
            "**Reading**: `JB-specific %` close to 100 → JB recruits mechanisms the "
            "benign prefix does NOT (strong JB-semantic signal). Close to 0 → JB's "
            "effect is mostly a prefix-length artifact, not genuine semantic mechanism.",
            "",
        ])

    lines.extend(["", "## Subcircuit definitions and top features", ""])
    for name in ordered:
        v = summary[name]
        if name in DEFINITIONS:
            defn = DEFINITIONS[name].format(min=convergent_min, lo=late_lo, hi=late_hi)
        elif name.endswith("_specific_vs_ctrl"):
            cls = name.removesuffix("_specific_vs_ctrl").removeprefix("jb_")
            defn = JB_SPECIFIC_DEF.format(cls=cls)
        elif name.endswith("_exclusive"):
            defn = CLASS_EXCLUSIVE_DEF.format(cls=name.removesuffix("_exclusive"))
        else:
            defn = "(no definition)"
        lines.extend([f"### `{name}` — n={v['size']}", "", defn, ""])
        if v["top_features"]:
            lines.append("**Top 5 by |attribution|:**")
            lines.append("")
            lines.append("| Feature | Layer | \\|attr\\| | Top logits |")
            lines.append("|---|---|---|---|")
            for f in v["top_features"]:
                logits = ", ".join(repr(x) for x in f["top_logits"])
                if len(logits) > 60:
                    logits = logits[:57] + "..."
                lines.append(
                    f"| `{f['key']}` | L{f['layer']} | "
                    f"{f['max_abs_attribution']:.3f} | {logits} |"
                )
            lines.append("")

    # Top pairwise overlaps
    names = overlap["names"]
    norm = overlap["normalized"]
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((norm[i][j], names[i], names[j]))
    pairs.sort(reverse=True)
    lines.extend([
        "## Pairwise overlap (top 10 by normalized intersection)",
        "",
        "Normalized overlap = |A ∩ B| / min(|A|, |B|). High values mean the smaller set is "
        "largely contained in the larger. `late_wave_layer24_32` naturally absorbs many.",
        "",
        "| A | B | norm. overlap |",
        "|---|---|---|",
    ])
    for v, a, b in pairs[:10]:
        lines.append(f"| `{a}` | `{b}` | {v:.2f} |")
    lines.extend([
        "",
        "## Suggested Stage 08 ablation targets (causal-impact order)",
        "",
        "1. `canonical_pro_refusal` — JB-specific pro-refusal recruitment. Ablation "
        "should *strengthen* JB bypass (removes the JB-only refusal boost).",
        "2. `jb_{cls}_specific_vs_ctrl` (per class) — the cleanest per-class JB-semantic "
        "mechanism. Ablating one should selectively restore ctrl-like behavior on that "
        "class (dissociation test).",
        "3. `sign_flip_convergent` — robust direction reversals. Ablation should partially "
        "restore bare behavior under JB.",
        "4. `dampening_specialists` — weakened pro-refusal features. Restoring them to "
        "bare strength should counter fiction/analytical bypass.",
        "5. `anti_refusal_amplifiers` — JB-amplified bypass signal. Suppressing them "
        "should increase refusal under JB.",
        "6. `ctrl_shared_refusal` — the prefix-invariant spine. Ablation should break "
        "refusal on BOTH ctrl and bare — a negative control proving these aren't "
        "JB-specific.",
        "7. `universal_refusal_core` — shared baseline. Ablation should break refusal "
        "on bare *and* JB (control — proves the subcircuits matter).",
        "",
    ])
    out_path.write_text("\n".join(lines) + "\n")


# ------------------------------ main ------------------------------------

def main():
    args = parse_args()
    run_dir = args.run_dir
    out_dir = get_stage_dir(run_dir, "07_subcircuits")

    print("=" * 60)
    print("STAGE 07: Rule-Based Subcircuit Identification")
    print("=" * 60)

    labels_path = run_dir / "04_labels" / "feature_labels.json"
    sets_path = run_dir / "04_labels" / "feature_class_sets.json"
    for p in (labels_path, sets_path):
        if not p.exists():
            print(f"  ERROR: missing {p}")
            sys.exit(1)

    feature_labels = load_json(labels_path)
    class_sets = load_json(sets_path)
    print(f"  Loaded {len(feature_labels)} features from feature_labels.json")
    ctrl_available = has_ctrl_data(class_sets)
    print(
        f"  Ctrl-aware data: {'AVAILABLE' if ctrl_available else 'missing (legacy schema — ctrl rules will emit empty)'}"
    )

    # Build subcircuits
    print("\n  Building subcircuits...")
    subcircuits: dict[str, list[str]] = {}
    subcircuits["universal_refusal_core"] = build_universal_core(feature_labels)
    subcircuits["canonical_pro_refusal"] = build_canonical_pro_refusal(feature_labels)
    subcircuits["sign_flip_convergent"] = build_convergent_bucket(
        class_sets, "sign_flipped", args.convergent_min,
    )
    subcircuits["dampening_specialists"] = build_convergent_bucket(
        class_sets, "dampened", args.convergent_min,
    )
    subcircuits["anti_refusal_amplifiers"] = build_convergent_bucket(
        class_sets, "amplified_anti", args.convergent_min,
    )
    subcircuits["late_wave_layer24_32"] = build_late_wave(
        feature_labels, args.late_wave_start, args.late_wave_end,
    )
    class_exclusive = build_class_exclusive(feature_labels)
    for cls in sorted(JB_CLASSES):
        subcircuits[f"{cls}_exclusive"] = class_exclusive[cls]

    # Ctrl-aware subcircuits (Task 10 — require per_condition_top50 from Task 7)
    subcircuits["ctrl_shared_refusal"] = build_ctrl_shared_refusal(class_sets)
    subcircuits["ctrl_only"] = build_ctrl_only(class_sets)
    jb_specific = build_jb_specific_vs_ctrl(class_sets)
    for cls in sorted(JB_CLASSES):
        subcircuits[f"jb_{cls}_specific_vs_ctrl"] = jb_specific[cls]

    for name, feats in subcircuits.items():
        print(f"    {name}: {len(feats)}")

    # Headline novel metric — JB vs ctrl contrast per class
    contrast = compute_jb_vs_ctrl_contrast(class_sets) if ctrl_available else {}
    if contrast:
        print("\n  JB-vs-ctrl contrast (fraction of JB machinery that's JB-semantic):")
        for cls in sorted(contrast.keys()):
            c = contrast[cls]
            print(
                f"    {cls:>22}: jb={c['jb_top50']:3d}  ctrl={c['ctrl_top50']:3d}  "
                f"jb_specific={c['jb_specific']:3d} ({c['jb_specific_frac']*100:4.1f}%)  "
                f"overlap={c['intersection']:3d} ({c['overlap_frac']*100:4.1f}%)"
            )

    # Invariants
    print("\n  Checking invariants...")
    excl_sets = {c: set(class_exclusive[c]) for c in JB_CLASSES}
    cls_list = sorted(JB_CLASSES)
    for i in range(len(cls_list)):
        for j in range(i + 1, len(cls_list)):
            a, b = cls_list[i], cls_list[j]
            assert excl_sets[a].isdisjoint(excl_sets[b]), (
                f"{a}_exclusive ∩ {b}_exclusive ≠ ∅"
            )
    uni_set = set(subcircuits["universal_refusal_core"])
    can_set = set(subcircuits["canonical_pro_refusal"])
    assert uni_set.isdisjoint(can_set), "universal_refusal_core ∩ canonical_pro_refusal ≠ ∅"
    for c in JB_CLASSES:
        assert not (excl_sets[c] & can_set), f"{c}_exclusive ∩ canonical_pro_refusal ≠ ∅"
    late_set = set(subcircuits["late_wave_layer24_32"])
    for k in late_set:
        layer = feature_labels.get(k, {}).get("layer", -1)
        assert args.late_wave_start <= layer <= args.late_wave_end, (
            f"{k} in late_wave but layer={layer}"
        )
    # Ctrl-aware invariants
    if ctrl_available:
        per = _per_cond_sets(class_sets)
        # jb_{cls}_specific_vs_ctrl must be disjoint from ctrl_{cls} top-50
        for cls in JB_CLASSES:
            jb_spec = set(subcircuits[f"jb_{cls}_specific_vs_ctrl"])
            ctrl_top = per.get(f"ctrl_{cls}", set())
            assert jb_spec.isdisjoint(ctrl_top), (
                f"jb_{cls}_specific_vs_ctrl intersects ctrl_{cls}_top50"
            )
        # ctrl_only must be disjoint from bare and any jb_*
        ctrl_only_set = set(subcircuits["ctrl_only"])
        assert ctrl_only_set.isdisjoint(per["bare"]), "ctrl_only intersects bare top-50"
        any_jb = set.union(*(per[f"jb_{c}"] for c in JB_CLASSES))
        assert ctrl_only_set.isdisjoint(any_jb), "ctrl_only intersects some jb_*"
        # ctrl_shared_refusal ⊆ bare
        ctrl_shared_set = set(subcircuits["ctrl_shared_refusal"])
        assert ctrl_shared_set <= per["bare"], "ctrl_shared_refusal not ⊆ bare top-50"
    print("    All invariants pass ✓")

    # Summarize
    print("\n  Summarizing subcircuits...")
    summary = {
        name: summarize_subcircuit(name, feats, feature_labels)
        for name, feats in subcircuits.items()
    }
    overlap = build_overlap_matrix(subcircuits)

    # Save JSON outputs
    out = {
        "metadata": {
            "rule": "set logic over feature_labels.conditions_seen + feature_class_sets "
                    "buckets + per_condition_top50 (ctrl-aware rules)",
            "convergent_min_classes": args.convergent_min,
            "late_wave_range": [args.late_wave_start, args.late_wave_end],
            "jb_classes": sorted(JB_CLASSES),
            "n_features_input": len(feature_labels),
            "ctrl_available": ctrl_available,
        },
        "subcircuits": summary,
        "jb_vs_ctrl_contrast": contrast,
    }
    save_json(out, out_dir / "subcircuits.json")
    save_json(
        {
            "sizes": {n: v["size"] for n, v in summary.items()},
            "overlap_matrix": overlap,
            "jb_vs_ctrl_contrast": contrast,
        },
        out_dir / "subcircuits_summary.json",
    )
    print("    Saved subcircuits.json + subcircuits_summary.json")

    # Plots
    print("\n  Generating plots...")
    plot_treemap(summary, out_dir / "subcircuits_treemap.png")
    print("    subcircuits_treemap.png")
    plot_by_layer(summary, out_dir / "subcircuits_by_layer.png")
    print("    subcircuits_by_layer.png")
    plot_overlap(overlap, out_dir / "subcircuits_overlap.png")
    print("    subcircuits_overlap.png")
    if contrast:
        plot_jb_vs_ctrl_contrast(contrast, out_dir / "jb_vs_ctrl_contrast.png")
        print("    jb_vs_ctrl_contrast.png")
        plot_jb_specific_by_layer(jb_specific, feature_labels,
                                  out_dir / "jb_specific_by_layer.png")
        print("    jb_specific_by_layer.png")

    # Report
    write_report(
        summary, overlap, out_dir / "SUBCIRCUITS_REPORT.md",
        args.convergent_min, args.late_wave_start, args.late_wave_end,
        contrast=contrast,
    )
    print("    SUBCIRCUITS_REPORT.md")

    total_unique = len({k for feats in subcircuits.values() for k in feats})
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Subcircuits identified:  {len(subcircuits)}")
    print(f"  Unique features tagged:  {total_unique}")
    print(f"  Universal core:          {summary['universal_refusal_core']['size']}")
    print(f"  Canonical pro-refusal:   {summary['canonical_pro_refusal']['size']}")
    print(f"  Sign-flip convergent:    {summary['sign_flip_convergent']['size']}")
    print(f"  Dampening specialists:   {summary['dampening_specialists']['size']}")
    print(f"  Anti-refusal amplifiers: {summary['anti_refusal_amplifiers']['size']}")
    print(f"  Late-wave (L24–L32):     {summary['late_wave_layer24_32']['size']}")
    print("  Class-exclusive:")
    for c in sorted(JB_CLASSES):
        print(f"    {c}: {summary[f'{c}_exclusive']['size']}")
    if ctrl_available:
        print("  Ctrl-aware:")
        print(f"    ctrl_shared_refusal:    {summary['ctrl_shared_refusal']['size']}")
        print(f"    ctrl_only:              {summary['ctrl_only']['size']}")
        for c in sorted(JB_CLASSES):
            print(f"    jb_{c}_specific_vs_ctrl: {summary[f'jb_{c}_specific_vs_ctrl']['size']}")
    print(f"  Outputs: {out_dir}/")
    print("DONE!")


if __name__ == "__main__":
    main()
