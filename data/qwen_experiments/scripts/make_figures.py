#!/usr/bin/env python
"""Generate publication figures for Qwen3-4B refusal experiments.

Reads results from `data/qwen_experiments/results_v2/` and writes PNGs to
`data/qwen_experiments/figures/`. Mirrors the figure set in
`data/tejas_experiments/figures/` so the two folders can be compared at a
glance.

Run:
    python data/qwen_experiments/scripts/make_figures.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_v2"
FIGS = ROOT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 220,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def _save(fig: plt.Figure, name: str) -> None:
    out = FIGS / name
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT.parent)}")


# --------------------------------------------------------------------------
# 1. Layer separation (best position) + 2D position x layer heatmap
# --------------------------------------------------------------------------

def fig_layer_separation() -> None:
    table = json.loads((RESULTS / "separation_table.json").read_text())
    positions = [-5, -4, -3, -2, -1]
    layers = sorted({int(k.split("_layer")[1]) for k in table})
    grid = np.zeros((len(positions), len(layers)))
    for i, p in enumerate(positions):
        for j, layer in enumerate(layers):
            grid[i, j] = table[f"pos{p}_layer{layer}"]

    sanity = json.loads((RESULTS / "sanity_check_v2.json").read_text())
    best_pos = sanity["best_pos"]
    best_layer = sanity["best_layer"]
    best_row = positions.index(best_pos)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5),
                                   gridspec_kw={"width_ratios": [1.5, 1]})

    seps = grid[best_row]
    colors = plt.cm.viridis(np.linspace(0.25, 0.95, len(layers)))
    bars = ax1.bar(layers, seps, color=colors, edgecolor="black", linewidth=0.4)
    bars[best_layer].set_edgecolor("red")
    bars[best_layer].set_linewidth(2.2)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Separation (|mean_harmful - mean_harmless|)")
    ax1.set_title(f"Qwen3-4B refusal-direction separation at position {best_pos}")
    ax1.axvline(best_layer, color="red", linestyle="--", alpha=0.4,
                label=f"best layer = {best_layer}")
    ax1.axvline(18, color="orange", linestyle=":", alpha=0.7,
                label="causal layer = 18")
    ax1.legend(loc="upper left")

    im = ax2.imshow(grid, aspect="auto", cmap="viridis",
                    extent=[layers[0] - 0.5, layers[-1] + 0.5,
                            positions[0] - 0.5, positions[-1] + 0.5],
                    origin="lower")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Position")
    ax2.set_yticks(positions)
    ax2.set_title("Separation heatmap (positions x layers)")
    ax2.scatter([best_layer], [best_pos], marker="*", color="red", s=180,
                edgecolor="white", linewidths=1.5, label="argmax")
    ax2.legend(loc="lower right")
    fig.colorbar(im, ax=ax2, fraction=0.045)

    _save(fig, "layer_separation.png")


# --------------------------------------------------------------------------
# 2. Sanity-check projection distributions
# --------------------------------------------------------------------------

def fig_sanity_check_projections() -> None:
    s = json.loads((RESULTS / "sanity_check_v2.json").read_text())
    cats = ["harmful", "harmless", "jailbroken"]
    means = [s[c]["mean"] for c in cats]
    stds = [s[c]["std"] for c in cats]
    palette = ["#c0392b", "#2c7a3a", "#f39c12"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(cats))
    bars = ax.bar(x, means, yerr=stds, color=palette,
                  edgecolor="black", linewidth=0.5,
                  capsize=8, error_kw={"elinewidth": 1.4, "ecolor": "#333"})
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in cats])
    ax.set_ylabel(f"Projection at L{s['best_layer']}, pos {s['best_pos']}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(
        f"Qwen3-4B sanity check: r-projection at best (layer, position) = "
        f"({s['best_layer']}, {s['best_pos']})"
    )
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m,
                f"{m:,.0f}", ha="center",
                va="bottom" if m >= 0 else "top",
                fontsize=10, fontweight="bold")
    ax.text(0.99, 0.02,
            "Jailbroken prompts sit between harmful and harmless\n"
            "(projection drops ~23% vs bare harmful).",
            ha="right", va="bottom", transform=ax.transAxes,
            fontsize=9, style="italic",
            bbox=dict(facecolor="#f8f8f8", edgecolor="#bbb", alpha=0.85))
    _save(fig, "sanity_check_projections.png")


# --------------------------------------------------------------------------
# 3. Causal Arditi: per-layer + per-class flip rate
# --------------------------------------------------------------------------

def fig_causal_arditi() -> None:
    full = json.loads((RESULTS / "causal_arditi" / "full_results.json").read_text())
    ir = full["intervention_results"]
    cr = full["control_results"]

    layers = sorted(full["test_layers"])
    classes = sorted({x["class"] for x in ir})

    # Per-layer aggregates
    per_layer = defaultdict(lambda: {"comply": 0, "flipped": 0, "coherent": 0})
    for x in ir:
        if x["baseline_cls"] == "COMPLY":
            per_layer[x["layer"]]["comply"] += 1
            if x["flipped"]:
                per_layer[x["layer"]]["flipped"] += 1
                if x["coherent"]:
                    per_layer[x["layer"]]["coherent"] += 1
    benign_refuse = defaultdict(lambda: {"tot": 0, "refuse": 0})
    for x in cr:
        benign_refuse[x["layer"]]["tot"] += 1
        if x.get("intervened_cls") == "REFUSE":
            benign_refuse[x["layer"]]["refuse"] += 1

    # Per-(layer, class) aggregates (limit plot to causal layer 18)
    per_lc = defaultdict(lambda: {"comply": 0, "flipped": 0})
    for x in ir:
        if x["baseline_cls"] == "COMPLY":
            per_lc[(x["layer"], x["class"])]["comply"] += 1
            if x["flipped"]:
                per_lc[(x["layer"], x["class"])]["flipped"] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    # Panel A: per-layer flip rate, with benign-side break rate as reference
    width = 0.32
    x = np.arange(len(layers))
    flip_rates = [per_layer[L]["flipped"] / max(per_layer[L]["comply"], 1) for L in layers]
    coh_rates = [per_layer[L]["coherent"] / max(per_layer[L]["flipped"], 1) for L in layers]
    benign_break = [benign_refuse[L]["refuse"] / max(benign_refuse[L]["tot"], 1) for L in layers]

    b1 = ax1.bar(x - width, flip_rates, width, label="JB flip rate",
                 color="#2980b9", edgecolor="black", linewidth=0.4)
    b2 = ax1.bar(x, coh_rates, width, label="coherent | flipped",
                 color="#27ae60", edgecolor="black", linewidth=0.4)
    b3 = ax1.bar(x + width, benign_break, width, label="benign harm rate (bad!)",
                 color="#c0392b", edgecolor="black", linewidth=0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"L{L}" for L in layers])
    ax1.set_ylabel("Rate")
    ax1.set_ylim(0, 1.08)
    ax1.set_title("Arditi intervention: per-layer effect (Qwen3-4B)")
    ax1.legend(loc="upper left", fontsize=9)
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            if h > 0.02:
                ax1.text(bar.get_x() + bar.get_width() / 2, h,
                         f"{h:.0%}", ha="center", va="bottom",
                         fontsize=8, fontweight="bold")
    ax1.text(0.5, -0.18,
             "Causal layer = L18: 88% flip with 100% coherent and 0% benign harm.",
             ha="center", va="top", transform=ax1.transAxes,
             fontsize=9, style="italic")

    # Panel B: per-class flip rate at L18
    L = 18
    flip_rates_c = []
    counts = []
    for c in classes:
        a = per_lc[(L, c)]
        rate = a["flipped"] / max(a["comply"], 1)
        flip_rates_c.append(rate)
        counts.append((a["flipped"], a["comply"]))
    colors_c = plt.cm.Pastel1(np.linspace(0, 0.9, len(classes)))
    bars = ax2.barh(np.arange(len(classes)), flip_rates_c,
                    color=colors_c, edgecolor="black", linewidth=0.4)
    ax2.set_yticks(np.arange(len(classes)))
    ax2.set_yticklabels(classes)
    ax2.set_xlim(0, 1.12)
    ax2.set_xlabel("flip rate (flipped / complied)")
    ax2.set_title(f"Per-class flip rate at L{L} (causal layer)")
    for i, (bar, (f, c)) in enumerate(zip(bars, counts)):
        w = bar.get_width()
        ax2.text(w + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{f}/{c}  ({w:.0%})",
                 va="center", fontsize=9, fontweight="bold")
    ax2.invert_yaxis()
    _save(fig, "causal_arditi_jailbreak.png")


# --------------------------------------------------------------------------
# 4. Mechanism comparison (bare vs RP vs Fiction)
# --------------------------------------------------------------------------

def fig_mechanism_comparison() -> None:
    d = json.loads((RESULTS / "v2_mechanism_comparison.json").read_text())
    topics = ["lock", "hack", "phish"]
    conds = [("bare", "Bare"), ("rp", "Roleplay"), ("fiction", "Fiction")]
    palette = {"bare": "#2c7a3a", "rp": "#c0392b", "fiction": "#8e44ad"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    # Panel A: net projection (bar groups) by topic and condition
    width = 0.27
    x = np.arange(len(topics))
    for i, (key, label) in enumerate(conds):
        vals = [d[f"{key}_{t}"]["net"] for t in topics]
        bars = ax1.bar(x + (i - 1) * width, vals, width,
                       label=label, color=palette[key],
                       edgecolor="black", linewidth=0.4)
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     v + (500 if v >= 0 else -500),
                     f"{v:,.0f}", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=8.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([t.capitalize() for t in topics])
    ax1.set_ylabel("Net refusal-direction projection (sum over features)")
    ax1.set_title("Qwen3-4B: bare vs jailbreak projection at L34")
    ax1.axhline(0, color="black", lw=0.8)
    ax1.legend(loc="upper right")

    # Panel B: pos vs neg components stacked, bare vs RP vs Fiction (averaged across topics)
    avg = {k: {"pos": np.mean([d[f"{k}_{t}"]["pos"] for t in topics]),
               "neg": np.mean([d[f"{k}_{t}"]["neg"] for t in topics])}
           for k, _ in conds}
    labels = [label for _, label in conds]
    pos_vals = [avg[k]["pos"] for k, _ in conds]
    neg_vals = [avg[k]["neg"] for k, _ in conds]
    xb = np.arange(len(labels))
    ax2.bar(xb, pos_vals, color="#2980b9", edgecolor="black", linewidth=0.4,
            label="pro-refusal (+)")
    ax2.bar(xb, neg_vals, color="#c0392b", edgecolor="black", linewidth=0.4,
            label="anti-refusal (-)")
    for i, (p, n) in enumerate(zip(pos_vals, neg_vals)):
        ax2.text(i, p, f"{p:,.0f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
        ax2.text(i, n, f"{n:,.0f}", ha="center", va="top",
                 fontsize=9, fontweight="bold")
    ax2.set_xticks(xb)
    ax2.set_xticklabels(labels)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("Mean component (averaged across topics)")
    ax2.set_title("Pro- vs anti-refusal components")
    ax2.legend(loc="lower left")

    _save(fig, "mechanism_comparison.png")


# --------------------------------------------------------------------------
# 5. Georg scaling sweep (mode collapse)
# --------------------------------------------------------------------------

def fig_georg_scaling() -> None:
    p = RESULTS / "causal_georg_arditi" / "scaling_scan.json"
    d = json.loads(p.read_text())
    agg = defaultdict(lambda: {"benign_tot": 0, "benign_coh": 0, "benign_refuse": 0,
                               "jb_tot": 0, "jb_coh": 0, "jb_flip": 0})
    for r in d["results"]:
        s = r["scale"]
        if r["kind"] == "benign":
            agg[s]["benign_tot"] += 1
            if r["coherent"]:
                agg[s]["benign_coh"] += 1
            if r["cls"] == "REFUSE":
                agg[s]["benign_refuse"] += 1
        else:
            agg[s]["jb_tot"] += 1
            if r["coherent"]:
                agg[s]["jb_coh"] += 1
            if r["cls"] == "REFUSE":
                agg[s]["jb_flip"] += 1
    scales = sorted(agg.keys())

    coh_benign = [agg[s]["benign_coh"] / max(agg[s]["benign_tot"], 1) for s in scales]
    coh_jb = [agg[s]["jb_coh"] / max(agg[s]["jb_tot"], 1) for s in scales]
    flip_jb = [agg[s]["jb_flip"] / max(agg[s]["jb_tot"], 1) for s in scales]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(scales, coh_benign, marker="o", lw=2, label="benign coherent",
            color="#2c7a3a")
    ax.plot(scales, coh_jb, marker="s", lw=2, label="JB coherent",
            color="#2980b9")
    ax.plot(scales, flip_jb, marker="^", lw=2, label="JB flipped to REFUSE",
            color="#c0392b")
    ax.set_xlabel(f"Scale on target projection (L{d['layer']}, target={d['target_proj']:.1f})")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.04, 1.04)
    ax.set_title("Georg exact-magnitude method on Qwen3-4B: total mode collapse")
    ax.legend(loc="center right")
    ax.text(0.5, 0.55,
            "All scales produce token-loop output\n"
            "(`ifiedified...`, `_____...`, `I I I...`).\n"
            "Method does NOT port from Gemma to Qwen.",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=10, style="italic",
            bbox=dict(facecolor="#fff7d6", edgecolor="#999", alpha=0.95))
    _save(fig, "georg_scaling_collapse.png")


# --------------------------------------------------------------------------
# 6. Per-position cosine heatmap at causal + best layer
# --------------------------------------------------------------------------

def fig_position_cosine() -> None:
    state = torch.load(RESULTS / "refusal_direction_v2.pt",
                       map_location="cpu", weights_only=False)
    positions = [-5, -4, -3, -2, -1]

    def cos_grid(layer: int) -> np.ndarray:
        vecs = [state[f"direction_pos{p}_layer{layer}"].float() for p in positions]
        normed = [v / (v.norm() + 1e-12) for v in vecs]
        n = len(positions)
        out = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                out[i, j] = float(normed[i] @ normed[j])
        return out

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, layer, title in [
        (axes[0], 18, "Qwen3-4B causal layer L18"),
        (axes[1], 34, "Qwen3-4B best layer L34"),
    ]:
        grid = cos_grid(layer)
        im = ax.imshow(grid, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(positions)))
        ax.set_yticks(range(len(positions)))
        ax.set_xticklabels(positions)
        ax.set_yticklabels(positions)
        ax.set_xlabel("Position")
        ax.set_ylabel("Position")
        ax.set_title(title)
        for i in range(len(positions)):
            for j in range(len(positions)):
                ax.text(j, i, f"{grid[i, j]:+.2f}",
                        ha="center", va="center",
                        color="black" if abs(grid[i, j]) < 0.6 else "white",
                        fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.045, label="cos")
    fig.suptitle(
        "Per-position cosine of refusal direction "
        "(Gemma shows ~-0.78 here; Qwen does not)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    _save(fig, "position_cosine_heatmap.png")


# --------------------------------------------------------------------------
# 7. Gemma vs Qwen summary panel
# --------------------------------------------------------------------------

def fig_summary() -> None:
    fig = plt.figure(figsize=(13, 7))
    fig.patch.set_facecolor("white")

    rows = [
        ("Refusal direction = single linear feature", "YES", "YES", "PASS"),
        ("Best layer at ~94% of network depth",       "L32/34",  "L34/36", "PASS"),
        ("Mid-network causal layer",                  "L15 (44%)", "L18 (50%)", "PASS"),
        ("Arditi causal intervention works",          "100% / 100% coh", "88% / 100% coh", "PASS"),
        ("Per-position directions anti-correlated",   "cos(-2,-3) ~-0.78", "all cos > 0",   "DIVERGES"),
        ("Georg exact-magnitude intervention",        "scale 1.0 OK",      "mode collapse", "DIVERGES"),
        ("Two-MLP-mechanism taxonomy (RP vs Fic)",    "yes",  "blocked (no attn SAEs)", "BLOCKED"),
        ("MLP fraction of refusal signal",            "0.4%", "blocked",                  "BLOCKED"),
    ]

    ax = fig.add_subplot(111)
    ax.axis("off")
    n = len(rows)
    col_x = [0.02, 0.50, 0.66, 0.84]
    headers = ["Finding", "Gemma-3-4B-IT", "Qwen3-4B", "Status"]
    for x, h in zip(col_x, headers):
        ax.text(x, 0.97, h, fontsize=12, fontweight="bold",
                transform=ax.transAxes)
    ax.plot([0.01, 0.99], [0.94, 0.94], color="black", transform=ax.transAxes,
            clip_on=False, lw=1.0)

    status_color = {"PASS": "#2c7a3a", "DIVERGES": "#c0392b", "BLOCKED": "#7f8c8d"}
    for i, (find, g, q, status) in enumerate(rows):
        y = 0.88 - i * (0.85 / n)
        ax.text(col_x[0], y, find, transform=ax.transAxes, fontsize=10.5)
        ax.text(col_x[1], y, g, transform=ax.transAxes, fontsize=10.5)
        ax.text(col_x[2], y, q, transform=ax.transAxes, fontsize=10.5)
        ax.text(col_x[3], y, status, transform=ax.transAxes,
                fontsize=10.5, fontweight="bold",
                color=status_color[status])

    ax.text(0.5, 0.02,
            "PASS = finding holds on Qwen   |   "
            "DIVERGES = Gemma-specific   |   "
            "BLOCKED = needs attention SAEs",
            ha="center", va="bottom", fontsize=9.5, style="italic",
            transform=ax.transAxes)
    fig.suptitle("Refusal mechanisms: Gemma-3-4B-IT vs Qwen3-4B",
                 fontsize=14, fontweight="bold", y=0.98)
    _save(fig, "summary_figure.png")


def main() -> None:
    print(f"Reading from: {RESULTS}")
    print(f"Writing to:   {FIGS}")
    fig_layer_separation()
    fig_sanity_check_projections()
    fig_causal_arditi()
    fig_mechanism_comparison()
    fig_georg_scaling()
    fig_position_cosine()
    fig_summary()
    print("done.")


if __name__ == "__main__":
    main()
