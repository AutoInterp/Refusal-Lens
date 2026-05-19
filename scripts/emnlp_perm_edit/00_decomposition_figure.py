"""Phase 0 — stacked-bar figure of direct_dot decomposition per condition.

X-axis: 11 conditions (bare, 5 jb_*, 5 ctrl_*).
Y-axis: signed magnitude of contribution to direct_dot.
Stacked: feature_signed | embedding_signed | error_signed.

Sign convention (per REPORT § 4): more negative direct_dot = LESS refusal
(JBs shift residual along -r_hat direction, more negative direct_dot).
Edge contributions inherit the same sign frame.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]

CONDITIONS = [
    "bare",
    "jb_fiction", "jb_roleplay", "jb_analytical", "jb_completion", "jb_cognitive_reframe",
    "ctrl_fiction", "ctrl_roleplay", "ctrl_analytical", "ctrl_completion", "ctrl_cognitive_reframe",
]
COMPONENT_COLORS = {
    "feature": "#4C72B0",
    "embedding": "#55A868",
    "error_node": "#C44E52",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-file", type=Path,
                   default=REPO / "data/results/emnlp_perm_edit/phase0_controllability/decomposition_by_condition.json")
    p.add_argument("--out", type=Path,
                   default=REPO / "data/results/emnlp_perm_edit/phase0_controllability/decomposition_figure.png")
    return p.parse_args()


def main():
    args = parse_args()
    data = json.loads(args.in_file.read_text())["per_condition"]

    n = len(CONDITIONS)
    feat = np.zeros(n)
    embed = np.zeros(n)
    err = np.zeros(n)
    total = np.zeros(n)

    for i, cond in enumerate(CONDITIONS):
        agg = data.get(cond, {})
        if agg.get("n", 0) == 0:
            continue
        feat[i] = agg.get("feature_signed_mean", 0.0)
        embed[i] = agg.get("embedding_signed_mean", 0.0)
        err[i] = agg.get("error_signed_mean", 0.0)
        total[i] = agg.get("all_signed_mean", 0.0)

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(n)
    width = 0.7

    # All values are mostly negative on this dataset; stack downward
    bottoms_pos = np.zeros(n)
    bottoms_neg = np.zeros(n)
    for vals, label, color in [
        (feat, "feature edges", COMPONENT_COLORS["feature"]),
        (embed, "embedding edges", COMPONENT_COLORS["embedding"]),
        (err, "error_node edges", COMPONENT_COLORS["error_node"]),
    ]:
        pos = np.where(vals > 0, vals, 0)
        neg = np.where(vals < 0, vals, 0)
        ax.bar(x, pos, width, bottom=bottoms_pos, color=color, label=label)
        ax.bar(x, neg, width, bottom=bottoms_neg, color=color)
        bottoms_pos += pos
        bottoms_neg += neg

    # Overlay total signed as a marker
    ax.scatter(x, total, color="black", marker="D", s=50, zorder=5,
               label="total signed (sum of stacked components)")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS, rotation=45, ha="right")
    ax.set_ylabel("Signed attribution sum into direct_dot at L15 pos=-2\n"
                  "(filtered edges; more negative = stronger 'pull away from refusal axis')")
    ax.set_title("Phase 0 — Linearization decomposition of direct_dot per condition\n"
                 "(50 prompts each; packed-graph edges filtered at threshold=0.98 per Stage 02c)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"[0a-figure] wrote {args.out}")


if __name__ == "__main__":
    main()
