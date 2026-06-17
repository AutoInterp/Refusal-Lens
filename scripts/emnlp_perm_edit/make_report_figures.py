"""Regenerate the figures embedded in docs/REFUSAL_DIRECTION_INVESTIGATION_2026-06-16.md.

Outputs to docs/figures/:
  fig1_rhat_barplots.png            — per-model r_hat per-dimension (Gemma spike vs Qwen)
  fig2_edge_ablation_crossmodel.png — cross-model per-variant edge ablation (n=50, bare)

CPU only. Run from repo root: .venv/bin/python scripts/emnlp_perm_edit/make_report_figures.py
"""
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
PH = REPO / "data/results/emnlp_perm_edit/phase0_controllability"
FIG = REPO / "docs/figures"
FIG.mkdir(parents=True, exist_ok=True)


def fig1():
    rg = torch.load(REPO / "data/results/pipeline_runs/run_20260430_023247/01_direction/unnormalized_r.pt",
                    weights_only=False)[15].float().numpy()
    rq = torch.load(REPO / "data/results/pipeline_runs_qwen/run_regen_L18/01_direction/positions_L18/pos_-1_unnormalized.pt",
                    weights_only=False).float().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
    # Gemma: the outlier is 1 bar of 2560 (sub-pixel) -> draw a wide highlighted bar +
    # annotation so it's visible regardless of rendering width.
    gi = int(np.abs(rg).argmax())
    ax[0].bar(np.arange(len(rg)), rg, width=1.0, color="0.6", linewidth=0)
    ax[0].bar([gi], [rg[gi]], width=8, color="crimson", linewidth=0, zorder=3)
    ax[0].plot([gi], [rg[gi]], "o", color="crimson", ms=4, zorder=4)
    ax[0].annotate(f"dim #{gi} = {rg[gi]:.0f}  (81% of sq-norm)", xy=(gi, rg[gi]),
                   xytext=(gi + 250, rg[gi] * 0.6), fontsize=8.5, color="crimson",
                   arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8))
    ax[0].set_title("Gemma-3-4B r_hat (L15) — dominated by one massive-activation dim")
    ax[0].set_xlabel("hidden dimension"); ax[0].set_ylabel("r_hat value"); ax[0].axhline(0, color="k", lw=0.5)
    qi = int(np.abs(rq).argmax())
    ax[1].bar(np.arange(len(rq)), rq, width=1.0, color="steelblue", linewidth=0)
    ax[1].set_title(f"Qwen3-4B r_hat (L18) — distributed (top dim #{qi} = {rq[qi]:.2f}, 0.9% of sq-norm)")
    ax[1].set_xlabel("hidden dimension"); ax[1].axhline(0, color="k", lw=0.5)
    fig.tight_layout(); fig.savefig(FIG / "fig1_rhat_barplots.png", dpi=120); plt.close(fig)


def _comply_bare(path):
    recs = json.load(open(path))["records"]
    out = {}
    for s in ["baseline", "complement", "outlier", "full"]:
        sel = [r for r in recs if r["setting"] == s and r["condition"] == "bare"]
        out[s] = 100 * sum(1 for r in sel if r["judge_class"] == "COMPLY") / len(sel)
    return out


def fig2():
    g = _comply_bare(PH / "gemma_pervariant_edgeabl_n50_judged.json")
    q = _comply_bare(PH / "qwen_pervariant_edgeabl_n50_judged.json")
    order = ["baseline", "complement", "outlier", "full"]
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    b1 = ax.bar(x - w / 2, [g[s] for s in order], w, label="Gemma", color="crimson")
    b2 = ax.bar(x + w / 2, [q[s] for s in order], w, label="Qwen", color="steelblue")
    ax.set_xticks(x); ax.set_xticklabels(["baseline", "complement", "outlier\n(#443)", "full"])
    ax.set_ylabel("refusal broken (COMPLY%, LLM-judge)"); ax.set_ylim(0, 100)
    ax.set_title("Per-variant edge ablation on harmful (bare) prompts, n=50\n"
                 "complement = refusal circuit; outlier inert; Gemma-full = gibberish")
    ax.bar_label(b1, fmt="%.0f"); ax.bar_label(b2, fmt="%.0f"); ax.legend()
    ax.annotate("gibberish\n(0% coherent)", xy=(3 - w / 2, 2), xytext=(2.4, 30), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="gray"))
    fig.tight_layout(); fig.savefig(FIG / "fig2_edge_ablation_crossmodel.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2()
    print("wrote docs/figures/fig1_rhat_barplots.png, fig2_edge_ablation_crossmodel.png")
