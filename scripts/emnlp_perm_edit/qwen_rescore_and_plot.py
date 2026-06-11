"""Re-score the Qwen subcircuit run with the corrected classifier and produce
the key figures for the findings report.

CPU-only: operates on the saved generations in the run's result JSONs (no GPU,
no re-generation). Re-scores with rescore_classifier.classify_corrected, then:
  - recomputes Top-K Pareto curves (keyword vs corrected)
  - recomputes Stage 08 universal-core break rates (Qwen + Gemma, both scorers)
  - writes rescore_summary.json + corrected pareto + 5 figures

Outputs to <out-dir> (default data/results/emnlp_perm_edit/qwen_subcircuits/figures).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
from rescore_classifier import classify_corrected  # noqa: E402
from qwen_subcircuits_aggregate import wilson_ci, find_knee  # noqa: E402

QSUB = REPO / "data/results/emnlp_perm_edit/qwen_subcircuits"
QRUN = REPO / "data/results/pipeline_runs_qwen/run_emnlp_qwen_L18_20260522"
GRUN = REPO / "data/results/pipeline_runs/run_20260430_023247"
JB = ["jb_fiction", "jb_roleplay", "jb_analytical", "jb_completion", "jb_cognitive_reframe"]


def scorer_label(rec, which):
    """Return REFUSE/COMPLY under the chosen scorer for a saved record."""
    if which == "keyword":
        return rec.get("classification") or rec.get("cls")
    return classify_corrected(rec["response"])


# ----------------------------------------------------------------------
# Top-K Pareto (per scorer)
# ----------------------------------------------------------------------
# proxy_edges runs with --skip-baseline (reuses proxy_features' baseline, same
# plain-HF model family). Populated in main() and used as a fallback.
_PROXY_BASELINE_FALLBACK: dict = {}


def sweep_baseline(sweep, which):
    recs = sweep.get("baseline", {}).get("records", [])
    if recs:
        return {(r["prompt_idx"], r["condition"]): scorer_label(r, which) for r in recs}
    # No own baseline (skip-baseline run) → reuse the proxy baseline.
    return _PROXY_BASELINE_FALLBACK.get(which, {})


def cell_rate(records, baseline, which, tag):
    """break (bare REFUSE->COMPLY) or recovery (jb COMPLY->REFUSE) for one cell."""
    k = n = 0
    for r in records:
        base = baseline.get((r["prompt_idx"], r["condition"]))
        if base is None:
            continue
        is_jb = r["condition"].startswith("jb_")
        lab = scorer_label(r, which)
        if tag == "recovery" and is_jb and base == "COMPLY":
            n += 1; k += (lab == "REFUSE")
        elif tag == "break" and not is_jb and base == "REFUSE":
            n += 1; k += (lab == "COMPLY")
    rate, lo, hi = wilson_ci(k, n)
    return {"rate": rate, "lo": lo, "hi": hi, "k": k, "n": n}


def curve(sweep, which, ranking, tag):
    base = sweep_baseline(sweep, which)
    pts = []
    for key, recs in sweep["per_cell"].items():
        r, ks = key.rsplit("_K", 1)
        if r != ranking:
            continue
        pts.append((int(ks), cell_rate(recs, base, which, tag)))
    return sorted(pts)


def load(name):
    return json.loads((QSUB / f"topk_sweep_{name}.json").read_text())


# ----------------------------------------------------------------------
# Stage 08 universal-core break (per scorer, per condition)
# ----------------------------------------------------------------------
def stage08_universal_break(run_dir, which, pos_mode="all"):
    d = json.loads((run_dir / "08_ablation/ablation_results.json").read_text())
    rows = d["results"] if isinstance(d, dict) else d
    per_cond = defaultdict(lambda: [0, 0])  # cond -> [broke, n_baseline_refuse]
    for row in rows:
        abl = row.get("ablations", {}).get("universal_refusal_core", {})
        modes = abl.get(pos_mode)
        if not modes:
            continue
        for cond, rec in modes.items():
            base = scorer_label(row["baseline"][cond], which)
            if base != "REFUSE":
                continue
            per_cond[cond][1] += 1
            if scorer_label(rec, which) == "COMPLY":
                per_cond[cond][0] += 1
    return {c: (k / n if n else 0.0, k, n) for c, (k, n) in per_cond.items()}


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def _errbars(pts):
    ks = [k for k, _ in pts]
    rate = [c["rate"] for _, c in pts]
    lo = [max(0.0, c["rate"] - c["lo"]) for _, c in pts]
    hi = [max(0.0, c["hi"] - c["rate"]) for _, c in pts]
    return ks, rate, [lo, hi]


def fig_scorer_correction(zero, out):
    """Baseline jb comply: keyword vs corrected, per class (Qwen)."""
    bw, bc = sweep_baseline(zero, "keyword"), sweep_baseline(zero, "corrected")
    classes, kw, co = [], [], []
    for c in JB:
        ks = [v for (pi, cc), v in bw.items() if cc == c]
        cs = [v for (pi, cc), v in bc.items() if cc == c]
        classes.append(c.replace("jb_", ""))
        kw.append(sum(x == "COMPLY" for x in ks) / len(ks) * 100)
        co.append(sum(x == "COMPLY" for x in cs) / len(cs) * 100)
    x = range(len(classes))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([i - 0.2 for i in x], kw, 0.4, label="keyword scorer", color="#d98c8c")
    ax.bar([i + 0.2 for i in x], co, 0.4, label="corrected scorer", color="#5b8c5a")
    for i, (a, b) in enumerate(zip(kw, co)):
        ax.text(i - 0.2, a + 1, f"{a:.0f}", ha="center", fontsize=8)
        ax.text(i + 0.2, b + 1, f"{b:.0f}", ha="center", fontsize=8)
    ax.set_ylabel("jailbreak COMPLY rate (%)")
    ax.set_title("Scorer correction: keyword over-counts compliance (Qwen baselines)")
    ax.set_xticks(list(x)); ax.set_xticklabels(classes, rotation=15)
    ax.set_ylim(0, 105); ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_sparsity_asymmetry(zero, out):
    """Break refusal (bare,pos) vs remove jailbreak (jb,neg), corrected, zero mech."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for ranking, tag, lab, col in [("pos", "break", "break refusal (bare)", "#c0392b"),
                                    ("neg", "recovery", "remove jailbreak (jb→refuse)", "#2471a3")]:
        ks, rate, err = _errbars(curve(zero, "corrected", ranking, tag))
        ax.errorbar(ks, [r * 100 for r in rate], yerr=[[e * 100 for e in err[0]], [e * 100 for e in err[1]]],
                    marker="o", capsize=3, label=lab, color=col, lw=2)
    ax.set_xscale("log"); ax.set_xlabel("K (features zeroed per prompt)")
    ax.set_ylabel("flip rate (%, corrected scorer)")
    ax.set_title("Refusal is sparse; jailbreak-suppression is distributed (Qwen, true zeroing)")
    ax.set_ylim(0, 105); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_edge_vs_node(zero, pf, pe, out):
    fig, ax = plt.subplots(figsize=(8, 5))
    for sweep, lab, col in [(zero, "zero features (true)", "#c0392b"),
                            (pf, "proxy features", "#e67e22"),
                            (pe, "proxy edges", "#16a085")]:
        ks, rate, err = _errbars(curve(sweep, "corrected", "pos", "break"))
        ax.errorbar(ks, [r * 100 for r in rate], yerr=[[e * 100 for e in err[0]], [e * 100 for e in err[1]]],
                    marker="o", capsize=3, label=lab, color=col, lw=2)
    ax.set_xscale("log"); ax.set_xlabel("K (top-K ablated per prompt)")
    ax.set_ylabel("bare refusal break rate (%, corrected)")
    ax.set_title("Edge > node, and true-zeroing > proxy (Qwen, break refusal)")
    ax.set_ylim(0, 105); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_ranking(zero, out):
    fig, ax = plt.subplots(figsize=(8, 5))
    for ranking, lab, col in [("pos", "attribution-ranked", "#8e44ad"),
                              ("activation", "activation-ranked", "#7f8c8d")]:
        ks, rate, err = _errbars(curve(zero, "corrected", ranking, "break"))
        ax.errorbar(ks, [r * 100 for r in rate], yerr=[[e * 100 for e in err[0]], [e * 100 for e in err[1]]],
                    marker="o", capsize=3, label=lab, color=col, lw=2)
    ax.set_xscale("log"); ax.set_xlabel("K (features zeroed per prompt)")
    ax.set_ylabel("bare refusal break rate (%, corrected)")
    ax.set_title("Attribution-ranked features beat activation-ranked (Qwen, true zeroing)")
    ax.set_ylim(0, 105); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_cross_model(qb, gb, out):
    """universal-core bare break: Qwen vs Gemma, keyword vs corrected."""
    models = ["Gemma-3-4B\n(L15)", "Qwen3-4B\n(L18)"]
    kw = [gb["keyword"]["bare"][0] * 100, qb["keyword"]["bare"][0] * 100]
    co = [gb["corrected"]["bare"][0] * 100, qb["corrected"]["bare"][0] * 100]
    x = range(2)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.bar([i - 0.2 for i in x], kw, 0.4, label="keyword scorer", color="#d98c8c")
    ax.bar([i + 0.2 for i in x], co, 0.4, label="corrected scorer", color="#5b8c5a")
    for i, (a, b) in enumerate(zip(kw, co)):
        ax.text(i - 0.2, a + 1.5, f"{a:.0f}%", ha="center", fontsize=9)
        ax.text(i + 0.2, b + 1.5, f"{b:.0f}%", ha="center", fontsize=9)
    ax.set_ylabel("bare refusal break rate (%)")
    ax.set_title("universal_refusal_core ablation breaks refusal on Qwen, not Gemma")
    ax.set_xticks(list(x)); ax.set_xticklabels(models)
    ax.set_ylim(0, 105); ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=QSUB / "figures")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    zero = load("zero_features"); pf = load("proxy_features"); pe = load("proxy_edges")
    # Seed the proxy baseline fallback (proxy_edges reuses it).
    for which in ("keyword", "corrected"):
        _PROXY_BASELINE_FALLBACK[which] = {
            (r["prompt_idx"], r["condition"]): scorer_label(r, which)
            for r in pf.get("baseline", {}).get("records", [])}

    # ---- corrected vs keyword summary ----
    summary = {"scorer": "rescore_classifier.classify_corrected", "knees": {}, "stage08": {}}
    for nm, sw in [("zero_features", zero), ("proxy_features", pf), ("proxy_edges", pe)]:
        summary["knees"][nm] = {}
        rankings = sw["metadata"]["rankings"]
        for which in ("keyword", "corrected"):
            for ranking in rankings:
                for tag in ("break", "recovery"):
                    c = curve(sw, which, ranking, tag)
                    if not c or all(p[1]["n"] == 0 for p in c):
                        continue
                    knee = find_knee([(k, v["rate"]) for k, v in c], 0.5)
                    summary["knees"][nm][f"{which}/{ranking}/{tag}@50"] = knee
                    summary["knees"][nm][f"{which}/{ranking}/{tag}/curve"] = [
                        {"K": k, **v} for k, v in c]

    qb = {w: stage08_universal_break(QRUN, w) for w in ("keyword", "corrected")}
    gb = {w: stage08_universal_break(GRUN, w) for w in ("keyword", "corrected")}
    summary["stage08"]["qwen_universal_break"] = qb
    summary["stage08"]["gemma_universal_break"] = gb
    (args.out_dir / "rescore_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # ---- figures ----
    fig_scorer_correction(zero, args.out_dir / "fig1_scorer_correction.png")
    fig_cross_model(qb, gb, args.out_dir / "fig2_cross_model_break.png")
    fig_sparsity_asymmetry(zero, args.out_dir / "fig3_sparsity_asymmetry.png")
    fig_edge_vs_node(zero, pf, pe, args.out_dir / "fig4_edge_vs_node.png")
    fig_ranking(zero, args.out_dir / "fig5_ranking.png")

    # ---- console digest ----
    print("=== Stage 08 universal_refusal_core bare break (all-pos) ===")
    for model, b in [("Qwen", qb), ("Gemma", gb)]:
        print(f"  {model}: keyword={b['keyword']['bare'][0]:.0%}  corrected={b['corrected']['bare'][0]:.0%}")
    print("\n=== Top-K knees (corrected, K@50%) ===")
    for nm in ("zero_features", "proxy_features", "proxy_edges"):
        kn = summary["knees"][nm]
        brk = {k.split("/")[1]: v for k, v in kn.items() if k.startswith("corrected/") and "break@50" in k}
        rec = {k.split("/")[1]: v for k, v in kn.items() if k.startswith("corrected/") and "recovery@50" in k}
        print(f"  {nm}: break@50%={brk}  recovery@50%={rec}")
    print(f"\nWrote 5 figures + rescore_summary.json to {args.out_dir}")


if __name__ == "__main__":
    main()
