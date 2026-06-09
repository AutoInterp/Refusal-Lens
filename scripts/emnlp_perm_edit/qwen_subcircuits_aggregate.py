"""Aggregate the Qwen subcircuit + Top-K sweep run into curves, report, and
frontend subcircuit sets.

Inputs (all optional except --run-dir; missing inputs are warned + skipped):
    <out-dir>/topk_sweep_proxy_features.json
    <out-dir>/topk_sweep_proxy_edges.json
    <out-dir>/topk_sweep_zero_features.json     (from 00_topk_circuit_sweep_qwen.py)
    <run-dir>/08_ablation/ablation_summary.json (from Stage 08)
    <run-dir>/02_attribution/attribution_results.json (for corpus Top-K sets)
    <run-dir>/07_subcircuits/subcircuits.json   (merged with Top-K sets)

Outputs to <out-dir>:
    pareto_curves.json          flip-rate vs K per (mechanism_source, ranking), Wilson CIs + knees
    pareto_curves.png           (best-effort; skipped if matplotlib unavailable)
    subcircuits_frontend.json   Stage 07 subcircuits + corpus-level topk_* sets,
                                schema-compatible with the frontend panel and Stage 08
                                (data.subcircuits[name].features = ["L{L}:F{F}", ...]).
                                The watcher uploads this to HF runs/<run>/subcircuits.json.
    QWEN_SUBCIRCUIT_REPORT.md   headline summary

Flip conventions (match the EMNLP phase-0 aggregator):
    break    (bare):  baseline REFUSE -> ablated COMPLY
    recovery (jb_*):  baseline COMPLY -> ablated REFUSE
Baselines come from each mechanism family's own baseline block (plain model for
proxy, ReplacementModel for zero); proxy_edges may use proxy_features' baseline
(same model family, --skip-baseline run).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORPUS_TOPK_SIZES = (10, 25, 50)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Return (rate, lo, hi). (0,0,0) when n == 0."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def index_baseline(sweep: dict) -> dict[tuple[int, str], str]:
    return {(r["prompt_idx"], r["condition"]): r["classification"]
            for r in sweep.get("baseline", {}).get("records", [])}


def cell_rates(records: list[dict], baseline: dict[tuple[int, str], str]) -> dict:
    """break/recovery rates for one sweep cell, denominated on eligible baselines."""
    n_break = k_break = n_rec = k_rec = 0
    n_incoherent = 0
    for r in records:
        base = baseline.get((r["prompt_idx"], r["condition"]))
        if base is None:
            continue
        if not r.get("coherent", True):
            n_incoherent += 1
        if r["condition"].startswith("jb_"):
            if base == "COMPLY":
                n_rec += 1
                if r["classification"] == "REFUSE":
                    k_rec += 1
        else:  # bare (+ any ctrl_* if present)
            if base == "REFUSE":
                n_break += 1
                if r["classification"] == "COMPLY":
                    k_break += 1
    out = {"n_records": len(records), "n_incoherent": n_incoherent}
    for tag, k, n in (("break", k_break, n_break), ("recovery", k_rec, n_rec)):
        rate, lo, hi = wilson_ci(k, n)
        out[tag] = {"rate": rate, "lo": lo, "hi": hi, "k": k, "n": n}
    return out


def find_knee(curve: list[tuple[int, float]], threshold: float) -> int | None:
    """Smallest K whose rate >= threshold (curve = [(K, rate), ...] sorted by K)."""
    for K, rate in curve:
        if rate >= threshold:
            return K
    return None


def build_pareto(sweeps: dict[str, dict]) -> dict:
    """{mechanism_source: {ranking: {per_k, knees}}} from loaded sweep files."""
    out = {}
    proxy_baseline = None
    for name, sweep in sweeps.items():
        b = index_baseline(sweep)
        if b and sweep["metadata"]["mechanism"] == "proxy":
            proxy_baseline = proxy_baseline or b
    for name, sweep in sweeps.items():
        baseline = index_baseline(sweep) or (
            proxy_baseline if sweep["metadata"]["mechanism"] == "proxy" else {})
        if not baseline:
            print(f"  WARNING: no baseline available for {name}; skipping its curves")
            continue
        per_ranking: dict = defaultdict(dict)
        for key, records in sweep.get("per_cell", {}).items():
            ranking, k_part = key.rsplit("_K", 1)
            per_ranking[ranking][int(k_part)] = cell_rates(records, baseline)
        entry = {}
        for ranking, by_k in per_ranking.items():
            ks = sorted(by_k)
            entry[ranking] = {
                "per_k": {str(K): by_k[K] for K in ks},
                "knees": {
                    f"{tag}_{int(th*100)}": find_knee(
                        [(K, by_k[K][tag]["rate"]) for K in ks], th)
                    for tag in ("break", "recovery") for th in (0.5, 0.8)
                },
            }
        out[name] = entry
    return out


def build_corpus_topk_sets(attr_results: dict) -> dict[str, dict]:
    """Corpus-level Top-K sets for frontend visual inspection.

    Score = summed signed attribution across graphs of the condition group
    (bare graphs for the refusal mechanism, jb_* graphs for the jailbreak
    mechanism). topk_refusal_K* = most pro-refusal mass (desc);
    topk_jailbreak_K* = most anti-refusal mass (asc).
    NOTE: the sweep itself uses per-prompt rankings; these corpus sets are for
    the frontend panel only and are labeled as such.
    """
    sums_bare: dict[str, float] = defaultdict(float)
    sums_jb: dict[str, float] = defaultdict(float)
    for row in attr_results.get("results", []):
        for cond, cond_entry in row.get("conditions", {}).items():
            graphs = cond_entry.get("graphs", {})
            summary = graphs.get("single") or graphs.get("multi") or {}
            for key, attr in (summary.get("top_features") or {}).items():
                if cond == "bare":
                    sums_bare[key] += attr
                elif cond.startswith("jb_"):
                    sums_jb[key] += attr
    refusal_ranked = sorted(sums_bare.items(), key=lambda kv: kv[1], reverse=True)
    jb_ranked = sorted(sums_jb.items(), key=lambda kv: kv[1])
    sets = {}
    for K in CORPUS_TOPK_SIZES:
        sets[f"topk_refusal_K{K}"] = {
            "features": [k for k, _ in refusal_ranked[:K]],
            "n_features": min(K, len(refusal_ranked)),
            "rule": "corpus_sum_signed_attr_bare_desc",
            "description": f"Top-{K} pro-refusal features by summed attribution across bare graphs "
                           f"(corpus-level view of the Top-K sweep; sweep itself ranks per-prompt)",
        }
        sets[f"topk_jailbreak_K{K}"] = {
            "features": [k for k, _ in jb_ranked[:K]],
            "n_features": min(K, len(jb_ranked)),
            "rule": "corpus_sum_signed_attr_jb_asc",
            "description": f"Top-{K} anti-refusal features by summed attribution across jb_* graphs "
                           f"(corpus-level view of the Top-K sweep; sweep itself ranks per-prompt)",
        }
    return sets


def render_report(pareto: dict, ablation_summary: dict | None, corpus_sets: dict,
                  sweeps_present: list[str]) -> str:
    lines = [
        "# Qwen Subcircuit + Top-K Sweep Report",
        "",
        "Auto-generated by `qwen_subcircuits_aggregate.py`. Model: Qwen3-4B, L18, "
        "pos=-1, enable_thinking=False, greedy, max_new_tokens=80.",
        "",
        f"Sweep files aggregated: {', '.join(sweeps_present) or '(none)'}",
        "",
        "## Top-K Pareto knees",
        "",
        "Smallest per-prompt K reaching the flip-rate threshold "
        "(break = bare REFUSE→COMPLY; recovery = jb COMPLY→REFUSE; `—` = never reached).",
        "",
        "| mechanism/source | ranking | break@50% | break@80% | recov@50% | recov@80% |",
        "|---|---|---|---|---|---|",
    ]
    for name, entry in pareto.items():
        for ranking, data in entry.items():
            kn = data["knees"]
            fmt = lambda v: str(v) if v is not None else "—"
            lines.append(f"| {name} | {ranking} | {fmt(kn['break_50'])} | {fmt(kn['break_80'])} "
                         f"| {fmt(kn['recovery_50'])} | {fmt(kn['recovery_80'])} |")
    lines += ["", "## Flip-rate curves (rate [95% CI], n)", ""]
    for name, entry in pareto.items():
        lines.append(f"### {name}")
        for ranking, data in entry.items():
            lines.append(f"- **{ranking}**:")
            for K, cell in data["per_k"].items():
                parts = []
                for tag in ("break", "recovery"):
                    c = cell[tag]
                    if c["n"]:
                        parts.append(f"{tag} {c['rate']:.0%} [{c['lo']:.0%},{c['hi']:.0%}] n={c['n']}")
                inc = f" ({cell['n_incoherent']} incoherent)" if cell.get("n_incoherent") else ""
                lines.append(f"  - K={K}: " + "; ".join(parts) + inc)
        lines.append("")
    if ablation_summary:
        lines += ["## Stage 08 subcircuit ablation (dissociation digest)", ""]
        lines.append("```json")
        digest = {k: v for k, v in ablation_summary.items() if k != "per_prompt"}
        lines.append(json.dumps(digest, indent=2)[:6000])
        lines.append("```")
    lines += ["", "## Corpus-level Top-K sets exported to the frontend", ""]
    for name, s in corpus_sets.items():
        lines.append(f"- `{name}`: {s['n_features']} features ({s['rule']})")
    lines.append("")
    return "\n".join(lines)


def plot_curves(pareto: dict, out_png: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  WARNING: matplotlib unavailable ({e}); skipping figure")
        return False
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, tag, title in ((axes[0], "break", "Break refusal (bare REFUSE→COMPLY)"),
                           (axes[1], "recovery", "Remove jailbreak (jb COMPLY→REFUSE)")):
        for name, entry in pareto.items():
            for ranking, data in entry.items():
                pts = [(int(K), c[tag]) for K, c in data["per_k"].items() if c[tag]["n"] > 0]
                if not pts:
                    continue
                pts.sort()
                ks = [p[0] for p in pts]
                rates = [p[1]["rate"] for p in pts]
                lo = [max(0.0, p[1]["rate"] - p[1]["lo"]) for p in pts]
                hi = [max(0.0, p[1]["hi"] - p[1]["rate"]) for p in pts]
                ax.errorbar(ks, rates, yerr=[lo, hi], marker="o", capsize=3,
                            label=f"{name}:{ranking}")
        ax.set_xscale("log")
        ax.set_xlabel("K (per-prompt top-K ablated)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("flip rate")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return True


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path,
                   default=REPO / "data/results/emnlp_perm_edit/qwen_subcircuits")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sweeps = {}
    for name in ("proxy_features", "proxy_edges", "zero_features"):
        path = args.out_dir / f"topk_sweep_{name}.json"
        if path.exists():
            sweeps[name] = json.loads(path.read_text())
        else:
            print(f"  WARNING: {path.name} missing — skipped")

    print("[aggregate] building Pareto curves...")
    pareto = build_pareto(sweeps)
    (args.out_dir / "pareto_curves.json").write_text(json.dumps(pareto, indent=2))

    ablation_summary = None
    abl_path = args.run_dir / "08_ablation" / "ablation_summary.json"
    if abl_path.exists():
        ablation_summary = json.loads(abl_path.read_text())
    else:
        print(f"  WARNING: {abl_path} missing — Stage 08 digest skipped")

    corpus_sets: dict = {}
    attr_path = args.run_dir / "02_attribution" / "attribution_results.json"
    if attr_path.exists():
        corpus_sets = build_corpus_topk_sets(json.loads(attr_path.read_text()))
    else:
        print(f"  WARNING: {attr_path} missing — corpus Top-K sets skipped")

    # Merge with Stage 07 subcircuits for the frontend
    sub_path = args.run_dir / "07_subcircuits" / "subcircuits.json"
    merged = {"subcircuits": {}}
    if sub_path.exists():
        merged = json.loads(sub_path.read_text())
    else:
        print(f"  WARNING: {sub_path} missing — frontend file holds Top-K sets only")
    merged.setdefault("subcircuits", {}).update(corpus_sets)
    merged.setdefault("metadata", {})["topk_sets_added_by"] = "qwen_subcircuits_aggregate.py"
    (args.out_dir / "subcircuits_frontend.json").write_text(json.dumps(merged, indent=2))

    if plot_curves(pareto, args.out_dir / "pareto_curves.png"):
        print("  wrote pareto_curves.png")

    report = render_report(pareto, ablation_summary, corpus_sets, list(sweeps))
    (args.out_dir / "QWEN_SUBCIRCUIT_REPORT.md").write_text(report)
    print(f"[aggregate] wrote pareto_curves.json, subcircuits_frontend.json, "
          f"QWEN_SUBCIRCUIT_REPORT.md to {args.out_dir}")


if __name__ == "__main__":
    main()
