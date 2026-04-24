"""
Stage 08a: Subcircuit ablation via ReplacementModel feature intervention
=========================================================================
Ablates Stage 07-defined subcircuits (or a manual cart) by zero-setting the
transcoder feature activations at runtime. For each (subcircuit, positions-mode,
prompt, condition), runs a baseline + an ablated generation and records the
classification flip.

Two position modes:
  all     — intervention applied at every position via slice(None) (Arditi-style;
            matches Stage 06 convention).
  anchors — intervention applied only at Gemma-3 template-anchor positions
            [-5, -3, -2], resolved per prompt to absolute indices. Matches the
            Stage 02 attribution-target choice.

Validation framework (the NeurIPS figure):
  POSITIVE control — universal_refusal_core: ablation must break bare refuse.
  NEGATIVE control — ctrl_shared_refusal:    ablation must NOT affect JB flip.
  DISSOCIATION     — jb_{class}_specific_vs_ctrl: ablating class X drops class
                     X's JB flip rate substantially more than other classes'.

Inputs:
    <run-dir>/07_subcircuits/subcircuits.json  — feature sets
    <run-dir>/06_causal/causal_results.json    — baselines (optional reuse)
    dataset/refusal_lens_controlled_dataset.json

Outputs to <run-dir>/08_ablation/:
    ablation_results.json           — per-(prompt, condition, subcircuit, positions-mode)
    ablation_checkpoint.json        — resume state
    ablation_summary.json           — aggregated flip-rate matrix
    dissociation_matrix.png         — class × subcircuit heatmap (main figure)
    positions_comparison.png        — all vs anchors flip rate comparison
    ABLATION_SUMMARY.md             — human-readable headline summary

Usage:
    PYTHONPATH=src python3 08_ablate_subcircuits.py \\
        --run-dir <run-dir> \\
        --subcircuits universal_refusal_core,jb_fiction_specific_vs_ctrl \\
        --positions both

    # Smoke test
    PYTHONPATH=src python3 08_ablate_subcircuits.py \\
        --run-dir <run-dir> --max-prompts 2

    # Manual cart (from Stage 05 frontend export)
    PYTHONPATH=src python3 08_ablate_subcircuits.py \\
        --run-dir <run-dir> --feature-file cart.json --ablation-name my_cart
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from utils import (
    classify_response,
    format_prompt,
    get_stage_dir,
    is_coherent,
    load_cart,
    load_controlled_dataset,
    load_json,
    load_subcircuit_features,
    resolve_anchor_positions,
    save_json,
)

JB_CLASSES = ("analytical", "cognitive_reframe", "completion", "fiction", "roleplay")
POSITIONS_MODES = ("all", "anchors")


def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 08a: subcircuit feature ablation via ReplacementModel",
    )
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Run directory containing 07_subcircuits/subcircuits.json")
    p.add_argument("--subcircuits", type=str, default=None,
                   help=("Comma-separated subcircuit names to ablate "
                         "(default: config.STAGE_08_DEFAULT_SUBCIRCUITS)"))
    p.add_argument("--feature-file", type=Path, default=None,
                   help="Optional cart.json from Stage 05 frontend. Produces a single "
                        "ablation condition named --ablation-name (default 'cart').")
    p.add_argument("--ablation-name", type=str, default="cart",
                   help="Name to label the --feature-file ablation condition")
    p.add_argument("--positions", choices=["all", "anchors", "both"], default="both",
                   help="Position mode(s). 'both' runs the comparative analysis.")
    p.add_argument("--conditions", type=str, default=None,
                   help="Comma-separated condition names (default: all 11)")
    p.add_argument("--layer", type=int, default=None,
                   help="If set, restrict ablation features to this source layer only "
                        "(useful for ablating a manual single-layer cart).")
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"],
                   default="bfloat16")
    p.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    p.add_argument("--max-prompts", type=int, default=None,
                   help="Limit to first N prompts (smoke test)")
    p.add_argument("--prompt-start", type=int, default=0)
    p.add_argument("--prompt-end", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--skip-baseline", action="store_true",
                   help="Skip baseline generation; read from 06_causal/causal_results.json")
    return p.parse_args()


# --------------------------------------------------------------------
# Ablation set resolution (subcircuits + manual cart)
# --------------------------------------------------------------------

def resolve_ablation_sets(args, run_dir: Path) -> dict[str, list[tuple[int, int]]]:
    """Build the mapping {ablation_name → [(layer, feat_idx), ...]} for this run.

    Two sources that can be combined:
      - --subcircuits: named subcircuit sets from Stage 07
      - --feature-file: a single cart.json (produces one ablation condition)
    """
    out: dict[str, list[tuple[int, int]]] = {}
    sub_json = run_dir / "07_subcircuits" / "subcircuits.json"

    names = args.subcircuits.split(",") if args.subcircuits else list(config.STAGE_08_DEFAULT_SUBCIRCUITS)
    if args.subcircuits != "" and not args.feature_file or args.subcircuits:
        if not sub_json.exists():
            raise FileNotFoundError(f"{sub_json} missing — run Stage 07 first or use --feature-file only.")
        for name in names:
            features = load_subcircuit_features(sub_json, [name])
            if args.layer is not None:
                features = [(L, F) for (L, F) in features if L == args.layer]
            out[name] = features

    if args.feature_file:
        cart = load_cart(args.feature_file)
        features = [(L, F) for (L, F, _v) in cart]
        if args.layer is not None:
            features = [(L, F) for (L, F) in features if L == args.layer]
        out[args.ablation_name] = features

    if not out:
        raise ValueError("No ablation sets resolved. Pass --subcircuits and/or --feature-file.")
    return out


# --------------------------------------------------------------------
# Intervention spec construction
# --------------------------------------------------------------------

def build_interventions(
    features: list[tuple[int, int]],
    positions_mode: str,
    tokenized_length: int,
) -> list[tuple]:
    """Return the list of (layer, position_spec, feat_idx, 0.0) interventions
    for `ReplacementModel.feature_intervention_generate`.

    positions_mode:
      - 'all'     → one intervention per feature at position slice(None).
                    The library remaps this to position 0 during generation via
                    _convert_open_ended_interventions (effectively "every token").
      - 'anchors' → one intervention per (feature, anchor_pos) at absolute
                    resolved indices of STAGE_08_TEMPLATE_ANCHORS.
    """
    if positions_mode == "all":
        pos_spec = slice(None)
        return [(L, pos_spec, F, 0.0) for (L, F) in features]

    if positions_mode == "anchors":
        anchors = resolve_anchor_positions(
            list(config.STAGE_08_TEMPLATE_ANCHORS), tokenized_length,
        )
        out: list[tuple] = []
        for (L, F) in features:
            for pos in anchors:
                out.append((L, pos, F, 0.0))
        return out

    raise ValueError(f"Unknown positions_mode {positions_mode!r}")


# --------------------------------------------------------------------
# Generation + classification
# --------------------------------------------------------------------

def _extract_response(full_decoded: str, formatted: str) -> str:
    """feature_intervention_generate returns the full decoded sequence including
    the input. Strip the formatted prompt prefix to recover just the response."""
    if full_decoded.startswith(formatted):
        return full_decoded[len(formatted):]
    # Fallback: look for 'model\n' (Gemma-3 chat template tail)
    marker = "<start_of_turn>model\n"
    idx = full_decoded.rfind(marker)
    if idx >= 0:
        return full_decoded[idx + len(marker):]
    return full_decoded


def generate_ablated(rm, tokenizer, formatted_prompt: str, interventions,
                     max_new_tokens: int) -> str:
    """One ablated generation via feature_intervention_generate. Returns response text."""
    decoded, _logits, _cache = rm.feature_intervention_generate(
        formatted_prompt,
        interventions,
        max_new_tokens=max_new_tokens,
        return_activations=False,
        do_sample=False,
    )
    return _extract_response(decoded, formatted_prompt).strip()


def generate_baseline_rm(rm, tokenizer, formatted_prompt: str, max_new_tokens: int) -> str:
    """Baseline generation via ReplacementModel with no interventions. We still use
    feature_intervention_generate for consistency (same tokenizer / decode path)."""
    decoded, _logits, _cache = rm.feature_intervention_generate(
        formatted_prompt,
        [],
        max_new_tokens=max_new_tokens,
        return_activations=False,
        do_sample=False,
    )
    return _extract_response(decoded, formatted_prompt).strip()


def classify_record(response: str) -> dict:
    return {
        "cls": classify_response(response),
        "coherent": is_coherent(response),
        "response": response[:300],
    }


# --------------------------------------------------------------------
# Baseline caching (reuse Stage 06 output when present)
# --------------------------------------------------------------------

def try_reuse_stage06_baselines(run_dir: Path) -> dict | None:
    """If 06_causal/causal_results.json exists, return {prompt_id: {cond: {cls, coherent, response}}}.

    Matches Stage 06's per-prompt baseline schema exactly. Returning None means
    caller should generate baselines fresh.
    """
    path = run_dir / "06_causal" / "causal_results.json"
    if not path.exists():
        return None
    try:
        data = load_json(path)
    except Exception:
        return None
    out: dict = {}
    for r in data.get("results", []):
        pid = r.get("prompt_id")
        if pid is None:
            continue
        out[pid] = r.get("baseline", {})
    return out if out else None


# --------------------------------------------------------------------
# Per-prompt pipeline
# --------------------------------------------------------------------

def process_prompt(
    rm, tokenizer, row, ablation_sets, conditions, positions_modes,
    max_new_tokens, reused_baselines,
):
    """Run baselines + ablation conditions for one prompt. Returns a result row.

    Ablation is applied at layer-specific source positions; since features in
    an ablation set may span multiple source layers, we build one combined
    intervention list per (ablation_name, positions_mode) and run a single
    generation per condition.
    """
    result = {
        "prompt_id": row["id"],
        "topic": row.get("topic"),
        "base": row["base"],
        "baseline": {},
        "ablations": {},  # {ablation_name: {positions_mode: {cond: classification}}}
    }

    # ---- Baselines ----
    if reused_baselines is not None and row["id"] in reused_baselines:
        for cond in conditions:
            if cond in reused_baselines[row["id"]]:
                result["baseline"][cond] = dict(reused_baselines[row["id"]][cond])
    missing_cond = [c for c in conditions if c not in result["baseline"]]
    for cond in missing_cond:
        formatted = format_prompt(tokenizer, row["conditions"][cond]["text"])
        resp = generate_baseline_rm(rm, tokenizer, formatted, max_new_tokens)
        result["baseline"][cond] = classify_record(resp)

    # ---- Ablation conditions ----
    for abl_name, features in ablation_sets.items():
        if not features:
            # Skip empty ablation sets (e.g. layer-filtered to nothing)
            continue
        result["ablations"][abl_name] = {"n_features": len(features)}
        for pos_mode in positions_modes:
            per_cond: dict = {}
            for cond in conditions:
                formatted = format_prompt(tokenizer, row["conditions"][cond]["text"])
                tokenized_length = len(tokenizer(formatted)["input_ids"])
                interventions = build_interventions(features, pos_mode, tokenized_length)
                resp = generate_ablated(rm, tokenizer, formatted, interventions, max_new_tokens)
                per_cond[cond] = classify_record(resp)
                per_cond[cond]["changed_vs_baseline"] = (
                    per_cond[cond]["cls"] != result["baseline"][cond]["cls"]
                )
            result["ablations"][abl_name][pos_mode] = per_cond

    return result


# --------------------------------------------------------------------
# Phase 3 — Dissociation matrix aggregation
# --------------------------------------------------------------------

def aggregate_summary(results, ablation_sets, positions_modes, conditions):
    """Compute the dissociation matrix: per (ablation_name, positions_mode, condition)
    the counts and rates of baseline / ablated / changed-response classifications.

    The headline NeurIPS metric is `flip_rate_refuse_recovery` for the jb_*
    conditions: fraction of prompts where baseline was COMPLY (i.e. the JB
    worked) and the ablated response is REFUSE — i.e. the ablation recovered
    refusal. For bare, we track the complementary `flip_rate_bare_break` —
    fraction where baseline REFUSED and ablated COMPLIED (positive control
    should have a high value for universal_refusal_core).
    """
    summary: dict = {"per_ablation": {}}
    for abl_name, features in ablation_sets.items():
        if not features:
            continue
        per_abl = {"n_features": len(features), "positions": {}}
        for pos_mode in positions_modes:
            per_pos: dict = {}
            for cond in conditions:
                baseline_refuse = 0
                baseline_comply = 0
                ablated_refuse = 0
                ablated_comply = 0
                changed = 0
                coherent_changed = 0
                recovered_refusal = 0  # comply baseline -> refuse ablated (JB recovery)
                broke_refusal = 0       # refuse baseline -> comply ablated (ctrl/bare break)
                n_seen = 0
                for r in results:
                    bl = r["baseline"].get(cond)
                    ab = r["ablations"].get(abl_name, {}).get(pos_mode, {}).get(cond)
                    if bl is None or ab is None:
                        continue
                    n_seen += 1
                    if bl["cls"] == "REFUSE":
                        baseline_refuse += 1
                    else:
                        baseline_comply += 1
                    if ab["cls"] == "REFUSE":
                        ablated_refuse += 1
                    else:
                        ablated_comply += 1
                    if ab.get("changed_vs_baseline"):
                        changed += 1
                        if ab.get("coherent"):
                            coherent_changed += 1
                        if bl["cls"] == "COMPLY" and ab["cls"] == "REFUSE":
                            recovered_refusal += 1
                        if bl["cls"] == "REFUSE" and ab["cls"] == "COMPLY":
                            broke_refusal += 1
                rates = {
                    "n_seen": n_seen,
                    "n_baseline_refuse": baseline_refuse,
                    "n_baseline_comply": baseline_comply,
                    "n_ablated_refuse": ablated_refuse,
                    "n_ablated_comply": ablated_comply,
                    "n_changed": changed,
                    "n_coherent_changed": coherent_changed,
                    "n_recovered_refusal": recovered_refusal,
                    "n_broke_refusal": broke_refusal,
                    "recovery_rate": round(recovered_refusal / baseline_comply, 3) if baseline_comply else 0.0,
                    "break_rate": round(broke_refusal / baseline_refuse, 3) if baseline_refuse else 0.0,
                }
                per_pos[cond] = rates
            per_abl["positions"][pos_mode] = per_pos
        summary["per_ablation"][abl_name] = per_abl
    return summary


def compute_dissociation_score(summary) -> dict:
    """Per-ablation diagnostic: does the ablation selectively affect one JB class
    more than others? Returns {ablation_name: {positions_mode: {target_class: avg_other_classes_delta}}}.

    For a class-specific ablation (e.g. jb_fiction_specific_vs_ctrl), the
    "target class" fiction should show recovery_rate on jb_fiction >> recovery_rate
    averaged across other jb_* classes. A positive dissociation score means
    selective class-specific patching.
    """
    out: dict = {}
    for abl_name, per_abl in summary["per_ablation"].items():
        # Heuristic target class detection from ablation name
        target_class = None
        for cls in JB_CLASSES:
            if f"jb_{cls}_" in abl_name or abl_name.endswith(f"_{cls}") or cls in abl_name.replace("jb_", "").replace("_specific_vs_ctrl", "").replace("_exclusive", ""):
                target_class = cls
                break
        if target_class is None:
            continue
        per_mode: dict = {}
        for pos_mode, per_cond in per_abl["positions"].items():
            target_cond = f"jb_{target_class}"
            other_conds = [f"jb_{c}" for c in JB_CLASSES if c != target_class]
            if target_cond not in per_cond:
                continue
            target_recovery = per_cond[target_cond]["recovery_rate"]
            other_rates = [per_cond[c]["recovery_rate"] for c in other_conds if c in per_cond]
            other_avg = sum(other_rates) / len(other_rates) if other_rates else 0.0
            per_mode[pos_mode] = {
                "target_class": target_class,
                "target_recovery_rate": target_recovery,
                "other_classes_avg_recovery_rate": round(other_avg, 3),
                "dissociation_delta": round(target_recovery - other_avg, 3),
            }
        out[abl_name] = per_mode
    return out


# --------------------------------------------------------------------
# Figures + report
# --------------------------------------------------------------------

def plot_dissociation_matrix(summary, positions_mode: str, out_path: Path) -> None:
    """Class × ablation heatmap of recovery_rate. The diagonal (class-specific
    ablation ↔ matching jb_class) should be strong; off-diagonal weak."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    abl_names = list(summary["per_ablation"].keys())
    # Rows = jb_* conditions, cols = ablations; value = recovery_rate
    row_conds = [f"jb_{c}" for c in JB_CLASSES] + ["bare"]
    matrix = np.zeros((len(row_conds), len(abl_names)))
    for j, abl in enumerate(abl_names):
        per_cond = summary["per_ablation"][abl]["positions"].get(positions_mode, {})
        for i, cond in enumerate(row_conds):
            if cond == "bare":
                matrix[i, j] = per_cond.get(cond, {}).get("break_rate", 0.0)
            else:
                matrix[i, j] = per_cond.get(cond, {}).get("recovery_rate", 0.0)

    fig, ax = plt.subplots(figsize=(max(8, len(abl_names) * 1.2), 6))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(abl_names)))
    ax.set_xticklabels(abl_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_conds)))
    ax.set_yticklabels(row_conds, fontsize=10)
    for i in range(len(row_conds)):
        for j in range(len(abl_names)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="black" if 0.2 < matrix[i, j] < 0.8 else "white", fontsize=8)
    ax.set_title(f"Stage 08 dissociation matrix ({positions_mode} positions)\n"
                 f"rows=condition, cols=ablation; "
                 f"jb_* rows show recovery_rate (COMPLY→REFUSE), "
                 f"bare row shows break_rate (REFUSE→COMPLY)")
    fig.colorbar(im, ax=ax, label="rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_positions_comparison(summary, out_path: Path) -> None:
    """For each ablation, side-by-side bars of recovery_rate per jb_ class,
    comparing 'all' vs 'anchors' positions modes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    per_abl = summary["per_ablation"]
    abl_names = list(per_abl.keys())
    if not abl_names:
        return
    jb_conds = [f"jb_{c}" for c in JB_CLASSES]

    fig, axes = plt.subplots(1, len(abl_names), figsize=(4 * len(abl_names), 5), sharey=True)
    if len(abl_names) == 1:
        axes = [axes]
    width = 0.38
    x = np.arange(len(jb_conds))
    for ax, abl in zip(axes, abl_names):
        positions = per_abl[abl]["positions"]
        all_rates = [positions.get("all", {}).get(c, {}).get("recovery_rate", 0.0) for c in jb_conds]
        anchor_rates = [positions.get("anchors", {}).get(c, {}).get("recovery_rate", 0.0) for c in jb_conds]
        ax.bar(x - width / 2, all_rates, width, label="all positions", color="#2e7d32")
        ax.bar(x + width / 2, anchor_rates, width, label="anchors [-5,-3,-2]", color="#8e24aa")
        ax.set_xticks(x)
        ax.set_xticklabels(JB_CLASSES, rotation=30, ha="right", fontsize=9)
        ax.set_title(abl, fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("recovery rate (COMPLY→REFUSE)")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Positions-mode comparison: all vs anchors", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_summary_md(summary, dissociation, positions_modes, elapsed_min: float,
                     out_path: Path) -> None:
    lines = [
        "# Stage 08 Subcircuit Ablation — Summary",
        "",
        f"**Method**: zero-ablation of transcoder features via "
        "`ReplacementModel.feature_intervention_generate`.",
        f"**Elapsed**: {elapsed_min:.1f} min.",
        f"**Positions modes**: {', '.join(positions_modes)}.",
        "",
        "## Per-ablation results",
        "",
    ]
    for abl_name, per_abl in summary["per_ablation"].items():
        lines += [
            f"### `{abl_name}` ({per_abl['n_features']} features)",
            "",
        ]
        for pos_mode, per_cond in per_abl["positions"].items():
            lines += [
                f"**Positions: {pos_mode}**",
                "",
                "| Condition | Baseline REFUSE | Baseline COMPLY | Ablated REFUSE | Recovery rate | Break rate |",
                "|---|---|---|---|---|---|",
            ]
            for cond, r in per_cond.items():
                lines.append(
                    f"| `{cond}` | {r['n_baseline_refuse']} | {r['n_baseline_comply']} | "
                    f"{r['n_ablated_refuse']} | {r['recovery_rate']*100:.1f}% | "
                    f"{r['break_rate']*100:.1f}% |"
                )
            lines.append("")
    if dissociation:
        lines += [
            "## Dissociation (class-specific ablations)",
            "",
            "Target class's own JB recovery vs. average across other classes.",
            "Positive `dissociation_delta` = class-selective patching.",
            "",
            "| Ablation | Mode | Target class | Target recovery | Others avg | Δ |",
            "|---|---|---|---|---|---|",
        ]
        for abl_name, per_mode in dissociation.items():
            for pos_mode, rec in per_mode.items():
                lines.append(
                    f"| `{abl_name}` | {pos_mode} | {rec['target_class']} | "
                    f"{rec['target_recovery_rate']*100:.1f}% | "
                    f"{rec['other_classes_avg_recovery_rate']*100:.1f}% | "
                    f"{rec['dissociation_delta']*100:+.1f}pp |"
                )
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = get_stage_dir(run_dir, "08_ablation")

    print("=" * 60)
    print("STAGE 08a: Subcircuit ablation via feature_intervention_generate")
    print("=" * 60)

    # Resolve ablation sets
    ablation_sets = resolve_ablation_sets(args, run_dir)
    for name, feats in ablation_sets.items():
        print(f"  {name:45s} → {len(feats)} features")

    # Positions modes
    positions_modes = (
        list(POSITIONS_MODES) if args.positions == "both" else [args.positions]
    )
    print(f"  positions_modes: {positions_modes}")

    # Load dataset
    rows = load_controlled_dataset(n_prompts=args.max_prompts)
    start = args.prompt_start
    end = args.prompt_end if args.prompt_end is not None else len(rows)
    rows = rows[start:end]
    print(f"  prompts: [{start}, {end}) → {len(rows)} rows")

    # Resolve conditions
    if args.conditions:
        conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    else:
        conditions = list(rows[0]["conditions"].keys()) if rows else []
    print(f"  conditions: {len(conditions)} ({conditions})")

    # Try to reuse Stage 06 baselines
    reused = None
    if not args.skip_baseline:
        reused = try_reuse_stage06_baselines(run_dir)
        if reused:
            print(f"  reusing baselines from 06_causal/: {len(reused)} prompts")

    # Load ReplacementModel
    print("\n  Loading ReplacementModel (this takes a minute)...")
    import torch
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    from circuit_tracer import ReplacementModel
    rm = ReplacementModel.from_pretrained(
        config.MODEL_NAME,
        config.TRANSCODER_PATH,
        dtype=dtype_map[args.dtype],
        backend="nnsight",
        lazy_encoder=False,  # we need encoder for intervention
    )
    tokenizer = rm.tokenizer
    print("  Model ready.")

    # Checkpoint / resume
    ckpt_path = out_dir / "ablation_checkpoint.json"
    if args.resume and ckpt_path.exists():
        ckpt = load_json(ckpt_path)
        results = ckpt.get("results", [])
        done_ids = {r["prompt_id"] for r in results}
        rows_todo = [r for r in rows if r["id"] not in done_ids]
        print(f"\n  [resume] skipping {len(done_ids)} done; {len(rows_todo)} remain")
    else:
        results = []
        rows_todo = rows

    # Main loop
    print("\n[PHASE 2] Ablation generation per prompt...")
    t0 = time.time()
    for i, row in enumerate(rows_todo):
        t_p = time.time()
        result = process_prompt(
            rm, tokenizer, row, ablation_sets, conditions, positions_modes,
            args.max_new_tokens, reused,
        )
        results.append(result)
        p_elapsed = time.time() - t_p
        n_ablations = sum(
            len(result["ablations"].get(abl, {}).get(pos_mode, {}))
            for abl in ablation_sets
            for pos_mode in positions_modes
        )
        print(f"  [{i+1}/{len(rows_todo)}] id={row['id']:3d}  "
              f"({p_elapsed:.0f}s)  ablated_gens={n_ablations}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (i + 1) % args.checkpoint_every == 0:
            save_json({"results": results}, ckpt_path)

    total_elapsed = time.time() - t0

    # Aggregate
    summary = aggregate_summary(results, ablation_sets, positions_modes, conditions)
    dissociation = compute_dissociation_score(summary)
    summary["dissociation"] = dissociation

    # Persist
    final = {
        "metadata": {
            "n_prompts": len(results),
            "ablation_sets": {k: len(v) for k, v in ablation_sets.items()},
            "positions_modes": positions_modes,
            "conditions": conditions,
            "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype,
            "source_run": run_dir.name,
            "elapsed_minutes": round(total_elapsed / 60, 2),
        },
        "results": results,
        "summary": summary,
    }
    save_json(final, out_dir / "ablation_results.json")
    save_json(summary, out_dir / "ablation_summary.json")
    print(f"\n  Saved ablation_results.json + ablation_summary.json")

    # Figures + report
    print("\n  Generating figures + report...")
    for pos_mode in positions_modes:
        plot_dissociation_matrix(summary, pos_mode,
                                 out_dir / f"dissociation_matrix_{pos_mode}.png")
        print(f"    dissociation_matrix_{pos_mode}.png")
    if len(positions_modes) > 1:
        plot_positions_comparison(summary, out_dir / "positions_comparison.png")
        print("    positions_comparison.png")
    write_summary_md(summary, dissociation, positions_modes,
                     total_elapsed / 60, out_dir / "ABLATION_SUMMARY.md")
    print("    ABLATION_SUMMARY.md")

    # Headline numbers
    print("\n" + "=" * 60)
    print("HEADLINE RESULTS")
    print("=" * 60)
    for abl_name, per_abl in summary["per_ablation"].items():
        for pos_mode in positions_modes:
            per_cond = per_abl["positions"].get(pos_mode, {})
            bare = per_cond.get("bare", {})
            jb_rates = [per_cond.get(f"jb_{c}", {}).get("recovery_rate", 0.0) for c in JB_CLASSES]
            jb_avg = sum(jb_rates) / len(jb_rates) if jb_rates else 0.0
            print(f"  {abl_name:45s} [{pos_mode:7s}] "
                  f"bare_break={bare.get('break_rate', 0.0)*100:5.1f}%  "
                  f"jb_recovery_avg={jb_avg*100:5.1f}%")
    if dissociation:
        print("\n  Class-specific dissociation (positive = class-selective):")
        for abl_name, per_mode in dissociation.items():
            for pos_mode, rec in per_mode.items():
                print(f"    {abl_name:45s} [{pos_mode:7s}] "
                      f"target={rec['target_class']:20s} "
                      f"Δ={rec['dissociation_delta']*100:+.1f}pp")
    print(f"\n  Elapsed: {total_elapsed / 60:.1f} min")
    print("DONE!")


if __name__ == "__main__":
    main()
