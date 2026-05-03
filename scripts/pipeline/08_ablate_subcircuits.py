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
                   help="Run directory containing 07_subcircuits/<subcircuits-file>")
    p.add_argument("--subcircuits-file", type=str, default="subcircuits.json",
                   help=("Filename in 07_subcircuits/ to load subcircuit "
                         "definitions from. Use one of: subcircuits.json "
                         "(legacy / corpus-aggregated, default), "
                         "subcircuits_k50_f50.json, subcircuits_k20_f50.json, "
                         "subcircuits_k100_f20.json (per-prompt sweeps)."))
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
    p.add_argument(
        "--low-coverage-threshold", type=float, default=0.30,
        help="If <threshold of an ablation's features are in this prompt's "
             "top-K attribution (Stage 02 top_features), flag the ablation "
             "as 'low_coverage' for this prompt — explains why generations "
             "may not change behavior even when they're being intervened on.",
    )
    p.add_argument(
        "--ablate-with-mean", action="store_true",
        help="(Future work — not yet supported) Ablate features to their "
             "dataset-mean activation rather than 0. Currently raises if set; "
             "implementation requires a pre-pass that collects mean "
             "activations per feature.",
    )
    return p.parse_args()


# --------------------------------------------------------------------
# Ablation set resolution (subcircuits + manual cart)
# --------------------------------------------------------------------

def resolve_ablation_sets(args, run_dir: Path) -> dict[str, list[tuple[int, int]]]:
    """Build the mapping {ablation_name → [(layer, feat_idx), ...]} for this run.

    Two sources that can be combined:
      - --subcircuits: named subcircuit sets from Stage 07's --subcircuits-file
      - --feature-file: a single cart.json (produces one ablation condition)
    """
    out: dict[str, list[tuple[int, int]]] = {}
    sub_json = run_dir / "07_subcircuits" / args.subcircuits_file

    names = args.subcircuits.split(",") if args.subcircuits else list(config.STAGE_08_DEFAULT_SUBCIRCUITS)
    if args.subcircuits != "" and not args.feature_file or args.subcircuits:
        if not sub_json.exists():
            raise FileNotFoundError(
                f"{sub_json} missing — run Stage 07 first (with the matching "
                f"sweep config) or use --feature-file only."
            )
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
# Per-prompt feature index — for the low-coverage diagnostic
# --------------------------------------------------------------------

def build_per_prompt_top_index(
    run_dir: Path, mode: str = "multi",
) -> dict[int, dict[str, dict[str, float]]]:
    """Read Stage 02 attribution and return per-prompt top features per
    condition. Returns ``{prompt_id: {cond_name: {feature_key: |attr|}}}``.

    Used by `compute_coverage` to assess what fraction of an ablation's
    features actually fired strongly in a specific (prompt, condition).
    """
    attr_path = run_dir / "02_attribution" / "attribution_results.json"
    if not attr_path.exists():
        return {}
    raw = load_json(attr_path)
    rows = raw["results"] if isinstance(raw, dict) else raw
    out: dict[int, dict[str, dict[str, float]]] = {}
    for row in rows:
        pid = row.get("prompt_id")
        if pid is None:
            continue
        per_cond: dict[str, dict[str, float]] = {}
        conds = row.get("conditions", {}) or {}
        for cond_name, cond in conds.items():
            if not isinstance(cond, dict) or "error" in cond:
                continue
            graph = cond.get("graphs", {}).get(mode)
            if not isinstance(graph, dict) or "error" in graph:
                continue
            top = graph.get("top_features") or graph.get("top50_features") or {}
            per_cond[cond_name] = {k: abs(float(v)) for k, v in top.items()}
        out[pid] = per_cond
    return out


def compute_coverage(
    feature_keys: list[str],
    prompt_top: dict[str, dict[str, float]],
    cond: str,
) -> dict:
    """Return the per-(prompt, condition) coverage diagnostic for an ablation.

    Reports:
      - n_in_top_k:           how many ablation features are in this prompt's top-K
      - frac_in_top_k:        n_in_top_k / len(feature_keys)
      - sum_abs_attribution:  Σ |attribution| of those features for this prompt
      - low_coverage:         True if frac_in_top_k < args.low_coverage_threshold
                              (the caller decides; we don't read args here, just
                              return the raw fraction for the caller to threshold)
    """
    cond_top = prompt_top.get(cond, {})
    in_top = [fk for fk in feature_keys if fk in cond_top]
    n_in = len(in_top)
    n_total = len(feature_keys) if feature_keys else 1
    return {
        "n_features": len(feature_keys),
        "n_in_top_k": n_in,
        "frac_in_top_k": round(n_in / n_total, 4),
        "sum_abs_attribution": round(
            sum(cond_top.get(fk, 0.0) for fk in feature_keys), 6,
        ),
    }


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
    per_prompt_top: dict | None = None, low_coverage_threshold: float = 0.30,
):
    """Run baselines + ablation conditions for one prompt. Returns a result row.

    Ablation is applied at layer-specific source positions; since features in
    an ablation set may span multiple source layers, we build one combined
    intervention list per (ablation_name, positions_mode) and run a single
    generation per condition.

    `per_prompt_top` (optional): output of `build_per_prompt_top_index`. If
    provided, attaches per-(ablation, condition) coverage diagnostics for this
    prompt (n_in_top_k, frac_in_top_k, sum_abs_attribution, low_coverage flag).
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

    prompt_top = (per_prompt_top or {}).get(row["id"], {})

    # ---- Ablation conditions ----
    for abl_name, features in ablation_sets.items():
        if not features:
            # Skip empty ablation sets (e.g. layer-filtered to nothing)
            continue
        feat_keys = [f"L{L}:F{F}" for (L, F) in features]
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
                # Per-prompt coverage diagnostic — explains why a 0% recovery
                # rate happens (low coverage = ablation features weren't
                # firing strongly on this prompt to begin with).
                if prompt_top:
                    cov = compute_coverage(feat_keys, prompt_top, cond)
                    cov["low_coverage"] = (
                        cov["frac_in_top_k"] < low_coverage_threshold
                    )
                    per_cond[cond]["coverage"] = cov
            result["ablations"][abl_name][pos_mode] = per_cond

    return result


def aggregate_weighted_summary(summary: dict) -> dict:
    """Add comply-weighted aggregates across all 5 JB classes (NeurIPS rigor).

    For each (ablation, positions_mode), reports:
      - weighted_recovery_rate: Σ(per-class recovery × n_baseline_comply)
                                / Σ(n_baseline_comply across jb_*)
        Reflects the model's behavior on JB-success cases proportional to
        how often each class succeeded.
      - weighted_break_rate:    same construction over bare + ctrl_*
                                using n_baseline_refuse as the weight.
      - per_class_jb:           {cls: {recovery_rate, comply_baseline,
                                       low_coverage_frac (if avail)}}

    Adds a `weighted` block to each `summary['per_ablation'][abl]['positions'][mode]`.
    Does not mutate per-class entries.
    """
    for abl_name, per_abl in summary.get("per_ablation", {}).items():
        for pos_mode, per_cond in per_abl.get("positions", {}).items():
            jb_total_comply = 0
            jb_weighted_recovery_num = 0.0
            per_cls: dict = {}
            for cls in JB_CLASSES:
                cond = f"jb_{cls}"
                rec = per_cond.get(cond)
                if not rec:
                    continue
                cb = rec.get("n_baseline_comply", 0)
                rate = rec.get("recovery_rate", 0.0)
                jb_total_comply += cb
                jb_weighted_recovery_num += rate * cb
                per_cls[cls] = {
                    "recovery_rate": rate,
                    "n_baseline_comply": cb,
                    "n_recovered_refusal": rec.get("n_recovered_refusal", 0),
                }

            ctrl_total_refuse = 0
            ctrl_weighted_break_num = 0.0
            for cls in JB_CLASSES:
                cond = f"ctrl_{cls}"
                rec = per_cond.get(cond)
                if not rec:
                    continue
                rb = rec.get("n_baseline_refuse", 0)
                rate = rec.get("break_rate", 0.0)
                ctrl_total_refuse += rb
                ctrl_weighted_break_num += rate * rb
            bare_rec = per_cond.get("bare", {})
            bare_rb = bare_rec.get("n_baseline_refuse", 0)
            bare_break = bare_rec.get("break_rate", 0.0)

            per_cond["weighted"] = {
                "jb_weighted_recovery_rate": (
                    round(jb_weighted_recovery_num / jb_total_comply, 4)
                    if jb_total_comply else 0.0
                ),
                "jb_total_baseline_comply": jb_total_comply,
                "ctrl_weighted_break_rate": (
                    round(ctrl_weighted_break_num / ctrl_total_refuse, 4)
                    if ctrl_total_refuse else 0.0
                ),
                "ctrl_total_baseline_refuse": ctrl_total_refuse,
                "bare_break_rate": bare_break,
                "bare_baseline_refuse": bare_rb,
                "per_class_jb": per_cls,
            }
    return summary


def aggregate_coverage_summary(results: list[dict]) -> dict:
    """Aggregate per-prompt coverage diagnostics across results.

    Returns ``{ablation_name: {positions_mode: {condition:
        {mean_frac_in_top_k, mean_sum_abs_attribution, n_low_coverage}}}}``.

    Used to flag ablations where most prompts had low coverage — explains
    null recovery rates (the features weren't firing strongly on those
    prompts to begin with).
    """
    out: dict = {}
    for r in results:
        for abl_name, per_pos in r.get("ablations", {}).items():
            for pos_mode, per_cond in per_pos.items():
                if pos_mode == "n_features":
                    continue
                if not isinstance(per_cond, dict):
                    continue
                for cond, rec in per_cond.items():
                    cov = rec.get("coverage")
                    if cov is None:
                        continue
                    bucket = (
                        out.setdefault(abl_name, {})
                           .setdefault(pos_mode, {})
                           .setdefault(cond, {
                               "_frac_sum": 0.0, "_attr_sum": 0.0,
                               "_n_low": 0, "_n": 0,
                           })
                    )
                    bucket["_frac_sum"] += cov.get("frac_in_top_k", 0.0)
                    bucket["_attr_sum"] += cov.get("sum_abs_attribution", 0.0)
                    bucket["_n_low"] += int(bool(cov.get("low_coverage")))
                    bucket["_n"] += 1
    # Finalize: emit means + counts, drop the underscore tallies.
    final: dict = {}
    for abl, per_pos in out.items():
        final[abl] = {}
        for pos_mode, per_cond in per_pos.items():
            final[abl][pos_mode] = {}
            for cond, b in per_cond.items():
                n = max(1, b["_n"])
                final[abl][pos_mode][cond] = {
                    "n_prompts": b["_n"],
                    "mean_frac_in_top_k": round(b["_frac_sum"] / n, 4),
                    "mean_sum_abs_attribution": round(b["_attr_sum"] / n, 6),
                    "n_low_coverage_prompts": b["_n_low"],
                    "frac_low_coverage": round(b["_n_low"] / n, 4),
                }
    return final


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


# --------------------------------------------------------------------
# Phase 4 — activation audit (uses Stage 02 attribution data, no GPU)
# --------------------------------------------------------------------

def audit_activations(run_dir: Path, ablation_sets: dict[str, list[tuple[int, int]]]) -> dict:
    """For each ablation set, summarize how often its features actually appear
    in the corpus top-50 across each condition class — a proxy for "is the
    feature firing during inference?".

    The Stage 07 ctrl-aware rules are corpus-aggregated (a feature is in
    `bare_top50` iff it's in the corpus-level top-50 for bare). But a feature
    can fire on individual JB prompts without aggregating into the JB corpus
    top-50. This audit measures per-feature, per-prompt top-50 hit rate so
    we can interpret recovery/break rates correctly:

      - high jb_* hit rate on a "ctrl_shared_refusal" feature → the
        Stage 07 rule's separation isn't clean at the per-prompt level
      - high target-class hit rate vs other JB classes on a class-specific
        ablation → correlational dissociation is in place; expect causal
        dissociation at scale.

    Reads `<run_dir>/02_attribution/attribution_results.json`. Skips silently
    (returns empty dict) if Stage 02 output isn't available.
    """
    attr_path = run_dir / "02_attribution" / "attribution_results.json"
    if not attr_path.exists():
        return {}

    attr = load_json(attr_path)

    feature_appearances: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    feature_attr_sum: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    total_per_cond: dict[str, int] = defaultdict(int)

    for row in attr.get("results", []):
        for cond_name, cond in row.get("conditions", {}).items():
            total_per_cond[cond_name] += 1
            top50 = cond.get("graphs", {}).get("multi", {}).get("top50_features", {}) or {}
            for fk, attribution in top50.items():
                feature_appearances[fk][cond_name] += 1
                feature_attr_sum[fk][cond_name] += abs(float(attribution))

    class_groups = {
        "bare": ["bare"],
        "jb_*": sorted(c for c in total_per_cond if c.startswith("jb_")),
        "ctrl_*": sorted(c for c in total_per_cond if c.startswith("ctrl_")),
    }

    out: dict = {
        "source": str(attr_path.relative_to(run_dir.parent.parent)) if run_dir.parent.parent in attr_path.parents else str(attr_path),
        "n_prompts_per_condition": {c: total_per_cond[c] for c in sorted(total_per_cond)},
        "per_ablation": {},
    }

    for abl_name, features in ablation_sets.items():
        if not features:
            continue
        feat_keys = [f"L{L}:F{F}" for (L, F) in features]
        per_class: dict = {}
        for grp_name, conds in class_groups.items():
            denom = sum(total_per_cond[c] for c in conds)
            if not denom:
                continue
            hits_total = 0.0
            attr_total = 0.0
            n_zero = 0
            for fk in feat_keys:
                hits = sum(feature_appearances[fk].get(c, 0) for c in conds)
                attr_sum = sum(feature_attr_sum[fk].get(c, 0.0) for c in conds)
                hits_total += hits / denom
                attr_total += attr_sum / denom
                if hits == 0:
                    n_zero += 1
            per_class[grp_name] = {
                "mean_top50_hit_rate": round(hits_total / len(feat_keys), 4),
                "mean_attr_per_prompt": round(attr_total / len(feat_keys), 5),
                "n_features_never_in_top50": n_zero,
                "n_features": len(feat_keys),
            }

        # Per-jb-class breakdown (for class-specific dissociation diagnostic)
        per_jb_class: dict = {}
        for c in class_groups["jb_*"]:
            denom = total_per_cond[c]
            if not denom:
                continue
            rates = [feature_appearances[fk].get(c, 0) / denom for fk in feat_keys]
            attrs = [feature_attr_sum[fk].get(c, 0.0) / denom for fk in feat_keys]
            per_jb_class[c] = {
                "mean_top50_hit_rate": round(sum(rates) / len(rates), 4),
                "mean_attr_per_prompt": round(sum(attrs) / len(attrs), 5),
            }

        out["per_ablation"][abl_name] = {
            "n_features": len(feat_keys),
            "by_class_group": per_class,
            "by_jb_class": per_jb_class,
        }

    return out


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
                     out_path: Path, activation_audit: dict | None = None,
                     subcircuits_file: str = "subcircuits.json",
                     low_coverage_threshold: float = 0.30,
                     coverage_summary: dict | None = None) -> None:
    lines = [
        "# Stage 08 Subcircuit Ablation — Summary",
        "",
        f"**Method**: zero-ablation of transcoder features via "
        "`ReplacementModel.feature_intervention_generate`.",
        f"**Subcircuits source**: `{subcircuits_file}`.",
        f"**Elapsed**: {elapsed_min:.1f} min.",
        f"**Positions modes**: {', '.join(positions_modes)}.",
        f"**Low-coverage threshold**: {low_coverage_threshold:.2f} "
        "(prompts where <threshold of ablation features are in top-K are flagged).",
        "",
        "## How to read these numbers",
        "",
        "- `recovery_rate` = baseline COMPLY → ablated REFUSE. "
        "`break_rate` = baseline REFUSE → ablated COMPLY.",
        "- **Per-class** rows show the unweighted rate for each condition.",
        "- **Comply-weighted JB recovery** (under each position mode) is "
        "Σ(per-class rate × per-class baseline_comply) / Σ(baseline_comply). "
        "Reflects the model's behavior on actual JB-success cases without "
        "dropping rare classes — the headline NeurIPS rigor metric.",
        "- The **per-prompt coverage** table flags prompts where the ablation "
        "features weren't in that prompt's top-K attribution. Low coverage on "
        "a class explains null recovery rates: the features couldn't be doing "
        "much because they weren't strongly active to begin with.",
        f"- For `{subcircuits_file}` runs (per-prompt sweep), the subcircuits "
        "are constructed from features in top-K for ≥F fraction of prompts in "
        "each condition; legacy `subcircuits.json` uses corpus union.",
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
                if cond == "weighted":
                    continue  # rendered separately below
                lines.append(
                    f"| `{cond}` | {r['n_baseline_refuse']} | {r['n_baseline_comply']} | "
                    f"{r['n_ablated_refuse']} | {r['recovery_rate']*100:.1f}% | "
                    f"{r['break_rate']*100:.1f}% |"
                )
            w = per_cond.get("weighted")
            if w:
                lines += [
                    "",
                    "**Comply-weighted aggregates:**",
                    "",
                    f"- JB recovery (weighted by baseline_comply across all 5 JB classes): "
                    f"**{w['jb_weighted_recovery_rate']*100:.1f}%** "
                    f"(n_jb_comply={w['jb_total_baseline_comply']})",
                    f"- ctrl break (weighted by baseline_refuse across all 5 ctrl classes): "
                    f"**{w['ctrl_weighted_break_rate']*100:.1f}%** "
                    f"(n_ctrl_refuse={w['ctrl_total_baseline_refuse']})",
                    f"- bare break: **{w['bare_break_rate']*100:.1f}%** "
                    f"(n_bare_refuse={w['bare_baseline_refuse']})",
                ]
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

    if coverage_summary:
        lines += [
            "## Per-prompt coverage diagnostic",
            "",
            "Mean fraction of ablation features in each prompt's top-K attribution, "
            f"plus the count of low-coverage prompts (frac < {low_coverage_threshold:.2f}).",
            "Low coverage means the features couldn't have a strong effect; null "
            "recovery on those (ablation, condition) pairs is uninformative.",
            "",
        ]
        for abl_name, per_pos in coverage_summary.items():
            lines += [f"### `{abl_name}`", ""]
            for pos_mode in positions_modes:
                per_cond = per_pos.get(pos_mode, {})
                if not per_cond:
                    continue
                lines += [
                    f"**Positions: {pos_mode}**",
                    "",
                    "| Condition | Mean frac in top-K | Mean Σ\\|attr\\| | Low-coverage prompts |",
                    "|---|---|---|---|",
                ]
                for cond, c in per_cond.items():
                    lines.append(
                        f"| `{cond}` | {c['mean_frac_in_top_k']*100:.1f}% | "
                        f"{c['mean_sum_abs_attribution']:.4f} | "
                        f"{c['n_low_coverage_prompts']}/{c['n_prompts']} "
                        f"({c['frac_low_coverage']*100:.0f}%) |"
                    )
                lines.append("")

    if activation_audit and activation_audit.get("per_ablation"):
        lines += [
            "## Activation audit (Stage 02 attribution data)",
            "",
            "Per-ablation, per-condition-class top-50 hit rate and mean |attribution|. "
            "Diagnoses whether the Stage 07 set logic produces a clean per-prompt "
            "separation, and whether class-specific subcircuits are correlationally "
            "selective for their target class.",
            "",
        ]
        for abl_name, audit in activation_audit["per_ablation"].items():
            lines += [
                f"### `{abl_name}` ({audit['n_features']} features) — corpus-level activity",
                "",
                "| Condition class | Mean top-50 hit rate | Mean \\|attr\\|/prompt | Features never in top-50 |",
                "|---|---|---|---|",
            ]
            for grp, stats in audit["by_class_group"].items():
                lines.append(
                    f"| `{grp}` | {stats['mean_top50_hit_rate']*100:.2f}% | "
                    f"{stats['mean_attr_per_prompt']:.5f} | "
                    f"{stats['n_features_never_in_top50']}/{stats['n_features']} |"
                )
            lines.append("")
            if audit.get("by_jb_class"):
                lines += [
                    "Per-JB-class breakdown (selectivity check):",
                    "",
                    "| JB class | Mean top-50 hit rate | Mean \\|attr\\|/prompt |",
                    "|---|---|---|",
                ]
                for c, stats in audit["by_jb_class"].items():
                    lines.append(
                        f"| `{c}` | {stats['mean_top50_hit_rate']*100:.2f}% | "
                        f"{stats['mean_attr_per_prompt']:.5f} |"
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

    if args.ablate_with_mean:
        raise NotImplementedError(
            "--ablate-with-mean requires a separate pre-pass to collect "
            "per-feature mean activations across the dataset. Not yet "
            "implemented — use zero-ablation (omit this flag) for now."
        )

    print("=" * 60)
    print("STAGE 08a: Subcircuit ablation via feature_intervention_generate")
    print(f"  subcircuits-file: {args.subcircuits_file}")
    print(f"  low-coverage threshold: {args.low_coverage_threshold:.2f}")
    print("=" * 60)

    # Resolve ablation sets
    ablation_sets = resolve_ablation_sets(args, run_dir)
    for name, feats in ablation_sets.items():
        print(f"  {name:45s} → {len(feats)} features")

    # Per-prompt top-feature index for the coverage diagnostic
    print("\n  Building per-prompt top-feature index from Stage 02...")
    per_prompt_top = build_per_prompt_top_index(run_dir, mode="multi")
    print(
        f"    Indexed {len(per_prompt_top)} prompts "
        f"({'per-prompt coverage diagnostic enabled' if per_prompt_top else 'no Stage 02 data — diagnostic disabled'})"
    )

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
            per_prompt_top=per_prompt_top,
            low_coverage_threshold=args.low_coverage_threshold,
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
    # Comply-weighted aggregates across all 5 JB classes (NeurIPS rigor)
    summary = aggregate_weighted_summary(summary)
    # Per-prompt coverage rollup (low-coverage diagnostic)
    coverage_summary = aggregate_coverage_summary(results)
    summary["coverage"] = coverage_summary

    # Phase 4 — activation audit from Stage 02 attribution (no GPU; CPU-only)
    print("\n[PHASE 4] Activation audit from Stage 02 attribution data...")
    activation_audit = audit_activations(run_dir, ablation_sets)
    if activation_audit:
        save_json(activation_audit, out_dir / "activation_audit.json")
        print(f"  Saved activation_audit.json — covers {len(activation_audit.get('per_ablation', {}))} ablation sets")
    else:
        print("  Skipped (no 02_attribution/attribution_results.json found)")

    # Persist
    final = {
        "metadata": {
            "n_prompts": len(results),
            "ablation_sets": {k: len(v) for k, v in ablation_sets.items()},
            "subcircuits_file": args.subcircuits_file,
            "low_coverage_threshold": args.low_coverage_threshold,
            "ablate_with_mean": args.ablate_with_mean,
            "positions_modes": positions_modes,
            "conditions": conditions,
            "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype,
            "source_run": run_dir.name,
            "elapsed_minutes": round(total_elapsed / 60, 2),
        },
        "results": results,
        "summary": summary,
        "activation_audit": activation_audit,
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
                     total_elapsed / 60, out_dir / "ABLATION_SUMMARY.md",
                     activation_audit=activation_audit,
                     subcircuits_file=args.subcircuits_file,
                     low_coverage_threshold=args.low_coverage_threshold,
                     coverage_summary=coverage_summary)
    print("    ABLATION_SUMMARY.md")

    # Headline numbers
    print("\n" + "=" * 60)
    print("HEADLINE RESULTS")
    print("=" * 60)
    print(f"  subcircuits source: {args.subcircuits_file}")
    for abl_name, per_abl in summary["per_ablation"].items():
        for pos_mode in positions_modes:
            per_cond = per_abl["positions"].get(pos_mode, {})
            w = per_cond.get("weighted", {})
            jb_rates = [per_cond.get(f"jb_{c}", {}).get("recovery_rate", 0.0) for c in JB_CLASSES]
            jb_avg = sum(jb_rates) / len(jb_rates) if jb_rates else 0.0
            print(
                f"  {abl_name:45s} [{pos_mode:7s}] "
                f"bare_break={w.get('bare_break_rate', 0.0)*100:5.1f}%  "
                f"jb_recovery_avg={jb_avg*100:5.1f}%  "
                f"jb_recovery_weighted={w.get('jb_weighted_recovery_rate', 0.0)*100:5.1f}%  "
                f"(n_jb_comply={w.get('jb_total_baseline_comply', 0)})"
            )
    if dissociation:
        print("\n  Class-specific dissociation (positive = class-selective):")
        for abl_name, per_mode in dissociation.items():
            for pos_mode, rec in per_mode.items():
                print(f"    {abl_name:45s} [{pos_mode:7s}] "
                      f"target={rec['target_class']:20s} "
                      f"Δ={rec['dissociation_delta']*100:+.1f}pp")
    if coverage_summary:
        print("\n  Coverage diagnostic (mean fraction of ablation features in prompt's top-K):")
        for abl_name in coverage_summary:
            for pos_mode in positions_modes:
                per_cond = coverage_summary[abl_name].get(pos_mode, {})
                low_total = sum(c["n_low_coverage_prompts"] for c in per_cond.values())
                n_total = sum(c["n_prompts"] for c in per_cond.values())
                if n_total:
                    print(
                        f"    {abl_name:45s} [{pos_mode:7s}] "
                        f"low_coverage_prompts={low_total}/{n_total} "
                        f"({100*low_total/n_total:5.1f}%)"
                    )
    print(f"\n  Elapsed: {total_elapsed / 60:.1f} min")
    print("DONE!")


if __name__ == "__main__":
    main()
