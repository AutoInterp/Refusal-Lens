#!/usr/bin/env python3
"""
Demo: Attribution Graphs to the Refusal Direction
==================================================

End-to-end proof-of-concept showing the Refusal-Lens pipeline:

1. Load contrastive dataset (harmful / harmless prompts)
2. Load a HuggingFace model and compute the refusal direction r-hat
3. Load a circuit-tracer ReplacementModel with transcoders
4. Run attribution to the refusal direction on:
   a. A bare harmful prompt
   b. The same harmful request wrapped in a roleplay jailbreak
5. Extract top features by |A_{s->R}| and compare
6. Save results + optional plots

Target model : google/gemma-2-2b-it  (~5 GB bfloat16)
Transcoders  : gemma (built-in preset for Gemma Scope 2 2B PLTs)
GPU          : RTX 4090 24 GB (should also fit on 12 GB cards)

Usage:
    python scripts/demo_attribution.py [--n-prompts 16] [--layers 12 14 16 18] [--no-plots]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure src/ is importable when running from repo root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("demo")

# ---------------------------------------------------------------------------
# Demo configuration
# ---------------------------------------------------------------------------
DEMO_MODEL_NAME = "google/gemma-2-2b-it"
DEMO_TRANSCODER = "gemma"  # built-in preset for Gemma Scope PLTs
DEMO_BACKEND = "transformerlens"  # gemma-2-2b works with TransformerLens
DEMO_DTYPE = torch.bfloat16
DEMO_LAYERS = [12, 14, 16, 18, 20]  # layers to compute refusal direction
DEMO_MEASUREMENT_LAYER = None  # None = last layer (unembed); set to int for intermediate
DEMO_TOP_K = 20
DEMO_BATCH_SIZE = 256  # for attribution backward passes
DEMO_N_PROMPTS = 16  # number of contrastive pairs for direction computation

from datetime import datetime

_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = REPO_ROOT / "data" / "results" / "demo" / "runs" / _TIMESTAMP


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demo: Attribution to Refusal Direction")
    p.add_argument("--n-prompts", type=int, default=DEMO_N_PROMPTS,
                    help="Number of contrastive pairs for direction computation")
    p.add_argument("--layers", type=int, nargs="+", default=DEMO_LAYERS,
                    help="Layers to compute refusal direction at")
    p.add_argument("--measurement-layer", type=int, default=None,
                    help="Layer for attribution measurement (None=last layer)")
    p.add_argument("--top-k", type=int, default=DEMO_TOP_K,
                    help="Number of top features to extract")
    p.add_argument("--no-plots", action="store_true",
                    help="Skip matplotlib plots")
    p.add_argument("--model", type=str, default=DEMO_MODEL_NAME,
                    help="HuggingFace model name")
    p.add_argument("--transcoder", type=str, default=DEMO_TRANSCODER,
                    help="Transcoder repo/preset")
    p.add_argument("--backend", type=str, default=DEMO_BACKEND,
                    help="Backend: transformerlens or nnsight")
    return p.parse_args()


# ===================================================================
# Step 1: Load contrastive dataset
# ===================================================================
def load_contrastive_prompts(n_prompts: int) -> tuple[list[str], list[str]]:
    """Load harmful and harmless prompts from the dataset splits."""
    from refusal_lens.data_loader import load_split

    log.info("Loading dataset splits...")
    harmful = load_split("harmful_train")
    harmless = load_split("harmless_train")

    harmful_texts = [inst.text for inst in harmful[:n_prompts]]
    harmless_texts = [inst.text for inst in harmless[:n_prompts]]

    log.info("  Loaded %d harmful, %d harmless prompts", len(harmful_texts), len(harmless_texts))
    return harmful_texts, harmless_texts


# ===================================================================
# Step 2: Compute refusal direction
# ===================================================================
def compute_direction(
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    layers: list[int],
    model_name: str,
) -> tuple[torch.Tensor, int, object, object]:
    """Load HF model, gather activations, compute r-hat, return (direction, best_layer, model, tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from refusal_lens.refusal_directions import (
        collect_resid_acts_multipos,
        compute_refusal_directions,
        find_best_layer,
    )

    log.info("Loading HuggingFace model: %s ...", model_name)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=DEMO_DTYPE, device_map="auto",
    )
    # Handle both Gemma 2 (flat config) and Gemma 3 (text_config) architectures
    cfg = getattr(model.config, "text_config", model.config)
    log.info(
        "  Model loaded in %.1fs (%d layers, d_model=%d)",
        time.time() - t0, cfg.num_hidden_layers, cfg.hidden_size,
    )

    log.info("Collecting activations at layers %s ...", layers)
    t0 = time.time()
    harmful_acts = collect_resid_acts_multipos(
        model, tokenizer, harmful_prompts, layers, positions="last",
    )
    harmless_acts = collect_resid_acts_multipos(
        model, tokenizer, harmless_prompts, layers, positions="last",
    )
    log.info("  Activations collected in %.1fs", time.time() - t0)

    log.info("Computing refusal directions (difference-in-means) ...")
    directions = compute_refusal_directions(harmful_acts, harmless_acts, method="mean")
    best_layer, best_result = find_best_layer(directions)
    log.info("  Best layer: %d (separation=%.4f)", best_layer, best_result.separation)

    r_hat = best_result.direction  # (d_model,) unit vector
    log.info("  r_hat shape: %s, norm: %.4f", r_hat.shape, r_hat.norm().item())

    return r_hat, best_layer, model, tokenizer


# ===================================================================
# Step 3: Load ReplacementModel and run attribution
# ===================================================================
def run_attribution(
    prompt: str,
    r_hat: torch.Tensor,
    *,
    model_name: str,
    transcoder: str,
    backend: str,
    measurement_layer: int | None,
    top_k: int,
    batch_size: int,
    label: str = "refusal_direction",
    replacement_model: object | None = None,
) -> tuple[list[dict], object]:
    """Run attribution to the refusal direction on a single prompt.

    Returns (top_features, replacement_model) — the model is returned so
    it can be reused across calls without reloading.
    """
    from circuit_tracer import ReplacementModel

    from refusal_lens.attribution import attribute_to_direction
    from refusal_lens.clt import extract_top_features

    # Load replacement model (once)
    if replacement_model is None:
        log.info("Loading ReplacementModel (%s, backend=%s) ...", model_name, backend)
        t0 = time.time()
        replacement_model = ReplacementModel.from_pretrained(
            model_name,
            transcoder,
            backend=backend,
            dtype=DEMO_DTYPE,
        )
        log.info("  ReplacementModel loaded in %.1fs", time.time() - t0)

    log.info("Running attribution on: %.80s...", prompt)
    t0 = time.time()
    graph = attribute_to_direction(
        prompt=prompt,
        model=replacement_model,
        direction=r_hat.to(replacement_model.device if hasattr(replacement_model, 'device') else 'cuda'),
        measurement_layer=measurement_layer,
        label=label,
        batch_size=batch_size,
        verbose=True,
    )
    log.info("  Attribution completed in %.1fs", time.time() - t0)

    top_features = extract_top_features(graph, top_k=top_k)
    return top_features, replacement_model


# ===================================================================
# Step 4: Compare features across conditions
# ===================================================================
def compare_features(
    harmful_features: list[dict],
    jailbreak_features: list[dict],
) -> dict:
    """Compare top features between harmful and jailbreak conditions."""
    def feature_key(f: dict) -> tuple:
        return (f["layer"], f["feature_idx"])

    harmful_set = {feature_key(f) for f in harmful_features}
    jailbreak_set = {feature_key(f) for f in jailbreak_features}

    shared = harmful_set & jailbreak_set
    harmful_only = harmful_set - jailbreak_set
    jailbreak_only = jailbreak_set - harmful_set

    # Find features with sign-flipped attribution
    harmful_map = {feature_key(f): f for f in harmful_features}
    jailbreak_map = {feature_key(f): f for f in jailbreak_features}

    sign_flipped = []
    for key in shared:
        h_attr = harmful_map[key]["attribution"]
        j_attr = jailbreak_map[key]["attribution"]
        if h_attr * j_attr < 0:  # opposite signs
            sign_flipped.append({
                "layer": key[0],
                "feature_idx": key[1],
                "harmful_attribution": h_attr,
                "jailbreak_attribution": j_attr,
            })

    # Features with strong negative attribution on jailbreak (RP suppression candidates)
    suppression_candidates = [
        f for f in jailbreak_features if f["attribution"] < -0.01
    ]

    return {
        "n_harmful_top": len(harmful_features),
        "n_jailbreak_top": len(jailbreak_features),
        "n_shared": len(shared),
        "n_harmful_only": len(harmful_only),
        "n_jailbreak_only": len(jailbreak_only),
        "shared_features": [list(k) for k in sorted(shared)],
        "harmful_only_features": [list(k) for k in sorted(harmful_only)],
        "jailbreak_only_features": [list(k) for k in sorted(jailbreak_only)],
        "sign_flipped": sign_flipped,
        "suppression_candidates": suppression_candidates,
    }


# ===================================================================
# Step 5: Output results
# ===================================================================
def print_feature_table(features: list[dict], title: str) -> None:
    """Print a formatted table of features."""
    print(f"\n{'=' * 78}")
    print(f"  {title}")
    print(f"{'=' * 78}")
    print(f"  {'Rank':>4}  {'Layer':>5}  {'Pos':>4}  {'Feature':>8}  {'Attribution':>12}  {'Activation':>10}")
    print(f"  {'-' * 4}  {'-' * 5}  {'-' * 4}  {'-' * 8}  {'-' * 12}  {'-' * 10}")
    for i, f in enumerate(features):
        sign = "+" if f["attribution"] >= 0 else ""
        print(
            f"  {i + 1:4d}  {f['layer']:5d}  {f['position']:4d}  "
            f"{f['feature_idx']:8d}  {sign}{f['attribution']:11.6f}  {f['activation']:10.4f}"
        )
    print()


def save_results(
    harmful_features: list[dict],
    jailbreak_features: list[dict],
    comparison: dict,
    metadata: dict,
    output_dir: Path,
) -> None:
    """Save all results as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_json(data: object, name: str) -> None:
        path = output_dir / name
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        log.info("  Saved %s", path)

    save_json(harmful_features, "top_features_harmful.json")
    save_json(jailbreak_features, "top_features_jailbreak.json")
    save_json(comparison, "comparison.json")
    save_json(metadata, "metadata.json")


def make_plots(
    harmful_features: list[dict],
    jailbreak_features: list[dict],
    output_dir: Path,
) -> None:
    """Generate matplotlib plots comparing attribution across conditions."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Side-by-side bar chart of top features ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, features, title in [
        (axes[0], harmful_features, "Bare Harmful Prompt"),
        (axes[1], jailbreak_features, "RP Jailbreak Prompt"),
    ]:
        labels = [f"L{f['layer']}:F{f['feature_idx']}" for f in features[:15]]
        attrs = [f["attribution"] for f in features[:15]]
        colors = ["#d9534f" if a > 0 else "#5bc0de" for a in attrs]

        bars = ax.barh(range(len(labels)), attrs, color=colors)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Attribution A_{s->R}")
        ax.set_title(title)
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.invert_yaxis()

    fig.suptitle("Top Features by Attribution to Refusal Direction", fontsize=14)
    fig.legend(
        [plt.Rectangle((0, 0), 1, 1, color="#d9534f"),
         plt.Rectangle((0, 0), 1, 1, color="#5bc0de")],
        ["Positive (drives refusal)", "Negative (suppresses refusal)"],
        loc="lower center", ncol=2, fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    path = output_dir / "attribution_bar_chart.png"
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("  Saved %s", path)

    # --- Plot 2: Scatter of harmful vs jailbreak attribution for shared features ---
    def feature_key(f: dict) -> tuple:
        return (f["layer"], f["feature_idx"])

    h_map = {feature_key(f): f["attribution"] for f in harmful_features}
    j_map = {feature_key(f): f["attribution"] for f in jailbreak_features}
    shared_keys = set(h_map) & set(j_map)

    if len(shared_keys) >= 3:
        fig, ax = plt.subplots(figsize=(8, 8))
        h_vals = [h_map[k] for k in shared_keys]
        j_vals = [j_map[k] for k in shared_keys]

        ax.scatter(h_vals, j_vals, alpha=0.7, s=50, edgecolors="black", linewidths=0.5)
        lim = max(abs(v) for v in h_vals + j_vals) * 1.1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.plot([-lim, lim], [-lim, lim], "--", color="gray", alpha=0.5, label="y=x")
        ax.axhline(0, color="black", linewidth=0.3)
        ax.axvline(0, color="black", linewidth=0.3)
        ax.set_xlabel("Attribution (harmful prompt)")
        ax.set_ylabel("Attribution (jailbreak prompt)")
        ax.set_title(f"Shared Features: Harmful vs Jailbreak Attribution (n={len(shared_keys)})")
        ax.legend()
        plt.tight_layout()
        path = output_dir / "scatter_harmful_vs_jailbreak.png"
        plt.savefig(path, dpi=150)
        plt.close()
        log.info("  Saved %s", path)
    else:
        log.info("  Not enough shared features for scatter plot (%d)", len(shared_keys))


# ===================================================================
# Main
# ===================================================================
def aggregate_features(
    all_features: list[list[dict]],
    top_k: int = 20,
) -> list[dict]:
    """Aggregate top features across multiple prompts.

    For each unique (layer, feature_idx), compute the mean attribution
    and mean activation across all prompts where it appeared.  Return
    the top-k by mean |attribution|.
    """
    from collections import defaultdict

    accum: dict[tuple, dict] = defaultdict(lambda: {
        "sum_attr": 0.0, "sum_act": 0.0, "count": 0,
        "sum_abs_attr": 0.0, "positions": [],
    })

    for features in all_features:
        for f in features:
            key = (f["layer"], f["feature_idx"])
            accum[key]["sum_attr"] += f["attribution"]
            accum[key]["sum_abs_attr"] += abs(f["attribution"])
            accum[key]["sum_act"] += f["activation"]
            accum[key]["count"] += 1
            accum[key]["positions"].append(f["position"])

    results = []
    for (layer, feat_idx), v in accum.items():
        n = v["count"]
        results.append({
            "layer": layer,
            "feature_idx": feat_idx,
            "position": max(set(v["positions"]), key=v["positions"].count),  # mode
            "attribution": v["sum_attr"] / n,
            "abs_attribution": v["sum_abs_attr"] / n,
            "activation": v["sum_act"] / n,
            "n_prompts": n,
        })

    results.sort(key=lambda r: r["abs_attribution"], reverse=True)
    return results[:top_k]


def main() -> None:
    args = parse_args()
    total_start = time.time()

    # Set up log file in the run directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / "run.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(file_handler)
    log.info("Log file: %s", log_path)

    # Get all matched pairs
    from refusal_lens.prompt_template import PromptTemplateLibrary
    lib = PromptTemplateLibrary()
    pairs = lib.get_matched_pairs()
    n_pairs = len(pairs)

    print("\n" + "=" * 78)
    print("  Refusal-Lens Demo: Attribution Graphs to the Refusal Direction")
    print("  (Multi-prompt aggregated run)")
    print("=" * 78)
    print(f"  Model         : {args.model}")
    print(f"  Transcoder    : {args.transcoder}")
    print(f"  Backend       : {args.backend}")
    print(f"  N dir. prompts: {args.n_prompts}")
    print(f"  N attr. pairs : {n_pairs}")
    print(f"  Layers (dir.) : {args.layers}")
    print(f"  Meas. layer   : {args.measurement_layer or 'last (unembed)'}")
    print(f"  Top-k         : {args.top_k}")
    print(f"  Output        : {OUTPUT_DIR}")
    print("=" * 78 + "\n")

    # Step 1: Load data for direction computation
    harmful_prompts, harmless_prompts = load_contrastive_prompts(args.n_prompts)

    # Step 2: Compute refusal direction
    r_hat, best_layer, hf_model, tokenizer = compute_direction(
        harmful_prompts, harmless_prompts, args.layers, args.model,
    )

    # Free HF model — we'll use the ReplacementModel for attribution
    log.info("Freeing HuggingFace model to save VRAM ...")
    del hf_model
    torch.cuda.empty_cache()

    # Step 3: Attribution on ALL matched pairs
    all_harmful_features: list[list[dict]] = []
    all_jailbreak_features: list[list[dict]] = []
    per_pair_results: list[dict] = []
    rep_model = None

    for i, (jailbreak_prompt, harmful_prompt) in enumerate(pairs):
        print(f"\n{'─' * 78}")
        print(f"  Pair {i + 1}/{n_pairs}")
        print(f"  Harmful  : {harmful_prompt[:90]}...")
        print(f"  Jailbreak: {jailbreak_prompt[:90]}...")
        print(f"{'─' * 78}")

        # Harmful
        h_features, rep_model = run_attribution(
            prompt=harmful_prompt,
            r_hat=r_hat,
            model_name=args.model,
            transcoder=args.transcoder,
            backend=args.backend,
            measurement_layer=args.measurement_layer,
            top_k=args.top_k,
            batch_size=DEMO_BATCH_SIZE,
            label="refusal_direction",
            replacement_model=rep_model,
        )

        # Jailbreak
        j_features, rep_model = run_attribution(
            prompt=jailbreak_prompt,
            r_hat=r_hat,
            model_name=args.model,
            transcoder=args.transcoder,
            backend=args.backend,
            measurement_layer=args.measurement_layer,
            top_k=args.top_k,
            batch_size=DEMO_BATCH_SIZE,
            label="refusal_direction",
            replacement_model=rep_model,
        )

        all_harmful_features.append(h_features)
        all_jailbreak_features.append(j_features)

        # Per-pair summary
        h_total = sum(f["attribution"] for f in h_features)
        j_total = sum(f["attribution"] for f in j_features)
        h_neg = sum(1 for f in h_features if f["attribution"] < 0)
        j_neg = sum(1 for f in j_features if f["attribution"] < 0)
        per_pair_results.append({
            "pair_idx": i,
            "harmful_prompt": harmful_prompt,
            "jailbreak_prompt": jailbreak_prompt[:200],
            "harmful_total_attribution": round(h_total, 4),
            "jailbreak_total_attribution": round(j_total, 4),
            "attribution_reduction": round(h_total - j_total, 4),
            "harmful_n_negative": h_neg,
            "jailbreak_n_negative": j_neg,
        })
        log.info(
            "  Pair %d: harmful_sum=%.3f  jailbreak_sum=%.3f  reduction=%.3f  (neg: %d vs %d)",
            i + 1, h_total, j_total, h_total - j_total, h_neg, j_neg,
        )

    # Step 4: Aggregate across all prompts
    log.info("\nAggregating features across %d pairs ...", n_pairs)
    agg_harmful = aggregate_features(all_harmful_features, top_k=args.top_k)
    agg_jailbreak = aggregate_features(all_jailbreak_features, top_k=args.top_k)
    comparison = compare_features(agg_harmful, agg_jailbreak)

    # Step 5: Output
    print_feature_table(agg_harmful, f"Aggregated Top Features: Bare Harmful ({n_pairs} prompts)")
    print_feature_table(agg_jailbreak, f"Aggregated Top Features: RP Jailbreak ({n_pairs} prompts)")

    # Per-pair attribution summary
    print(f"\n{'=' * 78}")
    print(f"  Per-Pair Attribution Summary (sum of top-{args.top_k} A_{{s->R}})")
    print(f"{'=' * 78}")
    print(f"  {'Pair':>4}  {'Harmful Sum':>12}  {'Jailbreak Sum':>14}  {'Reduction':>10}  {'Neg(H)':>6}  {'Neg(J)':>6}")
    print(f"  {'-' * 4}  {'-' * 12}  {'-' * 14}  {'-' * 10}  {'-' * 6}  {'-' * 6}")
    for pr in per_pair_results:
        print(
            f"  {pr['pair_idx'] + 1:4d}  {pr['harmful_total_attribution']:+12.4f}  "
            f"{pr['jailbreak_total_attribution']:+14.4f}  {pr['attribution_reduction']:+10.4f}  "
            f"{pr['harmful_n_negative']:6d}  {pr['jailbreak_n_negative']:6d}"
        )

    avg_h = sum(pr["harmful_total_attribution"] for pr in per_pair_results) / n_pairs
    avg_j = sum(pr["jailbreak_total_attribution"] for pr in per_pair_results) / n_pairs
    avg_red = sum(pr["attribution_reduction"] for pr in per_pair_results) / n_pairs
    print(f"  {'AVG':>4}  {avg_h:+12.4f}  {avg_j:+14.4f}  {avg_red:+10.4f}")

    print(f"\n{'=' * 78}")
    print("  Aggregated Contrastive Comparison")
    print(f"{'=' * 78}")
    print(f"  Shared features      : {comparison['n_shared']}")
    print(f"  Harmful-only features: {comparison['n_harmful_only']}")
    print(f"  Jailbreak-only       : {comparison['n_jailbreak_only']}")
    print(f"  Sign-flipped         : {len(comparison['sign_flipped'])}")
    print(f"  Suppression candidates (avg A < 0 on jailbreak): {len(comparison['suppression_candidates'])}")

    if comparison["sign_flipped"]:
        print(f"\n  Sign-flipped features (drives refusal on harmful, suppresses on jailbreak):")
        for sf in comparison["sign_flipped"]:
            print(
                f"    L{sf['layer']}:F{sf['feature_idx']}  "
                f"harmful={sf['harmful_attribution']:+.6f}  "
                f"jailbreak={sf['jailbreak_attribution']:+.6f}"
            )

    # Save
    metadata = {
        "model": args.model,
        "transcoder": args.transcoder,
        "backend": args.backend,
        "n_direction_prompts": args.n_prompts,
        "n_attribution_pairs": n_pairs,
        "layers": args.layers,
        "best_layer": best_layer,
        "measurement_layer": args.measurement_layer,
        "top_k": args.top_k,
        "total_time_s": round(time.time() - total_start, 1),
        "avg_harmful_attribution_sum": round(avg_h, 4),
        "avg_jailbreak_attribution_sum": round(avg_j, 4),
        "avg_attribution_reduction": round(avg_red, 4),
        "prompts": [{"harmful": h, "jailbreak": j[:200]} for j, h in pairs],
    }

    log.info("\nSaving results to %s ...", OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def save_json(data: object, name: str) -> None:
        path = OUTPUT_DIR / name
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        log.info("  Saved %s", path)

    save_json(agg_harmful, "top_features_harmful_aggregated.json")
    save_json(agg_jailbreak, "top_features_jailbreak_aggregated.json")
    save_json(comparison, "comparison.json")
    save_json(per_pair_results, "per_pair_results.json")
    save_json(metadata, "metadata.json")

    if not args.no_plots:
        log.info("Generating plots ...")
        make_plots(agg_harmful, agg_jailbreak, OUTPUT_DIR)

        # Additional plot: per-pair attribution reduction
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))
            x = list(range(1, n_pairs + 1))
            h_sums = [pr["harmful_total_attribution"] for pr in per_pair_results]
            j_sums = [pr["jailbreak_total_attribution"] for pr in per_pair_results]

            w = 0.35
            ax.bar([xi - w / 2 for xi in x], h_sums, w, label="Bare Harmful", color="#d9534f", alpha=0.8)
            ax.bar([xi + w / 2 for xi in x], j_sums, w, label="RP Jailbreak", color="#5bc0de", alpha=0.8)
            ax.set_xlabel("Prompt Pair")
            ax.set_ylabel("Sum of Top-k Attribution to R")
            ax.set_title("Total Refusal Attribution: Harmful vs Jailbreak per Pair")
            ax.set_xticks(x)
            ax.legend()
            ax.axhline(0, color="black", linewidth=0.5)
            plt.tight_layout()
            path = OUTPUT_DIR / "per_pair_attribution.png"
            plt.savefig(path, dpi=150)
            plt.close()
            log.info("  Saved %s", path)
        except ImportError:
            pass

    total_time = time.time() - total_start

    # Write summary
    summary_lines = [
        "Refusal-Lens Demo Run Summary (Multi-Prompt)",
        "=" * 50,
        f"Timestamp        : {_TIMESTAMP}",
        f"Model            : {args.model}",
        f"Transcoder       : {args.transcoder}",
        f"Backend          : {args.backend}",
        f"N dir. prompts   : {args.n_prompts}",
        f"N attr. pairs    : {n_pairs}",
        f"Layers (dir.)    : {args.layers}",
        f"Best layer       : {best_layer}",
        f"Meas. layer      : {args.measurement_layer or 'last (unembed)'}",
        f"Top-k            : {args.top_k}",
        f"Total time       : {total_time:.1f}s",
        "",
        "Per-pair results:",
    ]
    for pr in per_pair_results:
        summary_lines.append(
            f"  Pair {pr['pair_idx'] + 1}: harmful_sum={pr['harmful_total_attribution']:+.4f}  "
            f"jailbreak_sum={pr['jailbreak_total_attribution']:+.4f}  "
            f"reduction={pr['attribution_reduction']:+.4f}"
        )
    summary_lines += [
        "",
        f"Average harmful sum    : {avg_h:+.4f}",
        f"Average jailbreak sum  : {avg_j:+.4f}",
        f"Average reduction      : {avg_red:+.4f}",
        "",
        "Aggregated comparison:",
        f"  Shared features       : {comparison['n_shared']}",
        f"  Harmful-only features : {comparison['n_harmful_only']}",
        f"  Jailbreak-only        : {comparison['n_jailbreak_only']}",
        f"  Sign-flipped          : {len(comparison['sign_flipped'])}",
        f"  Suppression candidates: {len(comparison['suppression_candidates'])}",
        "",
        "Output files:",
    ]
    for p in sorted(OUTPUT_DIR.iterdir()):
        summary_lines.append(f"  {p.name}")
    (OUTPUT_DIR / "SUMMARY.txt").write_text("\n".join(summary_lines))

    print(f"\n{'=' * 78}")
    print(f"  Demo complete in {total_time:.1f}s")
    print(f"  Results saved to: {OUTPUT_DIR}")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
