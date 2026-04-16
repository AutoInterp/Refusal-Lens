"""
Stage 02: Run CLT Attribution Graphs
=====================================
For each prompt × condition (bare + 5 JB classes), compute circuit-tracer
attribution graphs targeting the refusal direction.

Extracts ALL active features per graph (no top-k filtering) and performs
cross-class feature comparison (shared/bare-only/cls-only/sign-flipped/
dampened/amplified-anti).

Inputs:  01_direction/refusal_direction.pt, dataset splits
Outputs: 02_attribution/attribution_results.json, feature_comparison_aggregate.json
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from utils import (
    format_prompt,
    load_experiment_dataset,
    save_json, 
    load_json,
    get_stage_dir,
    create_run_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run CLT attribution graphs")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--n-prompts", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dataset", type=Path, default=None,
        help="Path to curated dataset JSON (optional, falls back to random selection)",
    )
    return parser.parse_args()


def feature_key(layer: int, feat_idx: int) -> str:
    return f"L{layer}:F{feat_idx}"


def extract_all_features(graph) -> dict[str, dict]:
    """Extract ALL features and their attributions from a graph."""
    active = graph.active_features
    selected = graph.selected_features
    n_selected = len(selected)
    adj = graph.adjacency_matrix
    target_row = adj[-1, :n_selected]

    features = {}
    for i in range(n_selected):
        orig_idx = int(selected[i])
        feat = active[orig_idx]
        layer = int(feat[0])
        pos = int(feat[1])
        feat_idx = int(feat[2])
        key = feature_key(layer, feat_idx)
        features[key] = {
            "layer": layer,
            "position": pos,
            "feature_idx": feat_idx,
            "attribution": float(target_row[i]),
            "activation": float(graph.activation_values[orig_idx])
            if orig_idx < len(graph.activation_values)
            else 0.0,
        }
    return features


def graph_summary(graph) -> dict:
    """Compute summary stats from a single attribution graph."""
    adj = graph.adjacency_matrix
    last = adj[-1, :]
    n_feat = graph.active_features.shape[0]
    return {
        "pos_sum": float(last[last > 0].sum().item()),
        "neg_sum": float(last[last < 0].sum().item()),
        "net": float(last.sum().item()),
        "n_features": int(n_feat),
    }


def compare_features(bare_feats: dict, cls_feats: dict) -> dict:
    """Compare feature sets between bare and jailbreak conditions."""
    bare_keys = set(bare_feats.keys())
    cls_keys = set(cls_feats.keys())
    shared = bare_keys & cls_keys
    bare_only = bare_keys - cls_keys
    cls_only = cls_keys - bare_keys

    sign_flipped = []
    dampened = []
    amplified_anti = []

    for key in shared:
        b_attr = bare_feats[key]["attribution"]
        c_attr = cls_feats[key]["attribution"]
        delta = c_attr - b_attr

        if (b_attr > 0 and c_attr < 0) or (b_attr < 0 and c_attr > 0):
            sign_flipped.append({
                "key": key, "bare_attr": round(b_attr, 6), "cls_attr": round(c_attr, 6),
            })
        elif b_attr > 0 and delta < -0.01:
            dampened.append({"key": key, "delta": round(delta, 6)})
        elif b_attr < 0 and delta < -0.01:
            amplified_anti.append({"key": key, "delta": round(delta, 6)})

    return {
        "n_bare": len(bare_keys),
        "n_cls": len(cls_keys),
        "n_shared": len(shared),
        "n_bare_only": len(bare_only),
        "n_cls_only": len(cls_only),
        "n_sign_flipped": len(sign_flipped),
        "n_dampened": len(dampened),
        "n_amplified_anti": len(amplified_anti),
        "top_sign_flipped": sorted(sign_flipped, key=lambda x: abs(x["bare_attr"]), reverse=True)[:10],
        "top_dampened": sorted(dampened, key=lambda x: x["delta"])[:10],
        "top_amplified_anti": sorted(amplified_anti, key=lambda x: x["delta"])[:10],
    }


def main():
    args = parse_args()
    run_dir = args.run_dir
    out_dir = get_stage_dir(run_dir, "02_attribution")

    print("=" * 60)
    print("STAGE 02: Run CLT Attribution Graphs")
    print("=" * 60)

    # Load direction from Stage 01
    dir_path = run_dir / "01_direction" / "refusal_direction.pt"
    if not dir_path.exists():
        # Fallback to pre-computed
        dir_path = (
            config.REPO_ROOT / "data" / "results"
            / "meeting_experiments" / "refusal_direction_corrected.pt"
        )
    print(f"  Loading direction from {dir_path}")
    dir_data = torch.load(dir_path, map_location="cpu", weights_only=False)
    direction = dir_data["best_direction"].to(torch.float32).cuda()

    from circuit_tracer import ReplacementModel, attribute
    from circuit_tracer.attribution.targets import CustomTarget

    target = CustomTarget(token_str="refusal_direction", prob=1.0, vec=direction)

    tokenizer_module = __import__("transformers", fromlist=["AutoTokenizer"])
    tokenizer = tokenizer_module.AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.padding_side = "left"

    print("  Loading ReplacementModel...")
    model = ReplacementModel.from_pretrained(
        config.MODEL_NAME,
        config.TRANSCODER_PATH,
        dtype=torch.float32,
        backend="nnsight",
        lazy_encoder=True,
    )
    print("  Ready.")

    # Load dataset
    prompts = load_experiment_dataset(
        n_prompts=args.n_prompts, dataset_path=args.dataset,
    )
    print(f"  Selected {len(prompts)} diverse harmful prompts")

    # Checkpoint support
    checkpoint_path = out_dir / "attribution_checkpoint.json"
    checkpoint = load_json(checkpoint_path) if (args.resume and checkpoint_path.exists()) else None
    start_idx = checkpoint.get("last_completed", -1) + 1 if checkpoint else 0
    results = checkpoint.get("results", []) if checkpoint else []

    classes = ["bare"] + list(config.JB_CLASSES.keys())
    rng = random.Random(42)

    t0 = time.time()
    for i in range(start_idx, len(prompts)):
        prompt_text = prompts[i]["instruction"]
        print(f"\n  Prompt {i + 1}/{len(prompts)}: {prompt_text[:60]}...")
        torch.cuda.empty_cache()

        row = {"prompt_idx": i, "prompt": prompt_text, "conditions": {}}
        prompt_features = {}

        for cls in classes:
            if cls == "bare":
                input_text = prompt_text
                prefix = ""
            else:
                prefix = rng.choice(config.JB_CLASSES[cls])
                input_text = prefix + prompt_text.lower()

            formatted = format_prompt(tokenizer, input_text)

            try:
                g = attribute(
                    prompt=formatted,
                    model=model,
                    attribution_targets=[target],
                    batch_size=args.batch_size,
                    max_feature_nodes=args.max_features,
                    verbose=False,
                )
                summary = graph_summary(g)
                features = extract_all_features(g)
                summary["prefix"] = prefix
                summary["n_active"] = len(features)

                sorted_feats = sorted(
                    features.items(),
                    key=lambda x: abs(x[1]["attribution"]),
                    reverse=True,
                )[:50]
                summary["top50_features"] = {k: v["attribution"] for k, v in sorted_feats}

                row["conditions"][cls] = summary
                prompt_features[cls] = features

                print(
                    f"    {cls:>15}: net={summary['net']:+.3f}  "
                    f"pos={summary['pos_sum']:.1f}  neg={summary['neg_sum']:.1f}  "
                    f"n={summary['n_features']}"
                )
                del g

            except Exception as e:
                print(f"    {cls:>15}: ERROR — {e}")
                row["conditions"][cls] = {"error": str(e)[:200]}

        # Cross-class feature comparison
        bare_feats = prompt_features.get("bare", {})
        if bare_feats:
            prompt_comparison = {}
            for cls in config.JB_CLASSES:
                cls_feats = prompt_features.get(cls, {})
                if cls_feats:
                    prompt_comparison[cls] = compare_features(bare_feats, cls_feats)
            row["feature_comparison"] = prompt_comparison

        del prompt_features
        results.append(row)

        # Checkpoint
        save_json(
            {"last_completed": i, "results": results,
            "prompts": [p["instruction"] for p in prompts]},
            checkpoint_path,
        )

    elapsed = time.time() - t0
    print(f"\n  Attribution complete: {len(results)} prompts in {elapsed/60:.1f} min")

    # Save final results
    final = {
        "metadata": {
            "n_prompts": len(results),
            "model": config.MODEL_NAME,
            "transcoder": config.TRANSCODER_PATH,
            "measurement_layer": config.BEST_SEPARATION_LAYER,
            "elapsed_minutes": round(elapsed / 60, 1),
        },
        "results": results,
    }
    save_json(final, out_dir / "attribution_results.json")

    # Aggregate feature comparison
    agg = {}
    for cls in config.JB_CLASSES:
        cls_stats = {
            "n_shared": [], "n_bare_only": [], "n_cls_only": [],
            "n_sign_flipped": [], "n_dampened": [], "n_amplified_anti": [],
            "n_bare": [], "n_cls": [],
        }
        for row in results:
            comp = row.get("feature_comparison", {}).get(cls)
            if comp:
                for key in cls_stats:
                    cls_stats[key].append(comp[key])
        if cls_stats["n_shared"]:
            agg[cls] = {
                k: {
                    "mean": round(float(np.mean(v)), 1),
                    "std": round(float(np.std(v)), 1),
                    "min": int(min(v)),
                    "max": int(max(v)),
                }
                for k, v in cls_stats.items()
            }
    save_json(agg, out_dir / "feature_comparison_aggregate.json")

    print(f"  Saved to {out_dir}/")
    print("DONE!")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()