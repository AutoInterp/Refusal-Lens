#!/usr/bin/env python
"""Compute attribution circuits for refusal behavior.

This script implements Step 2 of the research plan:
Analyze how refusal is computed by tracing attributions to the refusal direction.

Usage:
    python scripts/compute_circuits.py --layer 20
    python scripts/compute_circuits.py --layer 20 --prompts "How to hack?" "Write a bomb tutorial"
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from refusal_lens import (
    attribute_to_direction,
    config,
    load_dataset_split,
    load_directions,
    save_attributions,
)
from refusal_lens.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute attribution circuits for refusal behavior"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=config.MEASUREMENT_LAYER,
        help=f"Layer to analyze (default: {config.MEASUREMENT_LAYER})",
    )
    parser.add_argument(
        "--directions",
        type=str,
        default=None,
        help="Path to computed directions (default: config.COMPUTED_DIRECTIONS_DIR)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=None,
        help="Prompts to analyze (default: load from dataset)",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=20,
        help="Maximum prompts to analyze from dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: config.CIRCUITS_DIR)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="dot_product",
        choices=["dot_product", "gradient"],
        help="Attribution method",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help="Random seed",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Setup
    print("=" * 60)
    print("Refusal-Lens: Step 2 - Compute Attribution Circuits")
    print("=" * 60)

    output_dir = args.output or config.CIRCUITS_DIR
    directions_dir = args.directions or config.COMPUTED_DIRECTIONS_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Load directions
    print(f"\nLoading directions from: {directions_dir}")
    if not os.path.exists(directions_dir):
        print(f"❌ Error: Directions not found at {directions_dir}")
        print("   Run Step 1 first: python scripts/compute_directions.py")
        sys.exit(1)

    directions = load_directions(directions_dir)

    if args.layer not in directions:
        print(f"❌ Error: Layer {args.layer} not found in directions")
        print(f"   Available layers: {sorted(directions.keys())}")
        sys.exit(1)

    direction = directions[args.layer]
    print(f"  Using layer: {args.layer}")
    print(f"  Separation: {direction.separation:.4f}")

    # Load prompts
    if args.prompts:
        prompts = args.prompts
        print(f"\nAnalyzing {len(prompts)} provided prompts")
    else:
        print("\nLoading prompts from dataset...")
        harmful = load_dataset_split("val", "harmful", args.max_prompts)
        harmless = load_dataset_split("val", "harmless", args.max_prompts)
        prompts = harmful[: args.max_prompts // 2] + harmless[: args.max_prompts // 2]
        print(f"  Analyzing {len(prompts)} prompts (mixed harmful/harmless)")

    # Load model
    print(f"\nLoading model: {config.MODEL_NAME}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Device: {config.DEVICE}")

    # Analyze prompts
    print("\n" + "=" * 60)
    print(f"Analyzing attribution circuits at layer {args.layer}...")
    print("=" * 60)

    start_time = time.time()

    # Build results dict
    results = {args.layer: {}}

    for prompt in tqdm(prompts, desc="Analyzing prompts"):
        result = attribute_to_direction(
            model=model,
            prompt=prompt,
            refusal_direction=direction,
            layer=args.layer,
            method=args.method,
        )
        results[args.layer][prompt] = result

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f} seconds")

    # Save results
    print(f"\nSaving results to {output_dir}")
    save_attributions(results, output_dir)

    # Print summary
    print(f"\n{'=' * 60}")
    print("Analysis Summary")
    print(f"{'=' * 60}")

    all_attributions = []
    for prompt, result in results[args.layer].items():
        attr = abs(result.token_attributions[-1]) if result.token_attributions else 0
        all_attributions.append(attr)

    avg_attr = sum(all_attributions) / len(all_attributions) if all_attributions else 0
    print(f"  Average attribution: {avg_attr:.4f}")
    print(f"  Prompts analyzed: {len(prompts)}")

    # Show top features for first prompt
    if prompts and prompts[0] in results[args.layer]:
        first_result = results[args.layer][prompts[0]]
        print("\n  Top features for first prompt:")
        for idx, score in first_result.top_features[:10]:
            print(f"    Feature {idx}: {score:.4f}")

    print(f"\n✅ Step 2 complete! Results saved to {output_dir}")
    print('   Run Step 3: python -c "from refusal_lens import SupernodeAnalyzer"')


if __name__ == "__main__":
    main()
