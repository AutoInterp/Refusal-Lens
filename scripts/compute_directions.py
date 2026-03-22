#!/usr/bin/env python
"""Compute refusal directions for all layers.

This script implements Step 1 of the research plan:
Compute r_ℓ = E[x_ℓ | harmful] - E[x_ℓ | harmless]

Usage:
    python scripts/compute_directions.py
    python scripts/compute_directions.py --harmful dataset/harmful_train.json --harmless dataset/harmless_train.json
    python scripts/compute_directions.py --layers 10 12 14 16 18 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from refusal_lens import (
    RefusalDirectionComputer,
    config,
    find_best_layer,
    load_dataset_split,
    save_directions,
)
from refusal_lens.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute refusal directions for each layer"
    )
    parser.add_argument(
        "--harmful",
        type=str,
        default=None,
        help="Path to harmful prompts JSON file (default: config.DATASET_HARMFUL_TRAIN)",
    )
    parser.add_argument(
        "--harmless",
        type=str,
        default=None,
        help="Path to harmless prompts JSON file (default: config.DATASET_HARMLESS_TRAIN)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Layers to compute (default: config.REFUSAL_LAYERS)",
    )
    parser.add_argument(
        "--position",
        type=str,
        default="last_prompt_token",
        choices=["last_prompt_token", "first_token"],
        help="Position to extract activations from",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Maximum prompts per class (for testing)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: config.COMPUTED_DIRECTIONS_DIR)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help="Random seed",
    )
    return parser.parse_args()


def load_prompts(harmful_path: str, harmless_path: str, max_prompts: int | None = None):
    """Load prompts from files or use defaults."""
    print("Loading prompts...")

    if harmful_path and os.path.exists(harmful_path):
        with open(harmful_path) as f:
            harmful = json.load(f)
        if isinstance(harmful, list):
            harmful = [p["text"] if isinstance(p, dict) else p for p in harmful]
    else:
        print(f"Loading from config: {config.DATASET_HARMFUL_TRAIN}")
        harmful = load_dataset_split("train", "harmful", max_prompts)

    if harmless_path and os.path.exists(harmless_path):
        with open(harmless_path) as f:
            harmless = json.load(f)
        if isinstance(harmless, list):
            harmless = [p["text"] if isinstance(p, dict) else p for p in harmless]
    else:
        print(f"Loading from config: {config.DATASET_HARMLESS_TRAIN}")
        harmless = load_dataset_split("train", "harmless", max_prompts)

    print(f"  Harmful prompts: {len(harmful)}")
    print(f"  Harmless prompts: {len(harmless)}")

    return harmful, harmless


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Setup
    print("=" * 60)
    print("Refusal-Lens: Step 1 - Compute Refusal Directions")
    print("=" * 60)

    output_dir = args.output or config.COMPUTED_DIRECTIONS_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Load prompts
    harmful, harmless = load_prompts(args.harmful, args.harmless, args.max_prompts)

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
    print(f"  Layers to compute: {args.layers or config.REFUSAL_LAYERS}")

    # Compute directions
    print("\n" + "=" * 60)
    print("Computing refusal directions...")
    print("=" * 60)

    start_time = time.time()

    computer = RefusalDirectionComputer(model, tokenizer)
    results = computer.compute_all_layers(
        harmful_prompts=harmful,
        harmless_prompts=harmless,
        layers=args.layers,
        position=args.position,
    )

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f} seconds")

    # Save results
    print(f"\nSaving results to {output_dir}")
    save_directions(results, output_dir)

    # Find best layer
    best_layer, best_result = find_best_layer(results)
    print(f"\n{'=' * 60}")
    print(f"Best layer: {best_layer}")
    print(f"  Separation: {best_result.separation:.4f}")
    print(f"{'=' * 60}")

    # Print summary
    print("\nLayer separation scores:")
    print("-" * 30)
    for layer in sorted(results.keys()):
        sep = results[layer].separation
        marker = " <-- BEST" if layer == best_layer else ""
        print(f"  Layer {layer:2d}: {sep:+.4f}{marker}")

    print(f"\n✅ Step 1 complete! Directions saved to {output_dir}")
    print(f"   Run Step 2: python scripts/compute_circuits.py --layer {best_layer}")


if __name__ == "__main__":
    main()
