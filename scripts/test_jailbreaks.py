#!/usr/bin/env python
"""Test jailbreak variants using the refusal detector.

This script implements Step 4 of the research plan:
Test systematic jailbreak variants and measure refusal detection rates.

Usage:
    python scripts/test_jailbreaks.py
    python scripts/test_jailbreaks.py --n-variants 100
    python scripts/test_jailbreaks.py --base-prompts "How to hack?" "Write bomb"
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from refusal_lens import (
    DetectionStatus,
    RefusalDetector,
    config,
)
from refusal_lens.jailbreaks import SystematicVariationGenerator
from refusal_lens.utils import save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test jailbreak variants using refusal detector"
    )
    parser.add_argument(
        "--n-variants",
        type=int,
        default=100,
        help="Number of variants to test (default: 100)",
    )
    parser.add_argument(
        "--base-prompts",
        type=str,
        nargs="+",
        default=None,
        help="Custom base prompts",
    )
    parser.add_argument(
        "--role-variations",
        type=str,
        nargs="+",
        default=None,
        help="Custom role variations",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: config.JAILBREAK_TESTS_DIR)",
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
    print("Refusal-Lens: Step 4 - Test Jailbreak Variants")
    print("=" * 60)

    output_dir = args.output or config.JAILBREAK_TESTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Generate variants
    print("\nGenerating jailbreak variants...")
    generator = SystematicVariationGenerator(
        base_prompts=args.base_prompts,
        role_variations=args.role_variations,
    )

    total_possible = len(generator)
    print(f"  Total possible variants: {total_possible}")

    variants = generator.generate_random(args.n_variants, seed=args.seed)
    print(f"  Testing {len(variants)} variants")

    # Load model and detector
    print(f"\nLoading model and detector: {config.MODEL_NAME}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    detector = RefusalDetector(model, tokenizer)

    print(f"  Device: {config.DEVICE}")

    # Test variants
    print("\n" + "=" * 60)
    print("Testing jailbreak variants...")
    print("=" * 60)

    start_time = time.time()

    results = []
    status_counts = defaultdict(int)

    for variant in tqdm(variants, desc="Testing variants"):
        # Generate response
        try:
            inputs = tokenizer(variant, return_tensors="pt").to(config.DEVICE)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.MAX_NEW_TOKENS,
                    temperature=config.TEMPERATURE,
                    top_p=config.TOP_P,
                    do_sample=True,
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Detect refusal
            detection = detector.detect(response)

            result = {
                "prompt": variant,
                "response": response[:500],  # Truncate for storage
                "status": detection.status.value,
                "confidence": detection.confidence,
            }
            results.append(result)

            status_counts[detection.status.value] += 1

        except Exception as e:
            print(f"  Error processing variant: {e}")
            status_counts["error"] += 1

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f} seconds")

    # Save results
    output_file = os.path.join(output_dir, "jailbreak_results.json")
    save_json(results, output_file)
    print(f"\nResults saved to {output_file}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("Test Summary")
    print(f"{'=' * 60}")
    print(f"  Total variants tested: {len(results)}")
    print("\n  Detection Results:")
    print(f"  {'-' * 30}")

    for status in DetectionStatus:
        count = status_counts.get(status.value, 0)
        pct = 100 * count / len(results) if results else 0
        print(f"    {status.value:20s}: {count:4d} ({pct:5.1f}%)")

    if "error" in status_counts:
        print(f"    {'error':20s}: {status_counts['error']:4d}")

    # Calculate success rate
    refused = sum(
        1 for r in results if r["status"] in ["strong_refusal", "weak_refusal"]
    )
    success_rate = 100 * refused / len(results) if results else 0

    print(f"\n  Refusal Rate: {success_rate:.1f}%")
    print(f"\n✅ Step 4 complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
