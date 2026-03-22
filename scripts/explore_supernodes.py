#!/usr/bin/env python
"""Explore supernodes using the SupernodeAnalyzer.

This script implements Step 3 of the research plan:
Use SAE supernodes to understand feature semantics.

Usage:
    python scripts/explore_supernodes.py
    python scripts/explore_supernodes.py --supernode 4
    python scripts/explore_supernodes.py --data-path data/supernodes.json
"""

from __future__ import annotations

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from refusal_lens import SupernodeAnalyzer, config
from refusal_lens.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Explore supernodes and analyze steering vectors"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to supernode data JSON file",
    )
    parser.add_argument(
        "--supernode",
        type=int,
        default=None,
        help="Specific supernode ID to analyze",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top neurons to display",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for analysis results",
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
    print("Refusal-Lens: Step 3 - Supernode Exploration")
    print("=" * 60)

    output_dir = args.output or "data/results/supernodes"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize analyzer
    analyzer = SupernodeAnalyzer()

    # Try to load data from various locations
    data_loaded = False

    if args.data_path:
        # User specified a data path
        if os.path.exists(args.data_path):
            print(f"\nLoading supernode data from: {args.data_path}")
            analyzer.load_supernode_data(args.data_path)
            data_loaded = True
        else:
            print(f"❌ Error: File not found: {args.data_path}")
            sys.exit(1)
    else:
        # Try default locations
        default_paths = [
            "data/supernodes.json",
            "data/neuronpedia_cache/supernodes.json",
            "test/data/test_supernodes.json",
        ]

        for path in default_paths:
            if os.path.exists(path):
                print(f"\nLoading supernode data from: {path}")
                try:
                    analyzer.load_supernode_data(path)
                    data_loaded = True
                    break
                except Exception as e:
                    print(f"  Failed to load {path}: {e}")

        if not data_loaded:
            print("\n⚠️  No supernode data found. Using demo data.")
            # Create demo data for testing
            demo_data = {
                "1": {
                    "neurons": [
                        {
                            "id": 0,
                            "strength": 0.8,
                            "features": ["harmful", "dangerous"],
                        },
                        {"id": 1, "strength": 0.6, "features": ["illegal", "crime"]},
                        {"id": 2, "strength": 0.4, "features": ["refusal", "safety"]},
                    ],
                    "steering_vector": [0.5, 0.3, -0.2, 0.1, -0.4],
                    "activation_distribution": {"high": 0.3, "medium": 0.5, "low": 0.2},
                },
                "2": {
                    "neurons": [
                        {"id": 0, "strength": 0.7, "features": ["help", "useful"]},
                        {"id": 1, "strength": 0.5, "features": ["safe", "positive"]},
                    ],
                    "steering_vector": [0.2, 0.4, 0.3, 0.1, 0.2],
                    "activation_distribution": {"high": 0.2, "medium": 0.6, "low": 0.2},
                },
            }
            import json

            demo_path = os.path.join(output_dir, "demo_supernodes.json")
            with open(demo_path, "w") as f:
                json.dump(demo_data, f)
            print(f"  Created demo data: {demo_path}")
            analyzer.load_supernode_data(demo_path)
            data_loaded = True

    # Get summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    summary = analyzer.get_summary_statistics()
    print(f"  Total supernodes: {summary.get('total_supernodes', 0)}")
    print(f"  Total neurons: {summary.get('total_neurons', 0)}")
    print(
        f"  Avg neurons/supernode: {summary.get('average_neurons_per_supernode', 0):.1f}"
    )
    print(f"  Supernode IDs: {summary.get('supernode_ids', [])}")

    # Analyze specific supernode or all supernodes
    supernode_ids = (
        [args.supernode] if args.supernode else list(analyzer.supernodes.keys())
    )

    print("\n" + "=" * 60)
    print("Supernode Analysis")
    print("=" * 60)

    all_results = {}

    for sn_id in supernode_ids:
        print(f"\n--- Supernode {sn_id} ---")

        # Get top neurons
        top_neurons = analyzer.get_top_neurons(sn_id, args.top_k)
        print(f"  Top {args.top_k} neurons:")
        for neuron in top_neurons:
            print(
                f"    Neuron {neuron.neuron_id}: strength={neuron.activation_strength:.3f}, features={neuron.features}"
            )

        # Analyze steering vector
        steering_stats = analyzer.analyze_steering_vector(sn_id)
        if steering_stats:
            print("  Steering vector stats:")
            print(f"    Magnitude: {steering_stats['magnitude']:.4f}")
            print(f"    Mean: {steering_stats['mean']:.4f}")
            print(f"    Std: {steering_stats['std']:.4f}")
            print(
                f"    Range: [{steering_stats['min']:.4f}, {steering_stats['max']:.4f}]"
            )
            print(f"    Dimension: {steering_stats['dimension']}")

        # Get feature distribution
        feature_dist = analyzer.get_feature_distribution(sn_id)
        if feature_dist:
            print("  Feature distribution (top 5):")
            for feature, count in list(feature_dist.items())[:5]:
                print(f"    {feature}: {count}")

        # Store results
        all_results[f"supernode_{sn_id}"] = {
            "top_neurons": [
                {
                    "neuron_id": n.neuron_id,
                    "activation_strength": n.activation_strength,
                    "features": n.features,
                }
                for n in top_neurons
            ],
            "steering_vector_stats": steering_stats,
            "feature_distribution": feature_dist,
        }

    # Save analysis
    output_file = os.path.join(output_dir, "supernode_analysis.json")
    import json

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Analysis saved to {output_file}")

    print("\n" + "=" * 60)
    print("Step 3 Complete!")
    print("=" * 60)
    print("Supernode exploration uses Neuronpedia data to understand")
    print("which features and neurons contribute to refusal behavior.")
    print("\nNext: Run Step 4 to test jailbreak variants:")
    print("  python scripts/test_jailbreaks.py --n-variants 10")


if __name__ == "__main__":
    main()
