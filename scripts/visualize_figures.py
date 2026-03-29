#!/usr/bin/env python
"""Visualize results from the Refusal-Lens pipeline.

This script generates publication-ready figures to explain:
1. Computed refusal directions across layers
2. Attribution circuits for harmful vs harmless prompts
3. Supernode analysis and feature distributions
4. Token-level attribution patterns

Usage:
    python scripts/visualize_figures.py
    python scripts/visualize_figures.py --output-dir images/figures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set style for publication-quality figures
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# Configure matplotlib for better rendering
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10


def load_json(filepath: str) -> Any:
    """Load JSON file."""
    with open(filepath) as f:
        return json.load(f)


def plot_direction_separation(directions_summary: dict, output_path: str) -> None:
    """Plot separation scores across layers.

    Args:
        directions_summary: Dictionary with layer separation data
        output_path: Path to save the figure
    """
    layers = sorted([int(k) for k in directions_summary])
    separations = [directions_summary[str(layer)]["separation"] for layer in layers]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bar plot with gradient colors
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(layers)))
    bars = ax.bar(layers, separations, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar, sep in zip(bars, separations, strict=False):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{sep:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Layer", fontweight="bold")
    ax.set_ylabel("Separation Score", fontweight="bold")
    ax.set_title(
        "Refusal Direction Separation Across Layers", fontweight="bold", pad=20
    )
    ax.set_xticks(layers)
    ax.grid(axis="y", alpha=0.3)

    # Add horizontal line for average separation
    avg_sep = np.mean(separations)
    ax.axhline(
        y=avg_sep,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Average: {avg_sep:.1f}",
    )
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  ✓ Saved direction separation plot: {output_path}")


def plot_token_attributions(circuit_data: dict, prompt: str, output_path: str) -> None:
    """Plot token-level attributions for a single prompt.

    Args:
        circuit_data: Circuit analysis data for a prompt
        prompt: The prompt text
        output_path: Path to save the figure
    """
    token_attributions = circuit_data["token_attributions"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])

    # Top plot: Attribution values
    x = range(len(token_attributions))
    colors = [
        "red" if attr > np.mean(token_attributions) else "blue"
        for attr in token_attributions
    ]

    ax1.bar(
        x, token_attributions, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5
    )
    ax1.set_xlabel("Token Position", fontweight="bold")
    ax1.set_ylabel("Attribution Score", fontweight="bold")
    ax1.set_title(f'Token Attributions: "{prompt[:50]}..."', fontweight="bold", pad=15)
    ax1.axhline(
        y=np.mean(token_attributions),
        color="green",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {np.mean(token_attributions):.2f}",
    )
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Bottom plot: Cumulative attribution
    cumulative = np.cumsum(token_attributions)
    ax2.fill_between(x, cumulative, alpha=0.3, color="purple")
    ax2.plot(x, cumulative, color="purple", linewidth=2)
    ax2.set_xlabel("Token Position", fontweight="bold")
    ax2.set_ylabel("Cumulative Attribution", fontweight="bold")
    ax2.set_title("Cumulative Attribution Across Tokens", fontweight="bold", pad=15)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  ✓ Saved token attribution plot: {output_path}")


def plot_top_features(
    circuit_data: dict, prompt: str, output_path: str, top_n: int = 20
) -> None:
    """Plot top features by attribution score.

    Args:
        circuit_data: Circuit analysis data for a prompt
        prompt: The prompt text
        output_path: Path to save the figure
        top_n: Number of top features to show
    """
    top_features = circuit_data["top_features"][:top_n]
    feature_ids = [f[0] for f in top_features]
    feature_scores = [f[1] for f in top_features]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create horizontal bar plot
    y_pos = np.arange(len(feature_ids))
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(feature_ids)))

    bars = ax.barh(
        y_pos, feature_scores, color=colors, edgecolor="black", linewidth=0.5
    )

    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Feature {fid}" for fid in feature_ids])
    ax.invert_yaxis()  # Top feature at top
    ax.set_xlabel("Attribution Score", fontweight="bold")
    ax.set_ylabel("Feature ID", fontweight="bold")
    ax.set_title(f'Top {top_n} Features: "{prompt[:40]}..."', fontweight="bold", pad=15)
    ax.grid(axis="x", alpha=0.3)

    # Add value labels
    for bar, score in zip(bars, feature_scores, strict=False):
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2.0,
            f"{score:.2f}",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  ✓ Saved top features plot: {output_path}")


def plot_supernode_activations(supernode_data: dict, output_path: str) -> None:
    """Plot neuron activations for each supernode.

    Args:
        supernode_data: Supernode analysis data
        output_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (supernode_name, data) in enumerate(supernode_data.items()):
        if idx >= 4:  # Only plot first 4 supernodes
            break

        ax = axes[idx]
        neurons = data["top_neurons"]

        neuron_ids = [n["neuron_id"] for n in neurons]
        activations = [n["activation_strength"] for n in neurons]

        # Create bar plot
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(neuron_ids)))
        bars = ax.bar(
            range(len(neuron_ids)),
            activations,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
        )

        # Customize
        ax.set_xlabel("Neuron Rank", fontweight="bold")
        ax.set_ylabel("Activation Strength", fontweight="bold")
        ax.set_title(f'{supernode_name.replace("_", " ").title()}', fontweight="bold")
        ax.set_xticks(range(len(neuron_ids)))
        ax.set_xticklabels([f"N{nid}" for nid in neuron_ids], rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar, act in zip(bars, activations, strict=False):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{act:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    plt.suptitle("Supernode Neuron Activations", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  ✓ Saved supernode activations plot: {output_path}")


def plot_feature_distributions(supernode_data: dict, output_path: str) -> None:
    """Plot feature distributions across supernodes.

    Args:
        supernode_data: Supernode analysis data
        output_path: Path to save the figure
    """
    # Collect all features
    all_features = {}
    for supernode_name, data in supernode_data.items():
        for feature, count in data["feature_distribution"].items():
            if feature not in all_features:
                all_features[feature] = {}
            all_features[feature][supernode_name] = count

    # Prepare data for stacked bar plot
    features = sorted(all_features.keys())
    supernodes = sorted(supernode_data.keys())

    data_matrix = np.zeros((len(features), len(supernodes)))
    for i, feature in enumerate(features):
        for j, supernode in enumerate(supernodes):
            data_matrix[i, j] = all_features[feature].get(supernode, 0)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Create stacked bar plot
    x = np.arange(len(features))
    width = 0.6
    bottom = np.zeros(len(features))

    colors = plt.cm.Set3(np.linspace(0, 1, len(supernodes)))

    for j, supernode in enumerate(supernodes):
        ax.bar(
            x,
            data_matrix[:, j],
            width,
            bottom=bottom,
            label=supernode.replace("_", " ").title(),
            color=colors[j],
            edgecolor="black",
            linewidth=0.5,
        )
        bottom += data_matrix[:, j]

    # Customize
    ax.set_xlabel("Feature", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.set_title("Feature Distribution Across Supernodes", fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  ✓ Saved feature distribution plot: {output_path}")


def plot_steering_vector_stats(supernode_data: dict, output_path: str) -> None:
    """Plot steering vector statistics across supernodes.

    Args:
        supernode_data: Supernode analysis data
        output_path: Path to save the figure
    """
    # Filter supernodes with steering vector stats
    supernodes_with_stats = {
        name: data
        for name, data in supernode_data.items()
        if data.get("steering_vector_stats") is not None
    }

    if not supernodes_with_stats:
        print("  ⚠ No steering vector statistics available")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Magnitude comparison
    ax1 = axes[0, 0]
    names = [name.replace("_", " ").title() for name in supernodes_with_stats]
    magnitudes = [
        data["steering_vector_stats"]["magnitude"]
        for data in supernodes_with_stats.values()
    ]

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    bars = ax1.bar(names, magnitudes, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Magnitude", fontweight="bold")
    ax1.set_title("Steering Vector Magnitude", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    for bar, mag in zip(bars, magnitudes, strict=False):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{mag:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Plot 2: Mean vs Std
    ax2 = axes[0, 1]
    means = [
        data["steering_vector_stats"]["mean"] for data in supernodes_with_stats.values()
    ]
    stds = [
        data["steering_vector_stats"]["std"] for data in supernodes_with_stats.values()
    ]

    ax2.scatter(means, stds, s=200, c=colors, edgecolor="black", linewidth=1)
    ax2.set_xlabel("Mean", fontweight="bold")
    ax2.set_ylabel("Standard Deviation", fontweight="bold")
    ax2.set_title("Mean vs Standard Deviation", fontweight="bold")
    ax2.grid(alpha=0.3)

    # Add labels
    for i, name in enumerate(names):
        ax2.annotate(
            name,
            (means[i], stds[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
        )

    # Plot 3: Min-Max range
    ax3 = axes[1, 0]
    mins = [
        data["steering_vector_stats"]["min"] for data in supernodes_with_stats.values()
    ]
    maxs = [
        data["steering_vector_stats"]["max"] for data in supernodes_with_stats.values()
    ]

    x_pos = np.arange(len(names))
    ax3.bar(
        x_pos, maxs, color="lightcoral", label="Max", edgecolor="black", linewidth=0.5
    )
    ax3.bar(
        x_pos, mins, color="lightblue", label="Min", edgecolor="black", linewidth=0.5
    )
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(names, rotation=45, ha="right")
    ax3.set_ylabel("Value", fontweight="bold")
    ax3.set_title("Min-Max Range", fontweight="bold")
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)

    # Plot 4: Dimension comparison
    ax4 = axes[1, 1]
    dimensions = [
        data["steering_vector_stats"]["dimension"]
        for data in supernodes_with_stats.values()
    ]

    bars = ax4.bar(names, dimensions, color=colors, edgecolor="black", linewidth=0.5)
    ax4.set_ylabel("Dimension", fontweight="bold")
    ax4.set_title("Steering Vector Dimension", fontweight="bold")
    ax4.grid(axis="y", alpha=0.3)

    for bar, dim in zip(bars, dimensions, strict=False):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{dim}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.suptitle("Steering Vector Statistics", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  ✓ Saved steering vector stats plot: {output_path}")


def plot_circuit_comparison(circuits_dir: str, output_path: str) -> None:
    """Compare attribution patterns across different prompts.

    Args:
        circuits_dir: Directory containing circuit results
        output_path: Path to save the figure
    """
    # Load all circuit files
    circuit_files = []
    for layer_dir in Path(circuits_dir).iterdir():
        if layer_dir.is_dir():
            for json_file in layer_dir.glob("*.json"):
                circuit_files.append(json_file)

    if not circuit_files:
        print("  ⚠ No circuit files found for comparison")
        return

    # Load data
    prompts = []
    avg_attributions = []
    max_attributions = []
    n_tokens = []

    for filepath in circuit_files[:10]:  # Limit to first 10 for readability
        data = load_json(str(filepath))
        prompt = data["prompt"]
        attributions = data["token_attributions"]

        prompts.append(prompt[:30] + "..." if len(prompt) > 30 else prompt)
        avg_attributions.append(np.mean(attributions))
        max_attributions.append(np.max(attributions))
        n_tokens.append(len(attributions))

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Average attribution
    ax1 = axes[0]
    x = np.arange(len(prompts))
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(prompts)))
    ax1.bar(x, avg_attributions, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(prompts, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Average Attribution", fontweight="bold")
    ax1.set_title("Average Token Attribution by Prompt", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # Plot 2: Maximum attribution
    ax2 = axes[1]
    ax2.bar(x, max_attributions, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(prompts, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Maximum Attribution", fontweight="bold")
    ax2.set_title("Maximum Token Attribution by Prompt", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    # Plot 3: Number of tokens
    ax3 = axes[2]
    ax3.bar(x, n_tokens, color=colors, edgecolor="black", linewidth=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(prompts, rotation=45, ha="right", fontsize=8)
    ax3.set_ylabel("Number of Tokens", fontweight="bold")
    ax3.set_title("Token Count by Prompt", fontweight="bold")
    ax3.grid(axis="y", alpha=0.3)

    plt.suptitle("Circuit Analysis Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  ✓ Saved circuit comparison plot: {output_path}")


def create_summary_dashboard(
    directions_summary: dict, supernode_data: dict, output_path: str
) -> None:
    """Create a comprehensive summary dashboard.

    Args:
        directions_summary: Direction separation data
        supernode_data: Supernode analysis data
        output_path: Path to save the figure
    """
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Panel 1: Direction separation
    ax1 = fig.add_subplot(gs[0, :2])
    layers = sorted([int(k) for k in directions_summary])
    separations = [directions_summary[str(layer)]["separation"] for layer in layers]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(layers)))
    ax1.bar(layers, separations, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Separation Score")
    ax1.set_title("Refusal Direction Separation", fontweight="bold")
    ax1.set_xticks(layers)
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2: Supernode feature distribution
    ax2 = fig.add_subplot(gs[0, 2:])
    all_features = {}
    for data in supernode_data.values():
        for feature, count in data["feature_distribution"].items():
            all_features[feature] = all_features.get(feature, 0) + count

    features = sorted(all_features.keys())
    counts = [all_features[f] for f in features]
    colors = plt.cm.Set3(np.linspace(0, 1, len(features)))
    ax2.pie(counts, labels=features, colors=colors, autopct="%1.1f%%", startangle=90)
    ax2.set_title("Feature Distribution (All Supernodes)", fontweight="bold")

    # Panel 3: Supernode activation strengths
    ax3 = fig.add_subplot(gs[1, :2])
    supernode_names = []
    avg_activations = []

    for name, data in supernode_data.items():
        neurons = data["top_neurons"]
        if neurons:
            supernode_names.append(name.replace("_", " ").title())
            avg_activations.append(np.mean([n["activation_strength"] for n in neurons]))

    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(supernode_names)))
    bars = ax3.bar(
        supernode_names, avg_activations, color=colors, edgecolor="black", linewidth=0.5
    )
    ax3.set_ylabel("Average Activation")
    ax3.set_title("Average Neuron Activation by Supernode", fontweight="bold")
    ax3.grid(axis="y", alpha=0.3)

    for bar, act in zip(bars, avg_activations, strict=False):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{act:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Panel 4: Steering vector magnitudes
    ax4 = fig.add_subplot(gs[1, 2:])
    sv_names = []
    sv_magnitudes = []

    for name, data in supernode_data.items():
        if data.get("steering_vector_stats"):
            sv_names.append(name.replace("_", " ").title())
            sv_magnitudes.append(data["steering_vector_stats"]["magnitude"])

    if sv_names:
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sv_names)))
        bars = ax4.bar(
            sv_names, sv_magnitudes, color=colors, edgecolor="black", linewidth=0.5
        )
        ax4.set_ylabel("Magnitude")
        ax4.set_title("Steering Vector Magnitude", fontweight="bold")
        ax4.grid(axis="y", alpha=0.3)

        for bar, mag in zip(bars, sv_magnitudes, strict=False):
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{mag:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # Panel 5: Summary statistics
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis("off")

    # Calculate summary statistics
    total_neurons = sum(len(data["top_neurons"]) for data in supernode_data.values())
    total_features = len(all_features)
    avg_sep = np.mean(separations)
    max_sep = np.max(separations)

    summary_text = (
        f"Summary Statistics:\n\n"
        f"• Layers Analyzed: {len(layers)}\n"
        f"• Average Separation: {avg_sep:.2f}\n"
        f"• Maximum Separation: {max_sep:.2f}\n"
        f"• Supernodes Identified: {len(supernode_data)}\n"
        f"• Total Neurons: {total_neurons}\n"
        f"• Unique Features: {total_features}\n"
        f"• Top Feature: {features[np.argmax(counts)]} ({max(counts)} occurrences)"
    )

    ax5.text(
        0.1,
        0.5,
        summary_text,
        transform=ax5.transAxes,
        fontsize=12,
        verticalalignment="center",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        fontfamily="monospace",
    )

    plt.suptitle(
        "Refusal-Lens Analysis Dashboard", fontsize=18, fontweight="bold", y=1.02
    )
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  ✓ Saved summary dashboard: {output_path}")


def main():
    """Main visualization pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate visualization figures for Refusal-Lens results"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/results",
        help="Directory containing result data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="images/figures",
        help="Directory to save generated figures",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format for figures",
    )
    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Refusal-Lens: Visualization Pipeline")
    print("=" * 60)

    # 1. Plot direction separation
    print("\n[1/6] Plotting direction separation...")
    directions_file = data_dir / "computed_directions" / "summary.json"
    if directions_file.exists():
        directions_summary = load_json(str(directions_file))
        plot_direction_separation(
            directions_summary, str(output_dir / f"direction_separation.{args.format}")
        )
    else:
        print("  ⚠ Directions summary not found")

    # 2. Plot token attributions for sample circuits
    print("\n[2/6] Plotting token attributions...")
    circuits_dir = data_dir / "circuits"
    if circuits_dir.exists():
        # Find first circuit file (skip summary.json)
        circuit_files = [
            f for f in circuits_dir.rglob("*.json") if f.name != "summary.json"
        ]
        if circuit_files:
            circuit_data = load_json(str(circuit_files[0]))
            plot_token_attributions(
                circuit_data,
                circuit_data["prompt"],
                str(output_dir / f"token_attributions_sample.{args.format}"),
            )
        else:
            print("  ⚠ No circuit files found")
    else:
        print("  ⚠ Circuits directory not found")

    # 3. Plot top features
    print("\n[3/6] Plotting top features...")
    if circuits_dir.exists() and circuit_files:
        circuit_data = load_json(str(circuit_files[0]))
        plot_top_features(
            circuit_data,
            circuit_data["prompt"],
            str(output_dir / f"top_features_sample.{args.format}"),
        )

    # 4. Plot supernode activations
    print("\n[4/6] Plotting supernode activations...")
    supernode_file = data_dir / "supernodes" / "supernode_analysis.json"
    if supernode_file.exists():
        supernode_data = load_json(str(supernode_file))
        plot_supernode_activations(
            supernode_data, str(output_dir / f"supernode_activations.{args.format}")
        )

        # 5. Plot feature distributions
        print("\n[5/6] Plotting feature distributions...")
        plot_feature_distributions(
            supernode_data, str(output_dir / f"feature_distributions.{args.format}")
        )

        # 6. Plot steering vector stats
        print("\n[6/6] Plotting steering vector statistics...")
        plot_steering_vector_stats(
            supernode_data, str(output_dir / f"steering_vector_stats.{args.format}")
        )
    else:
        print("  ⚠ Supernode analysis not found")

    # Create summary dashboard
    print("\n[Bonus] Creating summary dashboard...")
    if directions_file.exists() and supernode_file.exists():
        create_summary_dashboard(
            directions_summary,
            supernode_data,
            str(output_dir / f"summary_dashboard.{args.format}"),
        )

    # Circuit comparison
    print("\n[Bonus] Creating circuit comparison...")
    if circuits_dir.exists():
        plot_circuit_comparison(
            str(circuits_dir), str(output_dir / f"circuit_comparison.{args.format}")
        )

    print("\n" + "=" * 60)
    print("Visualization Complete!")
    print("=" * 60)
    print(f"\nFigures saved to: {output_dir}")
    print("\nGenerated figures:")
    for fig_file in sorted(output_dir.glob(f"*.{args.format}")):
        print(f"  • {fig_file.name}")


if __name__ == "__main__":
    main()
