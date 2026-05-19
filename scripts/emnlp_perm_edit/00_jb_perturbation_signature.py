"""Phase 0 — Sub-experiment 0g: JB-class perturbation profile + taxonomy synthesis figure.

For each cluster from 0f and each JB class, compute the perturbation:
  delta_vs_bare(cluster, jb_C)   = mean(cluster attr | jb_C) - mean(cluster attr | bare)
  delta_sem_vs_ctrl(cluster, jb_C) = mean(cluster attr | jb_C) - mean(cluster attr | ctrl_C)

The per-class vector of cluster perturbations is the "perturbation signature."

Headline two-panel figure:
  Left: heatmap of delta_sem_vs_ctrl (rows = JB classes, columns = clusters)
  Right: per-class top-perturbed-cluster bar chart

Tests H0-9 (per-class perturbation signatures localize to specific clusters).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
from graph_loader import (  # noqa: E402
    extract_feature_profile,
    find_measurement_target_node_id,
    load_packed_graph,
)

JB_CLASSES = ["fiction", "roleplay", "analytical", "completion", "cognitive_reframe"]
ALL_CONDS = ["bare"] + [f"jb_{c}" for c in JB_CLASSES] + [f"ctrl_{c}" for c in JB_CLASSES]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graph-data-dir", type=Path,
                   default=REPO / "data/results/pipeline_runs/run_20260430_023247/05_frontend/graph_data")
    p.add_argument("--taxonomy", type=Path,
                   default=REPO / "data/results/emnlp_perm_edit/phase0_controllability/feature_taxonomy.json")
    p.add_argument("--out-dir", type=Path,
                   default=REPO / "data/results/emnlp_perm_edit/phase0_controllability")
    p.add_argument("--n-prompts", type=int, default=50)
    p.add_argument("--mode", default="single")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[0g] loading taxonomy from {args.taxonomy}")
    taxonomy = json.loads(args.taxonomy.read_text())
    cluster_assignment = taxonomy["feature_cluster_assignment"]
    feature_to_cluster = {}
    for key, cluster_id in cluster_assignment.items():
        L_part, F_part = key.split("_")
        L = int(L_part[1:])
        F = int(F_part[1:])
        feature_to_cluster[(L, F)] = int(cluster_id)
    n_clusters = max(feature_to_cluster.values()) + 1
    print(f"  {len(feature_to_cluster)} features assigned to {n_clusters} clusters")

    print(f"[0g] computing per-cluster per-condition activations from packed graphs")
    cluster_cond_attrs = defaultdict(lambda: defaultdict(list))
    n_per_cond = defaultdict(int)
    for prompt_idx in range(args.n_prompts):
        slug_prompt = f"{prompt_idx:03d}"
        for cond in ALL_CONDS:
            slug = f"{slug_prompt}_{cond}_{args.mode}"
            path = args.graph_data_dir / f"{slug}.json.gz"
            if not path.exists():
                continue
            graph = load_packed_graph(path)
            target_id = find_measurement_target_node_id(graph)
            profile = extract_feature_profile(graph, target_id)
            n_per_cond[cond] += 1
            cluster_attrs_this_prompt = defaultdict(list)
            for (L, F), attr in profile.items():
                cid = feature_to_cluster.get((L, F))
                if cid is not None:
                    cluster_attrs_this_prompt[cid].append(attr)
            for cid, attrs in cluster_attrs_this_prompt.items():
                cluster_cond_attrs[cid][cond].append(np.mean(attrs))

    cluster_cond_mean = {}
    cluster_cond_std = {}
    for cid in range(n_clusters):
        cluster_cond_mean[cid] = {}
        cluster_cond_std[cid] = {}
        for cond in ALL_CONDS:
            vals = cluster_cond_attrs[cid].get(cond, [])
            cluster_cond_mean[cid][cond] = float(np.mean(vals)) if vals else 0.0
            cluster_cond_std[cid][cond] = float(np.std(vals)) if len(vals) > 1 else 0.0

    signatures = {}
    for cid in range(n_clusters):
        signatures[cid] = {}
        bare_mean = cluster_cond_mean[cid]["bare"]
        for jb in JB_CLASSES:
            jb_mean = cluster_cond_mean[cid][f"jb_{jb}"]
            ctrl_mean = cluster_cond_mean[cid][f"ctrl_{jb}"]
            signatures[cid][jb] = {
                "delta_vs_bare": jb_mean - bare_mean,
                "delta_sem_vs_ctrl": jb_mean - ctrl_mean,
                "jb_mean": jb_mean, "bare_mean": bare_mean, "ctrl_mean": ctrl_mean,
            }

    out = {
        "metadata": {"n_clusters": n_clusters, "n_prompts": args.n_prompts,
                     "n_per_cond": dict(n_per_cond)},
        "cluster_cond_mean": cluster_cond_mean,
        "cluster_cond_std": cluster_cond_std,
        "signatures": signatures,
    }
    (args.out_dir / "jb_perturbation_signatures.json").write_text(json.dumps(out, indent=2))
    print(f"[0g] wrote jb_perturbation_signatures.json")

    # Headline two-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: delta_sem heatmap
    matrix = np.zeros((len(JB_CLASSES), n_clusters))
    for i, jb in enumerate(JB_CLASSES):
        for cid in range(n_clusters):
            matrix[i, cid] = signatures[cid][jb]["delta_sem_vs_ctrl"]
    vmax = np.abs(matrix).max()
    im1 = ax1.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax1.set_xticks(range(n_clusters))
    ax1.set_xticklabels([f"C{c}" for c in range(n_clusters)])
    ax1.set_yticks(range(len(JB_CLASSES)))
    ax1.set_yticklabels(JB_CLASSES)
    ax1.set_xlabel("Cluster")
    ax1.set_ylabel("JB class")
    ax1.set_title("Δ_sem(cluster, jb_C) = mean(cluster | jb_C) − mean(cluster | ctrl_C)\n"
                  "negative = JB suppresses cluster relative to ctrl; positive = JB recruits")
    for i in range(len(JB_CLASSES)):
        for j in range(n_clusters):
            txt = f"{matrix[i,j]:+.1f}"
            color = "black" if abs(matrix[i,j]) < 0.6 * vmax else "white"
            ax1.text(j, i, txt, ha="center", va="center", fontsize=10, color=color)
    plt.colorbar(im1, ax=ax1, label="Δ_sem")

    # Right: per-class cluster perturbation magnitudes (bar chart)
    width = 0.15
    x = np.arange(n_clusters)
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    for i, jb in enumerate(JB_CLASSES):
        magnitudes = [abs(signatures[cid][jb]["delta_sem_vs_ctrl"]) for cid in range(n_clusters)]
        ax2.bar(x + i * width, magnitudes, width, label=jb, color=colors[i % len(colors)])
    ax2.set_xticks(x + width * (len(JB_CLASSES) - 1) / 2)
    ax2.set_xticklabels([f"C{c}" for c in range(n_clusters)])
    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("|Δ_sem| (cluster perturbation magnitude)")
    ax2.set_title("Per-class cluster perturbation magnitudes\n"
                  "(reveals which cluster each JB class perturbs most)")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_dir / "taxonomy_synthesis_figure.png", dpi=150)
    plt.close()
    print(f"[0g] wrote taxonomy_synthesis_figure.png (HEADLINE FIGURE)")

    # Report markdown
    md = ["# Phase 0 0g — JB-Class Perturbation Signatures + Taxonomy Synthesis\n",
          "## Per-class top-perturbed clusters (by |Δ_sem|)\n"]
    for jb in JB_CLASSES:
        per_cluster = [(cid, abs(signatures[cid][jb]["delta_sem_vs_ctrl"])) for cid in range(n_clusters)]
        per_cluster.sort(key=lambda x: -x[1])
        top = per_cluster[0]
        delta_sign = signatures[top[0]][jb]["delta_sem_vs_ctrl"]
        sign_str = "RECRUITS" if delta_sign > 0 else "SUPPRESSES"
        md.append(f"- **{jb}**: top-perturbed cluster = C{top[0]} ({sign_str}, |Δ_sem| = {top[1]:.2f})")

    md.append("\n## Δ_sem(cluster, jb_C) full matrix\n")
    md.append("| JB class | " + " | ".join(f"C{c}" for c in range(n_clusters)) + " |")
    md.append("|---|" + "---:|" * n_clusters)
    for jb in JB_CLASSES:
        row = [jb]
        for cid in range(n_clusters):
            row.append(f"{signatures[cid][jb]['delta_sem_vs_ctrl']:+.2f}")
        md.append("| " + " | ".join(row) + " |")

    md.append("\n## Pairwise cosine of JB-class signature vectors")
    md.append("\nLow cosine = distinct mechanisms across classes; high cosine = shared mechanism.\n")
    signature_vecs = {}
    for jb in JB_CLASSES:
        signature_vecs[jb] = np.array(
            [signatures[cid][jb]["delta_sem_vs_ctrl"] for cid in range(n_clusters)]
        )
    md.append("| Pair | cos |")
    md.append("|---|---:|")
    for i, jb1 in enumerate(JB_CLASSES):
        for jb2 in JB_CLASSES[i+1:]:
            v1, v2 = signature_vecs[jb1], signature_vecs[jb2]
            cos = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
            md.append(f"| {jb1} × {jb2} | {cos:+.3f} |")

    (args.out_dir / "TAXONOMY_REPORT.md").write_text("\n".join(md))
    print(f"[0g] wrote TAXONOMY_REPORT.md")

    # Stdout summary
    print(f"\n[0g] Per-class top-perturbed clusters:")
    for jb in JB_CLASSES:
        per_cluster = [(cid, abs(signatures[cid][jb]["delta_sem_vs_ctrl"])) for cid in range(n_clusters)]
        per_cluster.sort(key=lambda x: -x[1])
        top = per_cluster[0]
        delta_sign = signatures[top[0]][jb]["delta_sem_vs_ctrl"]
        sign_str = "recruits" if delta_sign > 0 else "suppresses"
        print(f"  {jb:22s}  top cluster = C{top[0]} ({sign_str}, |Δ_sem| = {top[1]:.2f})")

    print(f"\n[0g] Pairwise signature cosines:")
    for i, jb1 in enumerate(JB_CLASSES):
        for jb2 in JB_CLASSES[i+1:]:
            v1, v2 = signature_vecs[jb1], signature_vecs[jb2]
            cos = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
            print(f"  {jb1[:12]:12s} × {jb2[:12]:12s}: cos = {cos:+.3f}")


if __name__ == "__main__":
    main()
