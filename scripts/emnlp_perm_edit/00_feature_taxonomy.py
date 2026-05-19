"""Phase 0 — Sub-experiment 0f: feature role clustering + Stage 04 annotation.

For each transcoder feature appearing in any (prompt, condition) attribution graph,
build a 22-dim profile:
  - 11 dims: per-condition mean signed attribution to direct_dot
  - 11 dims: per-condition fraction of prompts in which the feature appears

Cluster features in this 22-dim space (hierarchical agglomerative, Ward linkage).
Annotate clusters with their top-25 features' Stage 04 Neuronpedia labels.

Output: feature_taxonomy.json + feature_taxonomy_clusters.md + feature_taxonomy_figure.png

Tests H0-8 (discrete feature roles for refusal).
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

CONDITIONS = [
    "bare",
    "jb_fiction", "jb_roleplay", "jb_analytical", "jb_completion", "jb_cognitive_reframe",
    "ctrl_fiction", "ctrl_roleplay", "ctrl_analytical", "ctrl_completion", "ctrl_cognitive_reframe",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graph-data-dir", type=Path,
                   default=REPO / "data/results/pipeline_runs/run_20260430_023247/05_frontend/graph_data")
    p.add_argument("--stage-04-labels", type=Path,
                   default=REPO / "data/results/pipeline_runs/run_20260430_023247/04_labels/feature_labels.json")
    p.add_argument("--out-dir", type=Path,
                   default=REPO / "data/results/emnlp_perm_edit/phase0_controllability")
    p.add_argument("--n-prompts", type=int, default=50)
    p.add_argument("--mode", default="single")
    p.add_argument("--n-clusters", type=int, default=6,
                   help="Number of clusters; tune after inspecting silhouette + structure.")
    p.add_argument("--min-feature-occurrences", type=int, default=5,
                   help="Filter out features appearing in fewer than N (prompt, condition) graphs.")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[0f] extracting per-feature profiles from {args.graph_data_dir}")
    # feature_id -> condition -> list of signed attributions
    by_feature_cond_attr = defaultdict(lambda: defaultdict(list))
    by_feature_cond_count = defaultdict(lambda: defaultdict(int))
    n_per_cond = defaultdict(int)
    feature_total_occurrences = defaultdict(int)

    for prompt_idx in range(args.n_prompts):
        slug_prompt = f"{prompt_idx:03d}"
        for cond in CONDITIONS:
            slug = f"{slug_prompt}_{cond}_{args.mode}"
            path = args.graph_data_dir / f"{slug}.json.gz"
            if not path.exists():
                continue
            graph = load_packed_graph(path)
            target_id = find_measurement_target_node_id(graph)
            profile = extract_feature_profile(graph, target_id)
            n_per_cond[cond] += 1
            for fid, signed_attr in profile.items():
                by_feature_cond_attr[fid][cond].append(signed_attr)
                by_feature_cond_count[fid][cond] += 1
                feature_total_occurrences[fid] += 1

    n_graphs_processed = sum(n_per_cond.values())
    n_unique_features_all = len(by_feature_cond_attr)
    features = sorted(
        [fid for fid, total in feature_total_occurrences.items()
         if total >= args.min_feature_occurrences]
    )
    print(f"  {n_unique_features_all} unique features across {n_graphs_processed} graphs")
    print(f"  {len(features)} features pass min_occurrences={args.min_feature_occurrences}")
    print(f"  per-condition graphs: {dict(n_per_cond)}")

    # Build 22-dim profile per feature
    n = len(features)
    profiles = np.zeros((n, 22), dtype=np.float32)
    for i, fid in enumerate(features):
        for j, cond in enumerate(CONDITIONS):
            attrs = by_feature_cond_attr[fid].get(cond, [])
            count = by_feature_cond_count[fid].get(cond, 0)
            profiles[i, j] = float(np.mean(attrs)) if attrs else 0.0
            profiles[i, j + 11] = count / max(n_per_cond[cond], 1)

    # Z-score standardize
    means = profiles.mean(axis=0)
    stds = profiles.std(axis=0) + 1e-8
    profiles_std = (profiles - means) / stds

    print(f"[0f] hierarchical agglomerative clustering with Ward linkage, n_clusters={args.n_clusters}")
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    clusterer = AgglomerativeClustering(n_clusters=args.n_clusters, linkage="ward")
    labels = clusterer.fit_predict(profiles_std)
    silhouette = silhouette_score(profiles_std, labels) if len(set(labels)) > 1 else 0.0
    print(f"  silhouette score: {silhouette:.3f}")

    # Per-cluster stats + top-25 features
    cluster_info = {}
    for cluster_id in sorted(set(labels)):
        member_idxs = [i for i in range(n) if labels[i] == cluster_id]
        members = [features[i] for i in member_idxs]
        # Per-member total |attribution| across the corpus (rank by impact)
        member_total_abs = []
        for fid in members:
            total_abs = sum(abs(np.mean(attrs)) for attrs in by_feature_cond_attr[fid].values())
            member_total_abs.append((fid, total_abs))
        member_total_abs.sort(key=lambda x: -x[1])

        # Per-cluster per-condition mean profile (for interpretation)
        member_profiles = profiles[member_idxs]
        cluster_per_cond_attr = {CONDITIONS[j]: float(member_profiles[:, j].mean()) for j in range(11)}
        cluster_per_cond_freq = {CONDITIONS[j]: float(member_profiles[:, j + 11].mean()) for j in range(11)}

        cluster_info[int(cluster_id)] = {
            "n_features": len(members),
            "mean_signed_attr_per_condition": cluster_per_cond_attr,
            "mean_frequency_per_condition": cluster_per_cond_freq,
            "top_25_features": [
                {"layer": L, "feature": F, "total_abs_attribution": ta}
                for (L, F), ta in member_total_abs[:25]
            ],
        }

    # Load Stage 04 annotations
    stage4 = {}
    if args.stage_04_labels.exists():
        try:
            raw = json.loads(args.stage_04_labels.read_text())
            for key, val in raw.items():
                # key format: "L<layer>:F<feature_idx>"
                if isinstance(val, dict) and "layer" in val and "feature_idx" in val:
                    stage4[(val["layer"], val["feature_idx"])] = val
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  WARNING: failed to parse Stage 04 labels ({e})")

    # Generate markdown report
    md = [f"# Phase 0 0f — Feature Role Taxonomy\n",
          f"- n_clusters: {args.n_clusters}",
          f"- silhouette score: {silhouette:.3f}",
          f"- n_features clustered: {len(features)} (min_occurrences={args.min_feature_occurrences})",
          ""]
    for cluster_id in sorted(cluster_info.keys()):
        info = cluster_info[cluster_id]
        md.append(f"## Cluster {cluster_id} ({info['n_features']} features)\n")
        md.append("**Per-condition mean signed attribution:**")
        for cond in CONDITIONS:
            md.append(f"  - {cond}: {info['mean_signed_attr_per_condition'][cond]:+.2f}  "
                     f"(freq {info['mean_frequency_per_condition'][cond]:.2f})")
        md.append("\n**Top-25 features by total |attribution|:**\n")
        md.append("| Layer | Feature | total |attr| | Top logits |")
        md.append("|---|---|---:|---|")
        for f in info["top_25_features"]:
            L, F = f["layer"], f["feature"]
            label_blob = stage4.get((L, F), {})
            top_logits = label_blob.get("top_logits", [])[:5]
            md.append(f"| L{L} | F{F} | {f['total_abs_attribution']:.1f} | "
                     f"{', '.join(repr(t) for t in top_logits)} |")
        md.append("")

    (args.out_dir / "feature_taxonomy_clusters.md").write_text("\n".join(md))

    # Save the full taxonomy JSON
    out = {
        "metadata": {
            "n_features_clustered": len(features),
            "n_features_total": n_unique_features_all,
            "min_feature_occurrences": args.min_feature_occurrences,
            "n_clusters": args.n_clusters,
            "silhouette_score": silhouette,
            "profile_dims": ["attr_mean_per_cond x11", "freq_per_cond x11"],
            "n_graphs_processed": n_graphs_processed,
        },
        "cluster_info": cluster_info,
        "feature_cluster_assignment": {
            f"L{L}_F{F}": int(labels[i]) for i, (L, F) in enumerate(features)
        },
    }
    (args.out_dir / "feature_taxonomy.json").write_text(json.dumps(out, indent=2))
    print(f"[0f] wrote feature_taxonomy.json + feature_taxonomy_clusters.md")

    # Clustered heatmap: rows = features sorted by cluster, cols = 11 conditions
    order = np.argsort(labels)
    attrs_sorted = profiles[order, :11]
    fig, ax = plt.subplots(figsize=(8, max(8, len(features) / 30)))
    vmax = np.abs(attrs_sorted).max()
    im = ax.imshow(attrs_sorted, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(11))
    ax.set_xticklabels(CONDITIONS, rotation=45, ha="right")
    ax.set_ylabel(f"Feature index (sorted by cluster; n_clusters={args.n_clusters})")
    ax.set_title(f"Phase 0 0f — Feature role taxonomy heatmap (silhouette={silhouette:.3f})")
    boundaries = np.where(np.diff(labels[order]) != 0)[0]
    for b in boundaries:
        ax.axhline(b, color="black", linewidth=0.8)
    cbar = plt.colorbar(im, ax=ax, label="mean signed attribution")
    plt.tight_layout()
    plt.savefig(args.out_dir / "feature_taxonomy_figure.png", dpi=150)
    plt.close()
    print(f"[0f] wrote feature_taxonomy_figure.png")

    # Headline summary printed to stdout
    print(f"\n[0f] Cluster summary (n={args.n_clusters}):")
    for cid in sorted(cluster_info):
        info = cluster_info[cid]
        # Identify the cluster's character: average attribution magnitude
        attrs = info["mean_signed_attr_per_condition"]
        max_pos_cond = max(attrs, key=lambda c: attrs[c])
        max_neg_cond = min(attrs, key=lambda c: attrs[c])
        print(f"  Cluster {cid}: n={info['n_features']:4d}  "
              f"max+: {max_pos_cond:25s} ({attrs[max_pos_cond]:+.2f})  "
              f"max-: {max_neg_cond:25s} ({attrs[max_neg_cond]:+.2f})")


if __name__ == "__main__":
    main()
