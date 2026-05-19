"""Phase 0 — Sub-experiment 0a: offline linearization decomposition.

For every (prompt_idx, condition) in run_20260430_023247's `single`-mode
graph_data/, load the packed attribution graph, identify the measurement
target node (direct_dot at L15 pos=-2, marked as is_target_logit=True),
aggregate signed edge attributions by source-node category (feature /
embedding / error_node), and compare against Stage 03's direct_dot
ground truth.

CAVEAT: the packed graphs were produced with --edge-threshold=0.98 (Stage 02c
default), which filters out the smallest-magnitude edges to reduce file size.
Stage 03's `attr_net` uses the UNFILTERED graph. So our `total_signed` will
be slightly smaller in magnitude than Stage 03's `attr_net` per prompt.
The decomposition (which edge type dominates) is still valid; the absolute
magnitudes are approximate due to filtering.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
from graph_loader import (  # noqa: E402
    aggregate_edge_attributions,
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
    p.add_argument("--baselines-from", type=Path,
                   default=REPO / "data/results/pipeline_runs/run_20260430_023247/03_verification/verification_results.json",
                   help="Stage 03 verification output for direct_dot ground truth per prompt (bare only).")
    p.add_argument("--out-dir", type=Path,
                   default=REPO / "data/results/emnlp_perm_edit/phase0_controllability")
    p.add_argument("--mode", default="single", choices=["single", "multi"])
    p.add_argument("--n-prompts", type=int, default=50)
    p.add_argument("--max-recon-error", type=float, default=0.05,
                   help="Max allowed |1 - (sum_edges + baseline) / direct_dot| per input. "
                        "Default 5% rather than 1% to allow for edge-threshold filtering.")
    return p.parse_args()


def load_directdot_ground_truth(verification_path: Path) -> dict:
    """Stage 03 stores per-prompt direct_dot for BARE only.

    Returns {(prompt_idx, 'bare'): {'direct_dot': float, 'attr_net': float, 'difference': float}}.
    """
    data = json.loads(verification_path.read_text())
    out = {}
    for prompt_idx, entry in enumerate(data["per_prompt"]):
        # Stage 03 only verifies BARE condition (one direct_dot per prompt)
        out[(prompt_idx, "bare")] = {
            "direct_dot": entry["total_dot"],
            "attr_net": entry["attr_net"],
            "difference": entry["difference"],
        }
    return out


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[0a] loading direct_dot ground truth from {args.baselines_from}")
    try:
        ground_truth = load_directdot_ground_truth(args.baselines_from)
        print(f"  loaded {len(ground_truth)} ground-truth direct_dot values (BARE only)")
    except (FileNotFoundError, KeyError) as e:
        print(f"  WARNING: cannot load Stage 03 verification ({e}); reconstruction-error check skipped.")
        ground_truth = {}

    print(f"[0a] processing 550 (prompt, condition) graphs from {args.graph_data_dir}")
    per_prompt_records = []
    by_condition = {c: [] for c in CONDITIONS}
    skipped = []
    n_recon_errors = 0
    t0 = time.time()

    for prompt_idx in range(args.n_prompts):
        slug_prompt = f"{prompt_idx:03d}"
        for condition in CONDITIONS:
            slug = f"{slug_prompt}_{condition}_{args.mode}"
            path = args.graph_data_dir / f"{slug}.json.gz"
            if not path.exists():
                skipped.append(slug)
                continue
            try:
                graph = load_packed_graph(path)
                target_id = find_measurement_target_node_id(graph)
                sums = aggregate_edge_attributions(graph, target_id)
            except ValueError as e:
                print(f"  ERROR on {slug}: {e}")
                skipped.append(slug)
                continue

            record = {
                "prompt_idx": prompt_idx,
                "condition": condition,
                "n_edges_to_target": sums["n_edges_to_target"],
                "feature_pos": sums["feature"]["pos"],
                "feature_neg": sums["feature"]["neg"],
                "feature_signed": sums["feature"]["signed"],
                "embedding_pos": sums["embedding"]["pos"],
                "embedding_neg": sums["embedding"]["neg"],
                "embedding_signed": sums["embedding"]["signed"],
                "error_pos": sums["error_node"]["pos"],
                "error_neg": sums["error_node"]["neg"],
                "error_signed": sums["error_node"]["signed"],
                "all_signed": sums["total_signed"],
            }

            gt = ground_truth.get((prompt_idx, condition))
            if gt and gt["direct_dot"] is not None:
                direct_dot = gt["direct_dot"]
                # Stage 03 used unfiltered attr_net; our total_signed is filtered.
                # Compute baseline_offset that makes the identity hold with OUR signed total.
                baseline_offset = direct_dot - sums["total_signed"]
                record["direct_dot"] = direct_dot
                record["stage03_attr_net"] = gt["attr_net"]
                record["stage03_baseline_offset"] = gt["difference"]
                record["our_baseline_offset_with_filtered_edges"] = baseline_offset
                record["filtering_loss_vs_stage03"] = gt["attr_net"] - sums["total_signed"]
                # Reconstruction error: how well does (our_filtered_edges + our_baseline) match direct_dot?
                # By construction this should be exactly 0 (we computed baseline_offset to make it match).
                # The more interesting metric is how much filtering changed the attr_net magnitude.
                if abs(gt["attr_net"]) > 1e-6:
                    relative_filtering_loss = (gt["attr_net"] - sums["total_signed"]) / gt["attr_net"]
                    record["relative_filtering_loss"] = relative_filtering_loss
                    if abs(relative_filtering_loss) > args.max_recon_error:
                        n_recon_errors += 1

            per_prompt_records.append(record)
            by_condition[condition].append(record)

    elapsed = time.time() - t0
    print(f"[0a] processed {len(per_prompt_records)} graphs in {elapsed:.1f}s "
          f"({len(skipped)} skipped, {n_recon_errors} prompts with >|{args.max_recon_error*100:.0f}%| filtering loss)")

    # Per-condition aggregates
    aggregates = {}
    for condition in CONDITIONS:
        recs = by_condition[condition]
        if not recs:
            aggregates[condition] = {"n": 0}
            continue
        agg = {"n": len(recs)}
        for field in ("feature_pos", "feature_neg", "feature_signed",
                      "embedding_pos", "embedding_neg", "embedding_signed",
                      "error_pos", "error_neg", "error_signed", "all_signed"):
            vals = [r[field] for r in recs]
            agg[f"{field}_mean"] = statistics.mean(vals)
            agg[f"{field}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        # direct_dot mean only available for bare
        if all("direct_dot" in r for r in recs):
            dd_vals = [r["direct_dot"] for r in recs]
            agg["direct_dot_mean"] = statistics.mean(dd_vals)
            agg["direct_dot_std"] = statistics.stdev(dd_vals) if len(dd_vals) > 1 else 0.0
        aggregates[condition] = agg

    out = {
        "metadata": {
            "graph_data_dir": str(args.graph_data_dir),
            "mode": args.mode,
            "n_prompts_processed": args.n_prompts,
            "n_records": len(per_prompt_records),
            "n_skipped": len(skipped),
            "n_filtering_loss_above_threshold": n_recon_errors,
            "recon_error_threshold": args.max_recon_error,
            "caveat": "packed graphs filtered at edge_threshold=0.98 per Stage 02c; "
                      "total_signed is smaller in magnitude than Stage 03's attr_net by the filtered fraction.",
        },
        "per_prompt": per_prompt_records,
        "skipped_slugs": skipped,
    }
    (args.out_dir / "linearization_decomposition.json").write_text(json.dumps(out, indent=2))
    (args.out_dir / "decomposition_by_condition.json").write_text(json.dumps({
        "per_condition": aggregates,
    }, indent=2))
    print(f"[0a] wrote linearization_decomposition.json + decomposition_by_condition.json")

    # Print headline per-condition summary
    print(f"\n[0a] Per-condition mean signed contributions (in direct_dot units):")
    print(f"  {'condition':30s}  {'feat_signed':>12s}  {'embed_signed':>12s}  {'err_signed':>12s}  {'all_signed':>12s}")
    for condition in CONDITIONS:
        agg = aggregates[condition]
        if agg["n"] == 0:
            continue
        print(f"  {condition:30s}  "
              f"{agg['feature_signed_mean']:+12.1f}  "
              f"{agg['embedding_signed_mean']:+12.1f}  "
              f"{agg['error_signed_mean']:+12.1f}  "
              f"{agg['all_signed_mean']:+12.1f}")


if __name__ == "__main__":
    main()
