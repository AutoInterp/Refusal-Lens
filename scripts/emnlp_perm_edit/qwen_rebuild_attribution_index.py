"""Reconstruct a Stage-02-schema attribution_results.json from packed graphs.

The Qwen L18 run (`run_emnlp_qwen_L18_20260522`) has its 550 packed attribution
graphs on HF but the Stage 02 raw output (`02_attribution/attribution_results.json`)
was not preserved. Stages 04/07/08 only consume a small slice of that file:

  results[i].conditions[cond].graphs[<mode>].top_features / top50_features
  results[i].feature_comparison[cls].{vs_bare, vs_ctrl, ctrl_vs_bare}

Both are derivable from the packed graphs (CPU-only), which avoids re-paying
the Stage 02 GPU cost. This script writes a reconstructed file in the exact
schema, flagged with metadata.reconstructed=True.

Known approximations vs a real Stage 02 run (documented per design spec
2026-06-01-qwen-subcircuits-topk-design.md §4):
  1. Packed graphs are pruned (node_threshold=0.8, edge_threshold=0.98), so the
     reconstructed feature sets are the pruned top mass. Top-50 membership is
     effectively unaffected; total counts (n_bare, n_cls...) are smaller.
  2. Stage 02's extract_all_features keeps the LAST position-instance of a
     feature on key collision; we SUM signed attribution across position
     instances (more stable for ranking). Set membership is robust to this.

Usage:
    python3 qwen_rebuild_attribution_index.py \
        --graph-data-dir <...>/graph_data \
        --run-dir data/results/pipeline_runs_qwen/run_emnlp_qwen_L18_20260522
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
from graph_loader import (  # noqa: E402
    extract_edge_records_to_target,
    find_measurement_target_node_id,
    load_packed_graph,
)

CONTROLLED_CLASSES = ("fiction", "roleplay", "analytical", "completion", "cognitive_reframe")
CONDITIONS = ["bare"] + [f"jb_{c}" for c in CONTROLLED_CLASSES] + [f"ctrl_{c}" for c in CONTROLLED_CLASSES]


def feature_key(layer: int, feat_idx: int) -> str:
    return f"L{layer}:F{feat_idx}"


def aggregate_feature_records(records: list[dict]) -> dict[str, dict]:
    """Aggregate per-edge records into per-(layer, feature) entries.

    signed attribution is summed across position instances; activation is the
    max across instances (a feature's strongest firing).
    """
    out: dict[str, dict] = {}
    for r in records:
        if r["category"] != "feature" or r["layer"] is None or r["feature"] is None:
            continue
        key = feature_key(r["layer"], r["feature"])
        entry = out.get(key)
        if entry is None:
            out[key] = {
                "layer": r["layer"],
                "feature_idx": r["feature"],
                "attribution": r["signed_attribution"],
                "activation": r["activation"],
            }
        else:
            entry["attribution"] += r["signed_attribution"]
            entry["activation"] = max(entry["activation"], r["activation"])
    return out


def compare_features(bare_feats: dict, cls_feats: dict) -> dict:
    """Verbatim replica of Stage 02's compare_features (02_run_attribution.py).

    Copied rather than imported because Stage 02 imports circuit-tracer at
    module level, which is unavailable on CPU-only boxes. Parity is covered by
    unit tests in tests/test_qwen_subcircuit_orchestration.py.
    """
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


def build_graph_summary(agg: dict[str, dict], save_top: int, target_positions: list[int]) -> dict:
    """Stage-02-shaped per-graph summary from aggregated features."""
    sorted_items = sorted(agg.items(), key=lambda kv: abs(kv[1]["attribution"]), reverse=True)
    attrs = [v["attribution"] for v in agg.values()]
    return {
        "net": sum(attrs),
        "pos_sum": sum(a for a in attrs if a > 0),
        "neg_sum": sum(a for a in attrs if a < 0),
        "n_features": len(agg),
        "n_active": len(agg),
        "target_positions": target_positions,
        "top_features": {k: v["attribution"] for k, v in sorted_items[:save_top]},
        "top50_features": {k: v["attribution"] for k, v in sorted_items[:50]},
        "reconstructed": True,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--graph-data-dir", type=Path, required=True,
                   help="Directory of packed <idx>_<cond>_<mode>.json.gz graphs")
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Qwen run dir; writes <run-dir>/02_attribution/attribution_results.json")
    p.add_argument("--dataset", type=Path,
                   default=REPO / "dataset/refusal_lens_controlled_dataset.json")
    p.add_argument("--graph-mode", default="single", choices=["single", "multi"])
    p.add_argument("--n-prompts", type=int, default=50)
    p.add_argument("--save-top", type=int, default=100,
                   help="top_features size (matches config.SAVE_TOP_FEATURES=100)")
    p.add_argument("--target-positions", default="-1",
                   help="Comma-separated; recorded in summaries (default -1 for Qwen single)")
    p.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.run_dir / "02_attribution"
    out_path = out_dir / "attribution_results.json"
    if out_path.exists() and not args.force:
        print(f"[rebuild] {out_path} already exists — leaving as-is (use --force to overwrite).")
        return

    dataset = json.loads(args.dataset.read_text())
    prompts = dataset["prompts"] if isinstance(dataset, dict) and "prompts" in dataset else dataset
    target_positions = [int(x) for x in str(args.target_positions).split(",")]

    rows = []
    skipped = []
    t0 = time.time()
    for i in range(min(args.n_prompts, len(prompts))):
        prow = prompts[i]
        row = {
            "prompt_idx": i,
            "prompt_id": prow.get("id", i),
            "prompt": prow.get("base") or prow.get("bare") or "",
            "topic": prow.get("topic"),
            "conditions": {},
        }
        prompt_features: dict[str, dict] = {}

        for cond in CONDITIONS:
            slug = f"{i:03d}_{cond}_{args.graph_mode}"
            path = args.graph_data_dir / f"{slug}.json.gz"
            if not path.exists():
                skipped.append(slug)
                continue
            graph = load_packed_graph(path)
            target_id = find_measurement_target_node_id(graph)
            records = extract_edge_records_to_target(graph, target_id, filter_category="feature")
            agg = aggregate_feature_records(records)
            row["conditions"][cond] = {
                "prefix": None,
                "graphs": {args.graph_mode: build_graph_summary(agg, args.save_top, target_positions)},
            }
            prompt_features[cond] = {k: {"attribution": v["attribution"]} for k, v in agg.items()}

        # 3-way feature comparison per class — mirrors Stage 02's assembly.
        bare_feats = prompt_features.get("bare", {})
        if bare_feats:
            prompt_comparison: dict = {}
            for cls in CONTROLLED_CLASSES:
                jb_feats = prompt_features.get(f"jb_{cls}", {})
                ctrl_feats = prompt_features.get(f"ctrl_{cls}", {})
                entry: dict = {}
                if jb_feats:
                    entry["vs_bare"] = compare_features(bare_feats, jb_feats)
                if jb_feats and ctrl_feats:
                    entry["vs_ctrl"] = compare_features(ctrl_feats, jb_feats)
                if ctrl_feats:
                    entry["ctrl_vs_bare"] = compare_features(bare_feats, ctrl_feats)
                if entry:
                    prompt_comparison[cls] = entry
            row["feature_comparison"] = prompt_comparison

        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"[rebuild] {i+1}/{args.n_prompts} prompts ({time.time()-t0:.1f}s)")

    out = {
        "metadata": {
            "reconstructed": True,
            "reconstructed_from": str(args.graph_data_dir),
            "graph_mode": args.graph_mode,
            "save_top": args.save_top,
            "n_prompts": len(rows),
            "n_skipped_graphs": len(skipped),
            "approximations": [
                "packed graphs pruned at node_threshold=0.8 / edge_threshold=0.98",
                "per-feature attribution summed across position instances "
                "(Stage 02 kept last instance on key collision)",
            ],
        },
        "results": rows,
        "skipped_slugs": skipped,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    n_conds = sum(len(r["conditions"]) for r in rows)
    print(f"[rebuild] wrote {out_path}")
    print(f"  {len(rows)} prompts, {n_conds} condition entries, {len(skipped)} graphs missing")
    if skipped:
        print(f"  skipped (first 5): {skipped[:5]}")


if __name__ == "__main__":
    main()
