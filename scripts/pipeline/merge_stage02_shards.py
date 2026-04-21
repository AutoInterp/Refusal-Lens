"""Merge per-shard Stage 02 outputs into a single attribution_results.json.

Each shard writes `attribution_checkpoint_{start:03d}_{end:03d}.json` during
its run (so concurrent shards don't clobber each other) and an
`attribution_results.json` at the end. When the whole run is parallelised,
those per-shard finals only contain that shard's prompts — this script
stitches them into one corpus-level file + rebuilds
feature_comparison_aggregate.json across all prompts.

Usage:
    python merge_stage02_shards.py --run-dir <run_dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: F401
from utils import load_json, save_json


def parse_args():
    p = argparse.ArgumentParser(description="Merge Stage 02 shards into a single result file")
    p.add_argument("--run-dir", type=Path, required=True)
    return p.parse_args()


def _load_shard_checkpoints(out_dir: Path) -> list[dict]:
    """Collect per-shard checkpoint files. Each one is a full `results` list
    for its prompt-range slice. We prefer the checkpoint over the final JSON
    because Stage 02 writes the checkpoint on every prompt (crash-safe),
    whereas the final JSON is only written at shard completion."""
    shards = sorted(out_dir.glob("attribution_checkpoint_*_*.json"))
    if not shards:
        raise FileNotFoundError(
            f"No per-shard checkpoints in {out_dir}. Did you run "
            f"run_stage02_parallel.sh?"
        )
    loaded = []
    for p in shards:
        data = load_json(p)
        loaded.append({"path": p, "data": data})
    return loaded


def _aggregate(results: list[dict], classes: tuple[str, ...]) -> dict:
    """Re-implement the Stage 02 aggregate across the merged prompt set.

    Stage 02's aggregator is scoped to one run; we can't just sum the per-shard
    aggregates because mean/std need the pooled sample, not a mean-of-means.
    """
    agg: dict = {}
    for cls in classes:
        per_comp: dict = {}
        for kind in ("vs_bare", "vs_ctrl", "ctrl_vs_bare"):
            buckets = {
                "n_shared": [], "n_bare_only": [], "n_cls_only": [],
                "n_sign_flipped": [], "n_dampened": [], "n_amplified_anti": [],
                "n_bare": [], "n_cls": [],
            }
            for row in results:
                comp = (
                    row.get("feature_comparison", {})
                    .get(cls, {})
                    .get(kind)
                )
                if comp:
                    for key in buckets:
                        buckets[key].append(comp[key])
            if buckets["n_shared"]:
                per_comp[kind] = {
                    k: {
                        "mean": round(float(np.mean(v)), 1),
                        "std": round(float(np.std(v)), 1),
                        "min": int(min(v)),
                        "max": int(max(v)),
                    }
                    for k, v in buckets.items()
                }
        if per_comp:
            agg[cls] = per_comp
    return agg


def main():
    args = parse_args()
    out_dir = args.run_dir / "02_attribution"
    if not out_dir.exists():
        raise FileNotFoundError(f"{out_dir} does not exist")

    shards = _load_shard_checkpoints(out_dir)
    print(f"Found {len(shards)} shard checkpoints in {out_dir}")

    all_results: list[dict] = []
    prompt_indices: set[int] = set()
    for s in shards:
        shard_results = s["data"].get("results", [])
        overlap = {r["prompt_idx"] for r in shard_results} & prompt_indices
        if overlap:
            raise ValueError(
                f"shard {s['path'].name} overlaps on prompt_idx {sorted(overlap)}; "
                f"concurrent shards must have disjoint prompt ranges"
            )
        for r in shard_results:
            prompt_indices.add(r["prompt_idx"])
            all_results.append(r)
        print(f"  {s['path'].name}: {len(shard_results)} prompts")

    all_results.sort(key=lambda r: r["prompt_idx"])
    print(f"Merged: {len(all_results)} prompts total")

    # Inherit metadata from any shard that has it (shards all run with the
    # same CLI config when launched via run_stage02_parallel.sh, so any
    # shard's checkpoint metadata is authoritative). Fall back to {} if
    # no shard stored metadata — older runs before this field was added.
    meta_src: dict = {}
    for s in shards:
        md = s["data"].get("metadata")
        if md:
            meta_src = md
            break

    classes = ("roleplay", "fiction", "analytical", "completion", "cognitive_reframe")
    agg = _aggregate(all_results, classes)

    final = {
        "metadata": {
            "n_prompts": len(all_results),
            "model": config.MODEL_NAME,
            "transcoder": config.TRANSCODER_PATH,
            "measurement_layer": meta_src.get("measurement_layer"),
            "measurement_position": meta_src.get("measurement_position"),
            "measurement_mode": meta_src.get("measurement_mode"),
            "target_positions_loaded": meta_src.get("target_positions_loaded"),
            "dataset": meta_src.get("dataset"),
            "n_shards": len(shards),
        },
        "results": all_results,
    }
    save_json(final, out_dir / "attribution_results.json")
    save_json(agg, out_dir / "feature_comparison_aggregate.json")
    print(f"Wrote {out_dir}/attribution_results.json ({len(all_results)} prompts)")
    print(f"Wrote {out_dir}/feature_comparison_aggregate.json")


if __name__ == "__main__":
    main()
