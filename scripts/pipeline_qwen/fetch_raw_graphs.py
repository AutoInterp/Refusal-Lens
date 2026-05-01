"""
Fetch archived raw .pt attribution graphs from the HF dataset back to the
local pipeline run directory. Only needed for analyses that go beyond the
Stage 05 frontend (e.g. re-pruning at different thresholds, full adjacency
matrix inspection, Stage 08 ablation work).

Usage:
    python3 fetch_raw_graphs.py --run run_20260418_172402 \\
        --dataset-repo moon70/refusal-lens-graphs

    # Subset (avoid pulling 80+ GB when you only need a few graphs)
    python3 fetch_raw_graphs.py --run run_20260418_172402 \\
        --dataset-repo moon70/refusal-lens-graphs \\
        --prompts 0,1,2
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

DEFAULT_DATASET_REPO = "AutoInterp/refusal-lens-graphs"
PT_SUBDIR = "raw_graphs"


def parse_args():
    p = argparse.ArgumentParser(description="Fetch raw .pt graphs from HF dataset")
    p.add_argument("--run", type=str, required=True)
    p.add_argument("--dataset-repo", type=str, default=DEFAULT_DATASET_REPO)
    p.add_argument("--prompts", type=str, default=None,
                   help="Comma-separated prompt indices (default: all)")
    p.add_argument("--classes", type=str, default=None,
                   help="Comma-separated condition names (default: all)")
    p.add_argument("--out-base", type=Path, default=None)
    return p.parse_args()


def patterns_for(run: str, prompts: str | None, classes: str | None) -> list[str]:
    """Build HF allow_patterns list from filters."""
    base = f"runs/{run}/{PT_SUBDIR}/"
    if not prompts and not classes:
        return [base + "*.pt"]
    prompt_ids = [f"{int(p):03d}" for p in prompts.split(",")] if prompts else ["*"]
    class_names = classes.split(",") if classes else ["*"]
    patterns = []
    for pid in prompt_ids:
        for cls in class_names:
            patterns.append(f"{base}{pid}_{cls}.pt")
    return patterns


def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. `pip install huggingface_hub`")
        sys.exit(1)

    args = parse_args()
    out_base = args.out_base or (config.REPO_ROOT / "data" / "results" / "pipeline_runs")
    run_dir = out_base / args.run
    dst = run_dir / "02_attribution" / "graphs"
    dst.mkdir(parents=True, exist_ok=True)

    patterns = patterns_for(args.run, args.prompts, args.classes)

    print(f"Downloading raw .pt graphs for {args.run} from {args.dataset_repo}...")
    print(f"  Patterns: {patterns}")
    try:
        snapshot = snapshot_download(
            repo_id=args.dataset_repo,
            repo_type="dataset",
            allow_patterns=patterns,
        )
    except Exception as e:
        print(f"ERROR downloading: {e}")
        sys.exit(1)

    src = Path(snapshot) / "runs" / args.run / PT_SUBDIR
    if not src.exists():
        print(f"ERROR: no raw graphs found at runs/{args.run}/{PT_SUBDIR}/ on {args.dataset_repo}")
        sys.exit(1)

    n, total_bytes = 0, 0
    for pt in sorted(src.glob("*.pt")):
        shutil.copy2(pt, dst / pt.name)
        n += 1
        total_bytes += pt.stat().st_size
    print(f"\nDONE! Copied {n} .pt files ({total_bytes/1e9:.2f} GB) to {dst}")


if __name__ == "__main__":
    main()
