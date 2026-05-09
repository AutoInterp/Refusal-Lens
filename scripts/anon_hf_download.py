"""Download paper-reproducibility artefacts from the anonymous HuggingFace dataset.

Usage:
    python3 scripts/anon_hf_download.py \\
        --hf-repo <anon-org-or-user>/refusal-lens-icml2026-data

The committed `data/results/pipeline_runs/run_20260430_023247/` directory
already contains every JSON / PNG / Markdown summary used by the figure
generator. This script is only required to obtain the raw `.pt` attribution
graphs (≈ 75 GB) and any other artefact too large to ship in the repo.

By default the script downloads into `data/results/pipeline_runs/`,
preserving the dataset's `runs/` layout.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = REPO_ROOT / "data" / "results" / "pipeline_runs"


def _require_hub():
    try:
        from huggingface_hub import snapshot_download  # noqa: F401
        return snapshot_download
    except ImportError as e:
        msg = "huggingface_hub is required. `pip install huggingface_hub`."
        raise SystemExit(msg) from e


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--hf-repo",
        required=True,
        help="HuggingFace dataset repo id, e.g. 'anon-icml/refusal-lens-icml2026-data'",
    )
    p.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"Where to materialise the downloaded runs (default: {DEFAULT_TARGET})",
    )
    p.add_argument(
        "--include",
        action="append",
        default=None,
        help=(
            "Optional glob pattern, relative to the dataset root, restricting "
            "which files to download (e.g. 'runs/run_20260430_023247/02_attribution/**'). "
            "Repeatable. Default: all files."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    snapshot_download = _require_hub()

    args.target_dir.mkdir(parents=True, exist_ok=True)
    staging = args.target_dir / ".anon_hf_staging"

    print(f"Downloading {args.hf_repo} into staging {staging} ...")
    snapshot_path = snapshot_download(
        repo_id=args.hf_repo,
        repo_type="dataset",
        local_dir=str(staging),
        allow_patterns=args.include,
    )
    snapshot_path = Path(snapshot_path)

    runs_root = snapshot_path / "runs"
    if not runs_root.exists():
        print(
            f"Note: dataset has no top-level 'runs/' directory; "
            f"files materialised under {snapshot_path}.",
        )
        return 0

    print(f"Moving runs into {args.target_dir} ...")
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        target = args.target_dir / run_dir.name
        if target.exists():
            print(f"  Merging into existing {target.name} ...")
            for src in run_dir.rglob("*"):
                if not src.is_file():
                    continue
                rel = src.relative_to(run_dir)
                dst = target / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.replace(dst)
        else:
            run_dir.rename(target)
            print(f"  -> {target.name}")

    # Clean up the staging tree (rename above leaves empty dirs).
    for d in sorted(staging.rglob("*"), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    if staging.exists() and not any(staging.iterdir()):
        staging.rmdir()

    print(f"\nDone. Runs available under {args.target_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
