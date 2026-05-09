"""Upload paper-reproducibility artefacts to an anonymous HuggingFace dataset.

Usage:
    # 1. Create a HuggingFace account that does not contain identifying info
    #    (email handle, display name, profile bio, etc.).
    # 2. Generate a write token at https://huggingface.co/settings/tokens.
    # 3. Run this script:
    HF_TOKEN=<your-write-token> python3 scripts/anon_hf_upload.py \\
        --hf-repo <anon-org-or-user>/refusal-lens-icml2026-data \\
        --create-repo

The script uploads the committed canonical run + sweep variants under the HF
dataset's `runs/` namespace, preserving directory structure. Re-running the
script (without --create-repo) syncs newer files. Reviewers can pull the
result with `scripts/anon_hf_download.py`.

Default run sources:
    data/results/pipeline_runs/run_20260430_023247
    data/results/pipeline_runs/run_20260430_023247_canonical_*
    data/results/pipeline_runs/run_20260430_023247_full_*
    data/results/pipeline_runs/run_20260430_023247_topN

The canonical run is required; the sweep variants are optional and are
included if present locally.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = REPO_ROOT / "data" / "results" / "pipeline_runs"
CANONICAL_RUN = "run_20260430_023247"


def _require_hub():
    try:
        from huggingface_hub import HfApi, create_repo  # noqa: F401
        return HfApi, create_repo
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
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF write token (or set HF_TOKEN env var)",
    )
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Directory containing pipeline runs (default: {DEFAULT_RUNS_DIR})",
    )
    p.add_argument(
        "--canonical-only",
        action="store_true",
        help="Only upload the canonical run, skip sweep variants",
    )
    p.add_argument(
        "--create-repo",
        action="store_true",
        help="Create the dataset repo if it does not exist (idempotent)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List which directories would be uploaded; do not upload",
    )
    return p.parse_args()


def discover_runs(runs_dir: Path, canonical_only: bool) -> list[Path]:
    canonical = runs_dir / CANONICAL_RUN
    if not canonical.exists():
        msg = (
            f"Canonical run not found: {canonical}\n"
            "Ensure the canonical paper run is present before uploading."
        )
        raise SystemExit(msg)
    runs = [canonical]
    if not canonical_only:
        for child in sorted(runs_dir.iterdir()):
            if child.name == CANONICAL_RUN or not child.is_dir():
                continue
            if child.name.startswith(CANONICAL_RUN + "_"):
                runs.append(child)
    return runs


def upload_run(api, hf_repo: str, run_dir: Path, token: str | None) -> None:
    rel = run_dir.relative_to(run_dir.parents[2])  # data/results/pipeline_runs/<run>
    print(f"  Uploading {rel} ({_dir_size_mb(run_dir):.1f} MB) ...")
    api.upload_folder(
        repo_id=hf_repo,
        repo_type="dataset",
        folder_path=str(run_dir),
        path_in_repo=f"runs/{run_dir.name}",
        token=token,
        commit_message=f"upload run {run_dir.name}",
    )


def _dir_size_mb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 * 1024)


def main() -> int:
    args = parse_args()
    if not args.token and not args.dry_run:
        msg = (
            "Provide a HF write token via --token or set HF_TOKEN. "
            "Generate one at https://huggingface.co/settings/tokens."
        )
        raise SystemExit(msg)

    HfApi, create_repo = _require_hub()
    api = HfApi(token=args.token)

    runs = discover_runs(args.runs_dir, args.canonical_only)
    print(f"Discovered {len(runs)} run(s) to upload to {args.hf_repo}:")
    for r in runs:
        print(f"  - {r.name}  ({_dir_size_mb(r):.1f} MB)")

    if args.dry_run:
        print("\nDry run. No upload performed.")
        return 0

    if args.create_repo:
        print(f"\nEnsuring HF dataset repo {args.hf_repo} exists ...")
        create_repo(
            repo_id=args.hf_repo,
            repo_type="dataset",
            token=args.token,
            exist_ok=True,
            private=False,
        )

    print("\nUploading runs ...")
    for r in runs:
        upload_run(api, args.hf_repo, r, args.token)
    print("\nDone. Reviewers can pull this dataset with:")
    print(f"  python3 scripts/anon_hf_download.py --hf-repo {args.hf_repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
