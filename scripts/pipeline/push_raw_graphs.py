"""
Push raw .pt attribution graphs (Stage 02 output) to the HF dataset for
long-term archive. These files are large (typ. 0.7-2 GB each, 10-50 GB
per full run) and are NOT needed for the Stage 05 frontend — they're
archived so future analyses can pull them back on demand:

  - re-pruning at different node/edge thresholds
  - full 20k×20k adjacency matrix inspection
  - cross-validation against Tejas's causal-intervention runs
  - any Stage 08 analysis that needs the unpruned attribution graph

After a successful upload, the script verifies every file is present on
HF before printing the `rm -rf` command to reclaim local disk space.

Usage:
    # Prereq: `hf auth login` with a write-scope token
    python3 push_raw_graphs.py \\
        --run-dir ../../data/results/pipeline_runs/run_20260418_172402 \\
        --dataset-repo moon70/refusal-lens-graphs

    # Dry-run: scan + size estimate without uploading
    python3 push_raw_graphs.py --run-dir <...> --dataset-repo <...> --dry-run

    # Resume after a failed upload (HF LFS dedupes by hash — safe to re-run)
    python3 push_raw_graphs.py --run-dir <...> --dataset-repo <...>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: F401

DEFAULT_DATASET_REPO = "AutoInterp/refusal-lens-graphs"
PT_SUBDIR = "raw_graphs"


def parse_args():
    p = argparse.ArgumentParser(description="Archive raw .pt graphs to HF dataset")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--dataset-repo", type=str, default=DEFAULT_DATASET_REPO)
    p.add_argument("--dry-run", action="store_true",
                   help="Scan + size estimate; do not upload")
    p.add_argument("--skip-verify", action="store_true",
                   help="Skip post-upload verification of remote file list")
    return p.parse_args()


def preflight_auth(api, dataset_repo: str, dry_run: bool) -> None:
    """Fail fast if the HF token can't write to the target namespace."""
    if dry_run:
        return
    try:
        info = api.whoami()
    except Exception as e:
        print(f"ERROR: HF auth failed ({e}). Run `hf auth login` with a write-scope token.")
        sys.exit(1)
    user_name = info.get("name", "?")
    token_role = info.get("auth", {}).get("accessToken", {}).get("role", "unknown")
    print(f"  Authenticated as: {user_name}  (token role: {token_role})")
    if token_role == "read":
        print("ERROR: token is read-only. Create a Write token at "
              "https://huggingface.co/settings/tokens and re-run `hf auth login`.")
        sys.exit(1)
    namespace = dataset_repo.split("/", 1)[0] if "/" in dataset_repo else user_name
    if namespace != user_name:
        orgs = {o.get("name") for o in info.get("orgs", []) if o.get("name")}
        if namespace not in orgs:
            print(f"ERROR: not a member of '{namespace}'. Use --dataset-repo {user_name}/...")
            sys.exit(1)


def main():
    from huggingface_hub import HfApi, create_repo

    args = parse_args()
    run_dir = args.run_dir.resolve()
    run_name = run_dir.name
    pt_dir = run_dir / "02_attribution" / "graphs"

    if not pt_dir.exists():
        print(f"ERROR: {pt_dir} does not exist. Did Stage 02 run with --save-graphs?")
        sys.exit(1)

    pt_files = sorted(pt_dir.glob("*.pt"))
    if not pt_files:
        print(f"ERROR: no .pt files in {pt_dir}")
        sys.exit(1)

    total_bytes = sum(p.stat().st_size for p in pt_files)
    total_gb = total_bytes / 1e9
    largest = max(pt_files, key=lambda p: p.stat().st_size)
    largest_gb = largest.stat().st_size / 1e9

    print("=" * 60)
    print(f"Archive raw .pt graphs: {run_name} → {args.dataset_repo}")
    print("=" * 60)
    print(f"  Source:      {pt_dir}")
    print(f"  Files:       {len(pt_files)} .pt files")
    print(f"  Total size:  {total_gb:.2f} GB")
    print(f"  Largest:     {largest.name} ({largest_gb:.2f} GB)")
    print(f"  Destination: runs/{run_name}/{PT_SUBDIR}/*.pt   (LFS-tracked)")

    api = HfApi()
    preflight_auth(api, args.dataset_repo, args.dry_run)

    if args.dry_run:
        print(f"\n[--dry-run] would upload {len(pt_files)} files, {total_gb:.2f} GB")
        return

    # Ensure repo exists
    try:
        create_repo(args.dataset_repo, repo_type="dataset", exist_ok=True)
        print(f"  Repo ready: https://huggingface.co/datasets/{args.dataset_repo}")
    except Exception as e:
        print(f"ERROR creating/accessing repo: {e}")
        sys.exit(1)

    # Rough ETA
    eta_min = total_gb * 1000 / 20 / 60  # assume 20 MB/s
    print(f"\nUploading. Estimated ~{eta_min:.0f} min at 20 MB/s on a home connection.")
    print(f"   HF LFS handles resumable uploads — re-run on interruption.")
    print()

    api.upload_folder(
        folder_path=str(pt_dir),
        repo_id=args.dataset_repo,
        repo_type="dataset",
        path_in_repo=f"runs/{run_name}/{PT_SUBDIR}",
        allow_patterns=["*.pt"],
        commit_message=f"archive raw .pt graphs for {run_name}",
    )

    # Post-upload verification
    if not args.skip_verify:
        print("\nVerifying remote file list...")
        try:
            remote = set(api.list_repo_files(args.dataset_repo, repo_type="dataset"))
        except Exception as e:
            print(f"  WARN: could not list remote ({e}); trust upload result")
            remote = None
        if remote is not None:
            expected = {f"runs/{run_name}/{PT_SUBDIR}/{p.name}" for p in pt_files}
            missing = expected - remote
            if missing:
                print(f"  ERROR: {len(missing)} files missing on remote:")
                for m in sorted(missing)[:5]:
                    print(f"    {m}")
                print(f"  Do NOT delete local .pt files. Re-run this script to resume.")
                sys.exit(1)
            print(f"  All {len(expected)} files confirmed on remote ✓")

    print(f"\nDONE! Browse: https://huggingface.co/datasets/{args.dataset_repo}/tree/main/runs/{run_name}/{PT_SUBDIR}")
    print(f"\nSafe to reclaim local disk:")
    print(f"  rm -rf {pt_dir}")
    print(f"\nPull them back later:")
    print(f"  python3 fetch_raw_graphs.py --run {run_name} --dataset-repo {args.dataset_repo}")


if __name__ == "__main__":
    main()
