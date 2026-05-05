"""Push a complete pipeline run to HuggingFace for long-term archival.

Uploads everything needed to reproduce downstream analysis without re-running
the GPU-expensive stages:

    runs/<run_name>/
        01_direction/                 ← refusal directions + metadata
            directions/layer_*.pt       ~35 KB × 34 layers (per-layer at pos=-2)
            positions_L15/pos_*.pt      ~35 KB × N positions (per-position at L15)
            refusal_direction.pt        ← legacy single-layer container
            unnormalized_r.pt           ← unnormalized per-layer (for intervention)
            direction_metadata.json
        02_attribution/
            graphs/*.pt                 ~0.5-2 GB each (raw attribution graphs)
            attribution_results.json    ← per-prompt summaries (keeps small)
            feature_comparison_aggregate.json
        config.json                   ← Stage 01's run config snapshot

Skipped:
    - 02_attribution/attribution_checkpoint_*.json  (shard resume state;
      not needed after run completes)
    - 02_attribution/logs/*.log                     (shard stdout logs)

Usage:
    # Prereq: `hf auth login` or HF_TOKEN env var (write scope)
    python push_run.py \\
        --run-dir /workspace/outputs/run_20260421_215920 \\
        --dataset-repo moon70/refusal-lens-graphs

    # Dry-run: size estimate, no upload
    python push_run.py --run-dir <...> --dataset-repo <...> --dry-run

    # Push only direction artifacts (if graphs were already pushed separately)
    python push_run.py --run-dir <...> --dataset-repo <...> --skip-graphs

After a successful upload the script prints the HF browse URL and a one-liner
to reclaim local disk.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: F401

DEFAULT_DATASET_REPO = "AutoInterp/refusal-lens-graphs"


def parse_args():
    p = argparse.ArgumentParser(description="Archive a run dir to HF")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--dataset-repo", type=str, default=DEFAULT_DATASET_REPO)
    p.add_argument("--dry-run", action="store_true",
                   help="Scan + size estimate; do not upload")
    p.add_argument("--skip-graphs", action="store_true",
                   help="Skip 02_attribution/graphs/ (use if you pushed them separately)")
    p.add_argument("--skip-directions", action="store_true",
                   help="Skip 01_direction/ (use if you only want to re-push Stage 02)")
    p.add_argument("--skip-verify", action="store_true",
                   help="Skip post-upload remote file listing check")
    return p.parse_args()


def _collect_files(run_dir: Path, skip_graphs: bool, skip_directions: bool) -> list[Path]:
    """Build the list of files to upload. Applied allow-patterns + skip rules."""
    selected: list[Path] = []

    # 01_direction: everything (small .pt files + JSON metadata)
    if not skip_directions and (run_dir / "01_direction").exists():
        for p in (run_dir / "01_direction").rglob("*"):
            if p.is_file():
                selected.append(p)

    # 02_attribution: JSON summaries always; graphs/ conditional
    attr_dir = run_dir / "02_attribution"
    if attr_dir.exists():
        # JSON summaries (attribution_results, feature_comparison_aggregate)
        # Skip ephemeral checkpoint_*.json and logs/
        for p in attr_dir.glob("*.json"):
            if p.name.startswith("attribution_checkpoint_"):
                continue
            selected.append(p)
        # Graphs (optional — huge)
        if not skip_graphs and (attr_dir / "graphs").exists():
            for p in (attr_dir / "graphs").rglob("*.pt"):
                selected.append(p)

    # Top-level run metadata
    for p in run_dir.glob("*.json"):
        selected.append(p)

    return selected


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
    if not run_dir.exists():
        print(f"ERROR: {run_dir} does not exist")
        sys.exit(1)
    run_name = run_dir.name

    files = _collect_files(run_dir, args.skip_graphs, args.skip_directions)
    if not files:
        print(f"ERROR: no uploadable files found in {run_dir}")
        sys.exit(1)

    total_bytes = sum(p.stat().st_size for p in files)
    total_gb = total_bytes / 1e9

    graphs = [p for p in files if p.parent.name == "graphs"]
    direction_files = [p for p in files if "01_direction" in p.parts]
    json_files = [p for p in files if p.suffix == ".json"]

    print("=" * 60)
    print(f"Archive run: {run_name} → {args.dataset_repo}")
    print("=" * 60)
    print(f"  Source:       {run_dir}")
    print(f"  Files total:  {len(files)} ({total_gb:.2f} GB)")
    print(f"  - 01_direction artifacts:  {len(direction_files)}")
    print(f"  - Raw attribution graphs:  {len(graphs)} "
          f"({sum(p.stat().st_size for p in graphs) / 1e9:.2f} GB)")
    print(f"  - JSON metadata:           {len(json_files)}")
    print(f"  Destination:  runs/{run_name}/   (LFS-tracked .pt files)")

    api = HfApi()
    preflight_auth(api, args.dataset_repo, args.dry_run)

    if args.dry_run:
        print(f"\n[--dry-run] would upload {len(files)} files, {total_gb:.2f} GB")
        return

    try:
        create_repo(args.dataset_repo, repo_type="dataset", exist_ok=True)
        print(f"  Repo ready: https://huggingface.co/datasets/{args.dataset_repo}")
    except Exception as e:
        print(f"ERROR creating/accessing repo: {e}")
        sys.exit(1)

    # HF's upload_folder preserves the relative dir structure under path_in_repo.
    # We point it at the run_dir and strip checkpoint/log files via ignore_patterns.
    eta_min = total_gb * 1000 / 20 / 60
    print(f"\nUploading ~{total_gb:.2f} GB. ETA ~{eta_min:.0f} min at 20 MB/s "
          f"(HF LFS handles resumable uploads — re-run on interruption).")
    print()

    ignore_patterns = [
        "02_attribution/attribution_checkpoint_*.json",
        "02_attribution/logs/**",
        "**/*.log",
    ]
    allow_patterns = None
    if args.skip_graphs:
        ignore_patterns.append("02_attribution/graphs/**")
    if args.skip_directions:
        ignore_patterns.append("01_direction/**")

    api.upload_folder(
        folder_path=str(run_dir),
        repo_id=args.dataset_repo,
        repo_type="dataset",
        path_in_repo=f"runs/{run_name}",
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        commit_message=f"archive run {run_name}",
    )

    if not args.skip_verify:
        print("\nVerifying remote file list...")
        try:
            remote = set(api.list_repo_files(args.dataset_repo, repo_type="dataset"))
        except Exception as e:
            print(f"  WARN: could not list remote ({e}); trust upload result")
            remote = None
        if remote is not None:
            expected = {
                f"runs/{run_name}/{p.relative_to(run_dir)}" for p in files
            }
            missing = expected - remote
            # Normalize path separators in case HF returns posix
            if missing:
                remote_norm = {r.replace("\\", "/") for r in remote}
                missing = {m.replace("\\", "/") for m in missing} - remote_norm
            if missing:
                print(f"  ERROR: {len(missing)} files missing on remote:")
                for m in sorted(missing)[:5]:
                    print(f"    {m}")
                print("  Re-run this script to resume — HF deduplicates by hash.")
                sys.exit(1)
            print(f"  All {len(expected)} files confirmed on remote ✓")

    print(f"\nDONE! Browse: https://huggingface.co/datasets/{args.dataset_repo}/tree/main/runs/{run_name}")
    if graphs:
        print(f"\nSafe to reclaim local disk (graphs only):")
        print(f"  rm -rf {run_dir / '02_attribution' / 'graphs'}")


if __name__ == "__main__":
    main()
