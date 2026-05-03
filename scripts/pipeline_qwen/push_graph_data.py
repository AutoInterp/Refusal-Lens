"""
Push a pipeline run's graph_data + metadata + subcircuits to the HuggingFace
dataset so collaborators can pull it with fetch_graph_data.py.

Upload source (local):
    <run_dir>/05_frontend/graph_data/*.json   (staged by 05_visualize_circuits.py)
    <run_dir>/05_frontend/data/graph-metadata.json
    <run_dir>/07_subcircuits/subcircuits.json   (or --subcircuits-run override)

Upload target (HF dataset, default AutoInterp/refusal-lens-graphs):
    runs/<run_name>/graph_data/*.json.gz        (gzipped in flight; ~12x smaller)
    runs/<run_name>/graph-metadata.json
    runs/<run_name>/subcircuits.json
    runs/<run_name>/run_info.json               (n_prompts, classes, source hash)

Incremental updates: re-push a run and only the files that changed get uploaded
(HF deduplicates by content hash).

Usage:
    # Prereq: HF_TOKEN env var or `huggingface-cli login`
    python3 push_graph_data.py --run-dir ../../data/results/pipeline_runs/run_20260418_172402
    python3 push_graph_data.py --run-dir <run> --subcircuits-run ../../data/results/pipeline_runs/run_20260417_010035
    python3 push_graph_data.py --run-dir <run> --dry-run    # bundle only; don't upload
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: F401
from utils_viz import gzip_json_files

DEFAULT_DATASET_REPO = "AutoInterp/refusal-lens-graphs-qwen"


def parse_args():
    p = argparse.ArgumentParser(description="Push graph data bundle to HF dataset")
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Local run directory (must have 05_frontend/graph_data/ staged)")
    p.add_argument("--subcircuits-run", type=Path, default=None,
                   help="Different run for subcircuits.json (default: --run-dir)")
    p.add_argument("--dataset-repo", type=str, default=DEFAULT_DATASET_REPO)
    p.add_argument("--dry-run", action="store_true",
                   help="Gzip + stage only; do not upload")
    p.add_argument("--keep-staging", action="store_true",
                   help="Keep the temp staging directory after upload (for inspection)")
    return p.parse_args()


def preflight_auth(api, dataset_repo: str, dry_run: bool) -> None:
    """Fail fast if the HF token can't write to the target namespace.

    We call `whoami()` to get the active identity + token permissions, then
    compare the requested dataset namespace against the user's own username
    and their org memberships. If the token is read-only or the user isn't
    in the namespace, print actionable guidance and exit before uploading.
    """
    if dry_run:
        return
    try:
        info = api.whoami()
    except Exception as e:
        print(f"ERROR: HF auth failed ({e}). Run `hf auth login` with a "
              f"token that has WRITE scope for datasets.")
        sys.exit(1)

    user_name = info.get("name", "?")
    token_auth = info.get("auth", {}).get("accessToken") or {}
    token_role = token_auth.get("role", "unknown")
    print(f"  Authenticated as: {user_name}  (token role: {token_role})")

    if token_role == "read":
        print(f"\nERROR: your token is READ-ONLY.")
        print(f"  Fix: create a WRITE token at https://huggingface.co/settings/tokens")
        print(f"  with scope 'Write' (or a fine-grained token with "
              f"'Write access to contents' on the target dataset repo),")
        print(f"  then run: hf auth login")
        sys.exit(1)

    # Check namespace membership
    namespace = dataset_repo.split("/", 1)[0] if "/" in dataset_repo else user_name
    if namespace == user_name:
        return
    orgs = {o.get("name") for o in info.get("orgs", []) if o.get("name")}
    if namespace not in orgs:
        print(f"\nERROR: you are not a member of '{namespace}'.")
        print(f"  Your namespace: {user_name}")
        print(f"  Your orgs:      {sorted(orgs) if orgs else '(none)'}")
        print(f"\nOptions to unblock:")
        print(f"  (a) Use your personal namespace:")
        print(f"      python3 push_graph_data.py --run-dir <...> "
              f"--dataset-repo {user_name}/refusal-lens-graphs")
        print(f"  (b) Create the dataset under '{namespace}' manually at "
              f"https://huggingface.co/new-dataset (you need write access to the org),")
        print(f"      then re-run this script.")
        print(f"  (c) Ask an org admin to add you to '{namespace}'.")
        sys.exit(1)


def main():
    from huggingface_hub import HfApi, create_repo

    args = parse_args()
    run_dir = args.run_dir.resolve()
    run_name = run_dir.name
    sc_run = (args.subcircuits_run or run_dir).resolve()

    # Required sources
    src_gd = run_dir / "05_frontend" / "graph_data"
    src_md = run_dir / "05_frontend" / "data" / "graph-metadata.json"
    sc_json = sc_run / "07_subcircuits" / "subcircuits.json"

    if not src_gd.exists():
        print(f"ERROR: {src_gd} not found. Run 05_visualize_circuits.py first.")
        sys.exit(1)
    if not src_md.exists():
        print(f"ERROR: {src_md} not found.")
        sys.exit(1)

    # Count files — may be a mix of .json and .json.gz if user already gzipped
    json_files = list(src_gd.glob("*.json"))
    gz_files = list(src_gd.glob("*.json.gz"))
    graph_files = json_files + gz_files
    graph_files = [f for f in graph_files if f.name != "graph-metadata.json"]
    if not graph_files:
        print(f"ERROR: no graph files found in {src_gd}")
        sys.exit(1)

    print("=" * 60)
    print(f"Pushing run '{run_name}' → {args.dataset_repo}")
    print("=" * 60)
    print(f"  Source graph_data:   {src_gd}")
    print(f"  Source metadata:     {src_md}")
    print(f"  Source subcircuits:  {sc_json}"
          f"{'  [MISSING — skipping]' if not sc_json.exists() else ''}")
    print(f"  Graphs found:        {len(graph_files)} ({len(json_files)} plain, {len(gz_files)} already gz)")

    # Pre-flight auth (skipped on --dry-run)
    api = HfApi()
    preflight_auth(api, args.dataset_repo, args.dry_run)

    keep_dir = None
    try:
        staging_root = tempfile.mkdtemp(prefix="refusal-lens-push-")
        staging = Path(staging_root)
        staged_gd = staging / "graph_data"
        staged_gd.mkdir(parents=True, exist_ok=True)
        keep_dir = staging if args.keep_staging else None

        # Copy plain JSONs into staging then gzip them
        if json_files:
            print(f"\n  Gzipping {len(json_files)} plain JSONs...")
            for jp in json_files:
                shutil.copy2(jp, staged_gd / jp.name)
            report = gzip_json_files(staged_gd, keep_plain=False)
            n = len(report["compressed"])
            ratio = report["total_plain"] / max(1, report["total_gz"])
            print(f"    {n} files: {report['total_plain']//1024//1024} MB → "
                  f"{report['total_gz']//1024//1024} MB  (×{ratio:.1f})")

        # Already-gzipped files: copy as-is
        for gz in gz_files:
            shutil.copy2(gz, staged_gd / gz.name)

        # Metadata + subcircuits at top level of the run dir
        shutil.copy2(src_md, staging / "graph-metadata.json")
        if sc_json.exists():
            shutil.copy2(sc_json, staging / "subcircuits.json")

        # Write a run_info.json describing this bundle
        run_info_path = run_dir / "run_config.json"
        if run_info_path.exists():
            run_info = json.loads(run_info_path.read_text())
        else:
            run_info = {}
        run_info["run_name"] = run_name
        run_info["n_graph_files"] = len(list(staged_gd.glob("*.json.gz")))
        run_info["bundled_subcircuits"] = sc_json.exists()
        run_info["source"] = str(run_dir)
        (staging / "run_info.json").write_text(json.dumps(run_info, indent=2))

        total_staged = sum(f.stat().st_size for f in staging.rglob("*") if f.is_file())
        print(f"\n  Staged bundle: {total_staged // 1024 // 1024} MB at {staging}")

        if args.dry_run:
            print(f"  [--dry-run] not uploading")
            print(f"  Inspect: ls -la {staging}")
            keep_dir = staging  # hold it open
            return

        # Ensure repo exists
        try:
            create_repo(args.dataset_repo, repo_type="dataset", exist_ok=True)
            print(f"  Repo ready: https://huggingface.co/datasets/{args.dataset_repo}")
        except Exception as e:
            print(f"\nERROR creating/accessing dataset repo '{args.dataset_repo}': {e}")
            print(f"  If this is a 403, your token lacks write permission on that "
                  f"namespace.")
            print(f"  Fix: see options in the preflight-auth error message, or create "
                  f"the repo manually at")
            print(f"       https://huggingface.co/new-dataset")
            sys.exit(1)

        print(f"\n  Uploading to runs/{run_name}/ ...")
        api.upload_folder(
            folder_path=str(staging),
            repo_id=args.dataset_repo,
            repo_type="dataset",
            path_in_repo=f"runs/{run_name}",
            commit_message=f"push graph_data for {run_name}",
        )
        print(f"\nDONE! Browse: https://huggingface.co/datasets/{args.dataset_repo}/tree/main/runs/{run_name}")
        print(f"\nCollaborators pull with:")
        print(f"  python3 fetch_graph_data.py --run {run_name}")
    finally:
        if keep_dir is None and 'staging' in locals():
            shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
