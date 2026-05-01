"""
Fetch a pipeline run's graph_data + subcircuits from the HuggingFace dataset
and set up a local, browsable frontend. Collaborator-facing entry point.

Produces:
    data/results/pipeline_runs/<run>/05_frontend/
        index.html, compare.html, util.js, ...           (vendor)
        overlap-colors.css, subcircuit-panel.css,
        overlap-annotate.js, subcircuit-panel.js,
        gzip-fetch.js, fetch-override.js                 (patches)
        graph_data/<slug>.json.gz                        (per-graph, compressed)
        data/graph-metadata.json                         (graph index)

Usage:
    # No HF auth required for public dataset pulls
    pip install huggingface_hub

    python3 fetch_graph_data.py --list                                # list available runs
    python3 fetch_graph_data.py --run run_20260418_172402             # fetch one run

Then serve locally:
    cd ../../data/results/pipeline_runs/run_20260418_172402/05_frontend
    python3 -m http.server 8000
    # open http://localhost:8000/                      (single-graph viewer)
    # open http://localhost:8000/compare.html          (side-by-side bare ↔ JB)

Incremental updates: re-run `fetch_graph_data.py --run <run>` to pick up new
files; huggingface_hub skips files whose content hash already matches disk.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from utils_viz import VENDOR_FRONTEND, stage_frontend

DEFAULT_DATASET_REPO = "AutoInterp/refusal-lens-graphs"


def check_vendor_submodule() -> None:
    """Fail fast if vendor/circuit-tracer submodule isn't populated.

    `git clone` without `--recurse-submodules` leaves the directory empty,
    and a cryptic FileNotFoundError surfaces 15 minutes into the flow.
    This preflight bails immediately with the one command that fixes it.
    """
    if VENDOR_FRONTEND.is_dir() and any(VENDOR_FRONTEND.iterdir()):
        return
    repo_root = config.REPO_ROOT
    print(f"ERROR: vendor frontend not found at {VENDOR_FRONTEND}")
    print(f"       This directory is a git submodule (vendor/circuit-tracer).")
    print(f"       If you cloned without --recurse-submodules, populate it now:")
    print(f"")
    print(f"           cd {repo_root}")
    print(f"           git submodule update --init --recursive")
    print(f"")
    print(f"       Or re-clone with: git clone --recurse-submodules <repo-url>")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description="Fetch graph data from HF dataset")
    p.add_argument("--run", type=str, default=None,
                   help="Run name (e.g. run_20260418_172402). Omit with --list.")
    p.add_argument("--list", action="store_true",
                   help="List available runs and exit")
    p.add_argument("--dataset-repo", type=str, default=DEFAULT_DATASET_REPO)
    p.add_argument("--out-base", type=Path, default=None,
                   help="Override local base (default: data/results/pipeline_runs)")
    return p.parse_args()


def list_runs(api, dataset_repo: str) -> list[str]:
    files = api.list_repo_files(repo_id=dataset_repo, repo_type="dataset")
    runs = set()
    for f in files:
        parts = f.split("/")
        if len(parts) >= 2 and parts[0] == "runs":
            runs.add(parts[1])
    return sorted(runs)


def main():
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run `pip install huggingface_hub`.")
        sys.exit(1)

    args = parse_args()
    api = HfApi()

    if args.list:
        try:
            runs = list_runs(api, args.dataset_repo)
        except Exception as e:
            print(f"ERROR listing {args.dataset_repo}: {e}")
            sys.exit(1)
        if not runs:
            print(f"(no runs found in {args.dataset_repo})")
            return
        print(f"Available runs in {args.dataset_repo}:")
        for r in runs:
            print(f"  {r}")
        return

    if not args.run:
        print("ERROR: --run required (or use --list)")
        sys.exit(1)

    out_base = args.out_base or (config.REPO_ROOT / "data" / "results" / "pipeline_runs")
    out_base.mkdir(parents=True, exist_ok=True)
    run_dir = out_base / args.run
    out_dir = run_dir / "05_frontend"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Preflight: vendor frontend must exist before we start downloading.
    # Avoids a 180-MB download that fails at the final stage_frontend() call.
    check_vendor_submodule()

    print("=" * 60)
    print(f"Fetching run '{args.run}' from {args.dataset_repo}")
    print("=" * 60)

    # Narrow allow_patterns so we ONLY pull frontend artifacts (~180 MB).
    # The raw `.pt` graphs under `runs/<run>/raw_graphs/` are archived for
    # deep analysis via fetch_raw_graphs.py — they must not come down here.
    try:
        snapshot_path = snapshot_download(
            repo_id=args.dataset_repo,
            repo_type="dataset",
            allow_patterns=[
                f"runs/{args.run}/graph_data/*.json",
                f"runs/{args.run}/graph_data/*.json.gz",
                f"runs/{args.run}/graph-metadata.json",
                f"runs/{args.run}/subcircuits.json",
                f"runs/{args.run}/run_info.json",
            ],
        )
    except Exception as e:
        print(f"ERROR downloading: {e}")
        print(f"If the dataset is private, run `hf auth login` first.")
        sys.exit(1)

    src_run = Path(snapshot_path) / "runs" / args.run
    if not src_run.exists():
        print(f"ERROR: no files found for run '{args.run}' in {args.dataset_repo}")
        print(f"Use `python3 fetch_graph_data.py --list` to see available runs.")
        sys.exit(1)

    # Place graph_data/*.json.gz into out_dir/graph_data
    graph_data_dst = out_dir / "graph_data"
    graph_data_dst.mkdir(exist_ok=True)
    src_gd = src_run / "graph_data"
    n_graphs = 0
    if src_gd.exists():
        for f in src_gd.iterdir():
            shutil.copy2(f, graph_data_dst / f.name)
            n_graphs += 1
    print(f"  Graphs copied: {n_graphs}")

    # graph-metadata.json → graph_data/ so stage_frontend picks it up
    md = src_run / "graph-metadata.json"
    if md.exists():
        shutil.copy2(md, graph_data_dst / "graph-metadata.json")
        print(f"  graph-metadata.json: copied")
    else:
        print(f"  WARN: no graph-metadata.json in bundle")

    # subcircuits.json goes next to 07_subcircuits/ if present — so the frontend
    # panel loads memberships from the same tree the pipeline produces.
    sc_src = src_run / "subcircuits.json"
    if sc_src.exists():
        sc_dir = run_dir / "07_subcircuits"
        sc_dir.mkdir(exist_ok=True)
        shutil.copy2(sc_src, sc_dir / "subcircuits.json")
        print(f"  subcircuits.json: copied into 07_subcircuits/")

    # Detect gzip: if we pulled .json.gz files, tell stage_frontend to inject
    # the gzip-fetch patch + USE_GZIP flag.
    has_gzip = any(graph_data_dst.glob("*.json.gz"))
    has_plain = any(
        f.name != "graph-metadata.json" and f.suffix == ".json"
        for f in graph_data_dst.iterdir()
    )

    print(f"\n  Staging frontend (gzip mode: {has_gzip})...")
    stage_frontend(graph_data_dir=graph_data_dst, frontend_out=out_dir, use_gzip=has_gzip)

    if has_gzip and has_plain:
        print(f"  NOTE: bundle contains both .json and .json.gz; gzip-fetch will prefer .gz")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"Serve locally:")
    print(f"  cd {out_dir}")
    print(f"  python3 -m http.server 8000")
    print(f"")
    print(f"Then open:")
    print(f"  http://localhost:8000/                      (single-graph viewer)")
    print(f"  http://localhost:8000/compare.html          (side-by-side bare ↔ JB)")


if __name__ == "__main__":
    main()
