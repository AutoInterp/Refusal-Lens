"""
Stage 05: Build the attribution-graph browser frontend
=======================================================
End-to-end orchestrator that takes .pt attribution graphs from Stage 02
and produces a browsable, annotated frontend tree with:

  - per-graph JSON in the vendored circuit-tracer format
  - overlap annotations (shared_with_bare / jb_unique / bare)
  - subcircuit membership annotations from Stage 07
  - auto-generated graph-metadata.json (via create_graph_files)
  - staged vendor frontend + patch files (overlap coloring, subcircuit panel,
    side-by-side compare.html, fetch-override)
  - optional gzip compression of the final graph_data/ tree

Inputs:
  --run-dir <run>              Run directory containing 02_attribution/graphs/*.pt
  --subcircuits-run <run>      (optional) Different run to pull 07_subcircuits/subcircuits.json
                                from; defaults to --run-dir. Useful when annotating a small
                                local run with subcircuits derived from a 50-prompt RunPod run.
  --prompts 0,1,2              (optional) Subset of prompt indices
  --classes bare,fiction       (optional) Subset of conditions (default: all 6)
  --out-dir <path>             (optional) Frontend output (default: <run_dir>/05_frontend)
  --skip-convert               Reuse existing graph_data/*.json (skip .pt conversion)
  --skip-overlap               Skip overlap annotation
  --skip-subcircuits           Skip subcircuit annotation
  --gzip                       Gzip final graph_data/ after staging
  --node-threshold 0.8         Pruning node threshold (higher = keep more)
  --edge-threshold 0.98        Pruning edge threshold

Outputs:
  <out_dir>/
      index.html, compare.html, util.js, style.css, ...   (vendor)
      overlap-colors.css, overlap-annotate.js,
      subcircuit-panel.css, subcircuit-panel.js,
      fetch-override.js                                    (patches)
      data/graph-metadata.json                             (index of graphs)
      graph_data/<slug>.json [or .json.gz]                 (per-graph data)

Usage:
    # From scripts/pipeline/:
    PYTHONPATH=src python3 05_visualize_circuits.py \\
        --run-dir ../../data/results/pipeline_runs/run_20260418_172402 \\
        --subcircuits-run ../../data/results/pipeline_runs/run_20260417_010035

    # Then browse:
    cd ../../data/results/pipeline_runs/run_20260418_172402/05_frontend
    python3 -m http.server 8000
    # open http://localhost:8000/           (single-graph viewer)
    # open http://localhost:8000/compare.html  (side-by-side bare ↔ JB)
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: F401
from utils_viz import (
    annotate_bare,
    annotate_overlap,
    annotate_subcircuits,
    convert_pt_to_frontend_json,
    gzip_json_files,
    stage_frontend,
)

DEFAULT_CLASSES = ("bare", "roleplay", "fiction", "analytical", "completion", "cognitive_reframe")


def parse_args():
    p = argparse.ArgumentParser(description="Stage 05: build annotated attribution-graph frontend")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--subcircuits-run", type=Path, default=None,
                   help="Run directory to pull 07_subcircuits/subcircuits.json from "
                        "(default: --run-dir)")
    p.add_argument("--prompts", type=str, default=None,
                   help="Comma-separated prompt indices (default: all)")
    p.add_argument("--classes", type=str, default=None,
                   help=f"Comma-separated condition names (default: all 6 = {','.join(DEFAULT_CLASSES)})")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Frontend output directory (default: <run-dir>/05_frontend)")
    p.add_argument("--skip-convert", action="store_true")
    p.add_argument("--skip-overlap", action="store_true")
    p.add_argument("--skip-subcircuits", action="store_true")
    p.add_argument("--gzip", action="store_true")
    p.add_argument("--node-threshold", type=float, default=0.8)
    p.add_argument("--edge-threshold", type=float, default=0.98)
    return p.parse_args()


def select_pt_files(
    graphs_dir: Path,
    prompt_filter: set[int] | None,
    class_filter: set[str] | None,
) -> list[Path]:
    """Return sorted .pt files matching the filters. Expects `{idx:03d}_{class}.pt`."""
    selected = []
    for pt in sorted(graphs_dir.glob("*.pt")):
        parts = pt.stem.split("_", 1)
        if len(parts) != 2:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        cls = parts[1]
        if prompt_filter is not None and idx not in prompt_filter:
            continue
        if class_filter is not None and cls not in class_filter:
            continue
        selected.append(pt)
    return selected


def group_by_prompt(json_paths: dict[str, Path]) -> dict[int, dict[str, Path]]:
    """Group {slug: path} by prompt index. Slug = `{idx:03d}_{class}`."""
    by_prompt: dict[int, dict[str, Path]] = defaultdict(dict)
    for slug, path in json_paths.items():
        idx_str, cls = slug.split("_", 1)
        by_prompt[int(idx_str)][cls] = path
    return by_prompt


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    sc_run_dir = (args.subcircuits_run or run_dir).resolve()
    out_dir = (args.out_dir or (run_dir / "05_frontend")).resolve()

    # Source paths
    graphs_pt_dir = run_dir / "02_attribution" / "graphs"
    subcircuits_json = sc_run_dir / "07_subcircuits" / "subcircuits.json"

    # Staging: convert outputs go into <out_dir>/graph_data/ directly.
    # stage_frontend() will copy vendor + patches around it.
    graph_data_dir = out_dir / "graph_data"

    # Parse filters
    prompt_filter = (
        {int(p) for p in args.prompts.split(",")} if args.prompts else None
    )
    class_filter = (
        set(args.classes.split(",")) if args.classes else None
    )

    print("=" * 60)
    print("STAGE 05: Frontend orchestration")
    print("=" * 60)
    print(f"  run_dir:            {run_dir}")
    print(f"  subcircuits_run:    {sc_run_dir}")
    print(f"  out_dir:            {out_dir}")
    print(f"  graphs source:      {graphs_pt_dir}")
    print(f"  subcircuits source: {subcircuits_json}"
          f"{'  [MISSING]' if not subcircuits_json.exists() else ''}")
    print(f"  prompt filter:      {sorted(prompt_filter) if prompt_filter else 'all'}")
    print(f"  class filter:       {sorted(class_filter) if class_filter else 'all'}")

    if not graphs_pt_dir.exists():
        print(f"\n  ERROR: {graphs_pt_dir} does not exist. "
              f"Did Stage 02 run with --save-graphs?")
        sys.exit(1)

    pt_files = select_pt_files(graphs_pt_dir, prompt_filter, class_filter)
    if not pt_files:
        print("\n  ERROR: no .pt files match the filters.")
        sys.exit(1)
    print(f"\n  Selected {len(pt_files)} .pt files")

    # ------------------------------------------------------------------
    # Step 1: convert .pt → graph_data/<slug>.json
    # (create_graph_files auto-appends to graph-metadata.json in the same dir)
    # ------------------------------------------------------------------
    json_paths: dict[str, Path] = {}
    if args.skip_convert:
        print("\n  Step 1: skipping .pt → JSON conversion (--skip-convert)")
        for pt in pt_files:
            slug = pt.stem
            jpath = graph_data_dir / f"{slug}.json"
            if not jpath.exists():
                print(f"    WARN: expected {jpath.name} not found")
                continue
            json_paths[slug] = jpath
    else:
        graph_data_dir.mkdir(parents=True, exist_ok=True)
        # Nuke any stale graph-metadata.json so we rebuild cleanly
        stale_metadata = graph_data_dir / "graph-metadata.json"
        if stale_metadata.exists():
            stale_metadata.unlink()

        print(f"\n  Step 1: Converting {len(pt_files)} .pt → JSON...")
        t0 = time.time()
        for i, pt in enumerate(pt_files, 1):
            slug = pt.stem
            try:
                jpath = convert_pt_to_frontend_json(
                    pt, slug, graph_data_dir,
                    node_threshold=args.node_threshold,
                    edge_threshold=args.edge_threshold,
                )
                json_paths[slug] = jpath
                print(f"    [{i}/{len(pt_files)}] {slug}.json ({jpath.stat().st_size // 1024} KB)")
            except Exception as e:
                print(f"    [{i}/{len(pt_files)}] {slug} FAILED: {e}")
        print(f"    Conversion took {time.time() - t0:.1f}s")

    # Sanity: need graph-metadata.json for the frontend
    metadata_path = graph_data_dir / "graph-metadata.json"
    if not metadata_path.exists():
        print(f"\n  ERROR: {metadata_path} not created. "
              f"Conversion may have failed on all graphs.")
        sys.exit(1)
    print(f"  graph-metadata.json: {metadata_path.stat().st_size} bytes")

    # ------------------------------------------------------------------
    # Step 2: annotate overlap (bare + JB vs bare per prompt)
    # ------------------------------------------------------------------
    by_prompt = group_by_prompt(json_paths)
    if args.skip_overlap:
        print("\n  Step 2: skipping overlap annotation (--skip-overlap)")
    else:
        print(f"\n  Step 2: Annotating overlap for {len(by_prompt)} prompts...")
        n_bare, n_jb, n_skip = 0, 0, 0
        for idx, conds in sorted(by_prompt.items()):
            bare_path = conds.get("bare")
            if bare_path is not None:
                annotate_bare(bare_path, idx)
                n_bare += 1
            for cls, jb_path in conds.items():
                if cls == "bare":
                    continue
                if bare_path is None:
                    print(f"    WARN: no bare for prompt {idx:03d}, skipping {cls}")
                    n_skip += 1
                    continue
                annotate_overlap(jb_path, bare_path, cls, idx)
                n_jb += 1
        print(f"    Annotated {n_bare} bare + {n_jb} JB graphs (skipped {n_skip})")

    # ------------------------------------------------------------------
    # Step 3: annotate subcircuits (Stage 07 memberships)
    # ------------------------------------------------------------------
    if args.skip_subcircuits:
        print("\n  Step 3: skipping subcircuit annotation (--skip-subcircuits)")
    elif not subcircuits_json.exists():
        print(f"\n  Step 3: {subcircuits_json.name} not found — skipping subcircuit annotation")
        print(f"          (run Stage 07 first, or pass --subcircuits-run <run-with-07>)")
    else:
        print(f"\n  Step 3: Annotating subcircuits from {subcircuits_json.relative_to(sc_run_dir.parent)}...")
        n_ok = 0
        for idx, conds in sorted(by_prompt.items()):
            for cls, path in conds.items():
                result = annotate_subcircuits(path, subcircuits_json)
                n_ok += 1
                if idx == sorted(by_prompt)[0] and cls == "bare":
                    count = result["metadata"].get("n_subcircuit_annotated", 0)
                    total_nodes = len(result.get("nodes", []))
                    print(f"    sample (prompt {idx:03d}/bare): "
                          f"{count}/{total_nodes} nodes tagged")
        print(f"    Annotated subcircuits on {n_ok} graphs")

    # ------------------------------------------------------------------
    # Step 4 (optional): gzip compression BEFORE staging so that the
    # subsequent stage_frontend() injects the USE_GZIP flag + gzip-fetch.js
    # ------------------------------------------------------------------
    if args.gzip:
        print(f"\n  Step 4a: Gzipping {graph_data_dir}...")
        report = gzip_json_files(graph_data_dir, keep_plain=False)
        n = len(report["compressed"])
        ratio = (report["total_plain"] / report["total_gz"]) if report["total_gz"] else 0
        print(f"    {n} files: {report['total_plain'] // 1024 // 1024} MB → "
              f"{report['total_gz'] // 1024 // 1024} MB  (×{ratio:.2f})")

    # ------------------------------------------------------------------
    # Step 4b: stage the frontend (vendor + patches + graph_data)
    # ------------------------------------------------------------------
    print(f"\n  Step 4b: Staging frontend at {out_dir}...")
    stage_frontend(graph_data_dir, out_dir, use_gzip=args.gzip)
    print(f"    Vendor + patches copied"
          + (" (gzip mode: client-side decompression via DecompressionStream)"
             if args.gzip else ""))

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"  Frontend:     {out_dir}/index.html")
    print(f"  Compare:      {out_dir}/compare.html")
    print(f"  Serve local:  cd {out_dir} && python3 -m http.server 8000")


if __name__ == "__main__":
    main()
