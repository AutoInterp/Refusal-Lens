"""
Stage 02c: Pack raw .pt attribution graphs into frontend-ready gzipped JSONs.
=============================================================================
Producer-side step that decouples the expensive `.pt → JSON` conversion
(requires circuit-tracer + significant RAM) from the Stage 05 annotation +
frontend staging pass. After this runs, the output can be:

  (a) pushed to HF via `push_graph_data.py --source 02c` so collaborators
      pull the ~2–5 GB compressed bundle instead of the ~80 GB raw .pt set,
  (b) fed into Stage 05 via `05_visualize_circuits.py --skip-convert` to
      do just overlap + subcircuit annotation + frontend staging.

Inputs:
    <run-dir>/02_attribution/graphs/*.pt    (from Stage 02 with --save-graphs)

Outputs:
    <run-dir>/graph_data/<slug>.json.gz     (per-graph, optionally plain)
    <run-dir>/graph_data/graph-metadata.json
    <run-dir>/graph_data/pack_report.json   (timing + file sizes + ratios)

Usage:
    # Convert + gzip (default)
    python3 02c_pack_graphs.py --run-dir <run>

    # Keep plain JSONs alongside .gz (useful for local debugging)
    python3 02c_pack_graphs.py --run-dir <run> --no-gzip

    # Filter to a subset (smoke test)
    python3 02c_pack_graphs.py --run-dir <run> --prompts 0,1,2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: F401
from utils_viz import convert_pt_to_frontend_json, gzip_json_files


def parse_args():
    p = argparse.ArgumentParser(description="Convert Stage 02 .pt graphs to gzipped frontend JSONs")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--prompts", type=str, default=None,
                   help="Comma-separated prompt indices (default: all)")
    p.add_argument("--classes", type=str, default=None,
                   help="Comma-separated condition names (default: all)")
    p.add_argument("--node-threshold", type=float, default=0.8)
    p.add_argument("--edge-threshold", type=float, default=0.98)
    p.add_argument("--no-gzip", action="store_true",
                   help="Keep plain JSONs only (skip gzip compression)")
    p.add_argument("--keep-plain", action="store_true",
                   help="Gzip but also keep the plain JSONs (default: delete plain after gzip)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-convert even if the target JSON already exists")
    return p.parse_args()


def select_pt_files(
    graphs_dir: Path,
    prompt_filter: set[int] | None,
    class_filter: set[str] | None,
) -> list[Path]:
    """Return sorted .pt files matching the filters. Handles both legacy and
    new slug formats — conversion itself doesn't care about mode/cond splitting."""
    selected = []
    for pt in sorted(graphs_dir.glob("*.pt")):
        parts = pt.stem.split("_", 1)
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        tail = parts[1]
        # For class_filter, check if any token matches — tolerates both "fiction"
        # legacy and "jb_fiction_multi" new-schema slugs.
        if prompt_filter is not None and idx not in prompt_filter:
            continue
        if class_filter is not None and not any(c in tail for c in class_filter):
            continue
        selected.append(pt)
    return selected


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    graphs_dir = run_dir / "02_attribution" / "graphs"
    out_dir = run_dir / "graph_data"

    if not graphs_dir.exists():
        print(f"ERROR: {graphs_dir} does not exist.")
        print(f"  Stage 02 must run with --save-graphs. If graphs are on HF,")
        print(f"  fetch them first: python3 fetch_raw_graphs.py --run {run_dir.name}")
        sys.exit(1)

    prompt_filter = (
        {int(p) for p in args.prompts.split(",")} if args.prompts else None
    )
    class_filter = set(args.classes.split(",")) if args.classes else None

    pt_files = select_pt_files(graphs_dir, prompt_filter, class_filter)
    if not pt_files:
        print(f"ERROR: no .pt files matched (prompts={prompt_filter}, classes={class_filter})")
        sys.exit(1)

    print("=" * 60)
    print(f"STAGE 02c: Packing graphs for {run_dir.name}")
    print("=" * 60)
    print(f"  Source:          {graphs_dir}")
    print(f"  Out:             {out_dir}")
    print(f"  N .pt files:     {len(pt_files)}")
    print(f"  Thresholds:      node={args.node_threshold}  edge={args.edge_threshold}")
    print(f"  Gzip:            {'no' if args.no_gzip else 'yes'}")

    out_dir.mkdir(parents=True, exist_ok=True)
    # Nuke any stale graph-metadata.json so it's rebuilt cleanly by
    # create_graph_files (which appends to the file on each call).
    stale_md = out_dir / "graph-metadata.json"
    if stale_md.exists() and args.overwrite:
        stale_md.unlink()

    t0 = time.time()
    n_ok = n_skip = n_fail = 0
    failures: list[str] = []
    for i, pt in enumerate(pt_files, 1):
        slug = pt.stem
        json_path = out_dir / f"{slug}.json"
        gz_path = out_dir / f"{slug}.json.gz"
        if not args.overwrite and (json_path.exists() or gz_path.exists()):
            n_skip += 1
            continue
        try:
            convert_pt_to_frontend_json(
                pt, slug, out_dir,
                node_threshold=args.node_threshold,
                edge_threshold=args.edge_threshold,
            )
            n_ok += 1
            if i % 10 == 0 or i == len(pt_files):
                elapsed = time.time() - t0
                print(f"  [{i}/{len(pt_files)}] converted ({elapsed:.1f}s elapsed, "
                      f"{elapsed / max(i, 1):.2f}s/graph avg)")
        except Exception as e:
            n_fail += 1
            failures.append(f"{slug}: {e}")
            print(f"  [{i}/{len(pt_files)}] {slug} FAILED: {e}")

    print(f"\n  Converted: {n_ok} ok, {n_skip} skipped (already existed), "
          f"{n_fail} failed  (wall: {time.time() - t0:.1f}s)")

    if not (out_dir / "graph-metadata.json").exists():
        print(f"\n  ERROR: graph-metadata.json was not produced (no successful conversions)")
        if failures:
            print("  Failures:")
            for f in failures[:10]:
                print(f"    {f}")
        sys.exit(1)

    report = {
        "run_name": run_dir.name,
        "n_pt_files": len(pt_files),
        "n_converted": n_ok,
        "n_skipped": n_skip,
        "n_failed": n_fail,
        "failures": failures[:20],
        "convert_seconds": round(time.time() - t0, 2),
        "gzipped": not args.no_gzip,
    }

    if not args.no_gzip:
        print(f"\n  Gzipping {out_dir} ...")
        gz_report = gzip_json_files(out_dir, keep_plain=args.keep_plain)
        n = len(gz_report["compressed"])
        ratio = gz_report["total_plain"] / max(1, gz_report["total_gz"])
        print(f"    {n} files: {gz_report['total_plain']//1024//1024} MB → "
              f"{gz_report['total_gz']//1024//1024} MB  (×{ratio:.1f})")
        report["gzip"] = {
            "n_compressed": n,
            "total_plain_bytes": gz_report["total_plain"],
            "total_gz_bytes": gz_report["total_gz"],
            "ratio": round(ratio, 2),
            "skipped": gz_report.get("skipped", []),
        }

    # Write the report OUTSIDE graph_data/ so push_graph_data.py doesn't upload it
    # along with the per-graph JSONs.
    (run_dir / "02c_pack_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nDONE! Packed bundle at {out_dir}/")
    print(f"Next step (push to HF):")
    print(f"  python3 scripts/pipeline/push_graph_data.py --run-dir {run_dir} --source 02c")


if __name__ == "__main__":
    main()
