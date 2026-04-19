"""
Rebuild graph-metadata.json from existing *.json or *.json.gz files.

Use this when graph-metadata.json is stale or missing but the per-graph
data files are intact. Reads each graph's own `metadata` block and
concatenates them into the index the frontend loads at page load.

Usage:
    python3 rebuild_graph_metadata.py \\
        --graph-data-dir ../../data/results/pipeline_runs/run_20260418_172402/05_frontend/graph_data
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Rebuild graph-metadata.json from graph files")
    p.add_argument("--graph-data-dir", type=Path, required=True)
    return p.parse_args()


def load_graph(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


def main():
    args = parse_args()
    gd = args.graph_data_dir.resolve()
    if not gd.is_dir():
        print(f"ERROR: not a directory: {gd}")
        sys.exit(1)

    files = sorted(list(gd.glob("*.json.gz")) + list(gd.glob("*.json")))
    files = [f for f in files if f.name != "graph-metadata.json"]
    if not files:
        print(f"ERROR: no graph files found in {gd}")
        sys.exit(1)

    graphs = []
    for f in files:
        try:
            g = load_graph(f)
            md = g.get("metadata")
            if md is None:
                print(f"  WARN: {f.name} has no 'metadata' block; skipping")
                continue
            graphs.append(md)
        except Exception as e:
            print(f"  WARN: {f.name} failed to parse ({e}); skipping")

    out = gd / "graph-metadata.json"
    out.write_text(json.dumps({"graphs": graphs}, indent=2))
    print(f"Wrote {out}")
    print(f"  entries: {len(graphs)}")
    print(f"  size:    {out.stat().st_size} bytes")


if __name__ == "__main__":
    main()
