"""Assemble an N-column attribution-graph compare site.

Fetches each configured run (Gemma variants + Qwen) from the HF dataset into
one parent dir, builds a compare_manifest.json (shared prompt x condition
options + per-column slug maps), and stages the manifest-driven compare harness.
Pure functions (parse_condition, build_compare_manifest) are unit-tested; the
CLI (main) does fetch + filesystem assembly.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PATCHES = Path(__file__).resolve().parent / "05_frontend_patches"
_SLUG_RE = re.compile(r"^(\d+)_(.+)$")

# Uniform gridsnap layout baked into every column's graphs so the per-column
# subviews stack consistently for side-by-side reading: attribution graph on top
# (full width), then Input/Output features, then Token Predictions, then Subgraph.
# Format is circuit-tracer's serializeGrid: "<class><xxyyWWhh>" (each 2 digits),
# joined by "_". maxX is 14 (from init-cg.js default gridData), so w=14 = full width.
# initGridsnap reads this from each graph's qParams.gridsnap (init-cg.js).
STACK_GRIDSNAP = ("link-graph00001410_node-connections00101405"
                  "_feature-detail00151407_subgraph00221403")


def parse_condition(slug: str, mode_suffixes=("single", "multi")):
    """Return (idx, condition) with any trailing mode suffix stripped, or None."""
    m = _SLUG_RE.match(slug)
    if not m:
        return None
    idx, cond = m.group(1), m.group(2)
    parts = cond.split("_")
    if len(parts) > 1 and parts[-1] in mode_suffixes:
        cond = "_".join(parts[:-1])
    return idx, cond


def build_compare_manifest(columns: list[dict], title: str) -> dict:
    """columns: [{label, dir, model, target, graphs:[{slug, prompt}]}].
    Offers only (idx, cond) present in ALL columns; resolves each column's
    actual slug per (idx, cond)."""
    per_col = []
    for c in columns:
        smap, ptext = {}, {}
        for g in c["graphs"]:
            pc = parse_condition(g["slug"])
            if not pc:
                continue
            idx, cond = pc
            smap[f"{idx}_{cond}"] = g["slug"]
            ptext.setdefault(idx, g.get("prompt", "") or "")
        per_col.append({**c, "smap": smap, "ptext": ptext})

    keysets = [set(col["smap"].keys()) for col in per_col]
    shared = sorted(set.intersection(*keysets)) if keysets else []
    idxs = sorted({k.split("_", 1)[0] for k in shared}, key=int)
    conds = sorted({k.split("_", 1)[1] for k in shared})

    ptext0 = per_col[0]["ptext"] if per_col else {}
    return {
        "title": title,
        "columns": [{"label": col["label"], "dir": col["dir"], "model": col["model"],
                     "target": col["target"],
                     "slugmap": {k: col["smap"][k] for k in shared}}
                    for col in per_col],
        "prompts": [{"idx": i, "text": ptext0.get(i, "")} for i in idxs],
        "conditions": conds,
    }


def _fetch_run(run: str, dataset_repo: str, out_base: Path) -> None:
    """Shell out to fetch_graph_data.py so each column is a self-contained viewer.

    Each run is ~551 small files; HF's Xet backend requests a read-token per file,
    which blows the 1000-requests/5-min API quota across 4 runs. Disable Xet
    (regular CDN download path) to cut the per-file token requests, and pass the
    HF token through so we get the authenticated (higher) rate-limit tier.
    """
    env = dict(os.environ)
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    cmd = [sys.executable, str(Path(__file__).resolve().parent / "fetch_graph_data.py"),
           "--run", run, "--dataset-repo", dataset_repo, "--out-base", str(out_base)]
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def _load_column_graphs(run_dir: Path) -> list[dict]:
    """Read a fetched column's data/graph-metadata.json -> [{slug, prompt}]."""
    md = run_dir / "05_frontend" / "data" / "graph-metadata.json"
    meta = json.loads(md.read_text())
    return [{"slug": g["slug"], "prompt": g.get("prompt", "")} for g in meta["graphs"]]


def _apply_gridsnap_layout(run_dir: Path, layout: str) -> int:
    """Bake qParams.gridsnap=<layout> into every packed graph JSON of a column so
    all columns render the same stacked subview layout. Idempotent; handles both
    .json and .json.gz. Returns the number of graphs updated."""
    gd = run_dir / "05_frontend" / "graph_data"
    n = 0
    for f in sorted(gd.glob("*.json*")):
        if f.name == "graph-metadata.json":
            continue
        if f.suffix == ".gz":
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                d = json.load(fh)
            d.setdefault("qParams", {})["gridsnap"] = layout
            with gzip.open(f, "wt", encoding="utf-8") as fh:
                json.dump(d, fh)
        else:
            d = json.loads(f.read_text())
            d.setdefault("qParams", {})["gridsnap"] = layout
            f.write_text(json.dumps(d))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description="Assemble the N-column compare site")
    ap.add_argument("--config", type=Path, default=PATCHES / "compare_config.json")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[2] / "data/results/compare_3way")
    ap.add_argument("--skip-fetch", action="store_true",
                    help="Reuse already-fetched columns under --out (no HF download). "
                         "Combine with the default gridsnap baking to re-layout columns "
                         "already on disk without re-downloading.")
    ap.add_argument("--no-gridsnap", action="store_true",
                    help="Do not bake the uniform stacked subview layout into the graphs "
                         "(leave circuit-tracer's default graph-left layout).")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)

    columns = []
    for col in cfg["columns"]:
        run = col["run"]
        run_dir = args.out / run
        md_path = run_dir / "05_frontend" / "data" / "graph-metadata.json"
        # Idempotent/resumable: a run that already finished fetching (its
        # graph-metadata.json exists) is skipped, so re-running after an HF
        # rate-limit pause only fetches what's still missing.
        if not args.skip_fetch and not md_path.exists():
            try:
                _fetch_run(run, cfg["dataset_repo"], args.out)
            except subprocess.CalledProcessError:
                print(f"\nERROR: fetch for '{run}' failed (often HF 429 rate limit — "
                      f"1000 requests/5 min). Cached progress is saved. Wait ~5 minutes "
                      f"and re-run this script: already-fetched runs are skipped and the "
                      f"partial run resumes from cache. Ensure HF_TOKEN is exported for the "
                      f"higher authenticated limit.")
                sys.exit(1)
        elif md_path.exists():
            print(f"  [{run}] already fetched (graph-metadata.json present); skipping.")
        if not md_path.exists():
            print(f"ERROR: no graph-metadata for column '{run}' under {run_dir}")
            sys.exit(1)
        columns.append({"label": col["label"], "dir": f"{run}/05_frontend",
                        "model": col["model"], "target": col["target"],
                        "graphs": _load_column_graphs(run_dir)})
        if not args.no_gridsnap:
            n = _apply_gridsnap_layout(run_dir, STACK_GRIDSNAP)
            print(f"  [{run}] applied stacked gridsnap layout to {n} graphs")

    manifest = build_compare_manifest(columns, title=cfg.get("title", "compare"))
    (args.out / "compare_manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.copy2(PATCHES / "compare_multi.html", args.out / "compare.html")

    print(f"\nAssembled {len(columns)} columns; "
          f"{len(manifest['prompts'])} shared prompts x {len(manifest['conditions'])} conditions.")
    print(f"Serve:\n  cd {args.out}\n  python3 -m http.server 8000")
    print(f"  open http://localhost:8000/compare.html")


if __name__ == "__main__":
    main()
