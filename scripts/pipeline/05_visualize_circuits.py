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
    annotate_ctrl,
    annotate_overlap,
    annotate_overlap_3way,
    annotate_subcircuits,
    convert_pt_to_frontend_json,
    gzip_json_files,
    stage_frontend,
)

# Legacy default class list (used only when user doesn't pass --classes and all
# slugs are in the old flat format). New-schema runs use cond_names like
# bare / jb_fiction / ctrl_fiction — picked up automatically from the .pt names.
DEFAULT_CLASSES = ("bare", "roleplay", "fiction", "analytical", "completion", "cognitive_reframe")
GRAPH_MODES = ("multi", "single")


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
    p.add_argument("--source-graph-data", type=Path, default=None,
                   help="When --skip-convert is set, read pre-converted JSONs from "
                        "this directory (e.g. <run>/graph_data produced by 02c_pack_graphs.py). "
                        "If omitted, falls back to <out-dir>/graph_data.")
    p.add_argument("--node-threshold", type=float, default=0.8)
    p.add_argument("--edge-threshold", type=float, default=0.98)
    p.add_argument("--mode", type=str, choices=["multi", "single", "both"], default="single",
                   help="Which graph mode to stage for the new-schema two-graph scheme "
                        "(multi = template-anchors [-5,-3,-2], single = causally-verified [-2]). "
                        "'both' stages each mode side-by-side with distinct slugs. "
                        "Ignored for legacy runs without mode-suffixed .pt files.")
    return p.parse_args()


def parse_slug(stem: str) -> tuple[int, str, str | None]:
    """Return (prompt_idx, cond_name, mode_or_None) for a .pt stem.

    Supports both schemas:
      legacy flat: '013_fiction'             → (13, 'fiction', None)
      new nested:  '013_jb_fiction_multi'    → (13, 'jb_fiction', 'multi')
                   '013_ctrl_fiction_single' → (13, 'ctrl_fiction', 'single')
                   '013_bare_multi'          → (13, 'bare', 'multi')
    """
    parts = stem.split("_")
    idx = int(parts[0])
    if len(parts) >= 3 and parts[-1] in GRAPH_MODES:
        mode = parts[-1]
        cond_name = "_".join(parts[1:-1])
    else:
        mode = None
        cond_name = "_".join(parts[1:])
    return idx, cond_name, mode


def select_pt_files(
    graphs_dir: Path,
    prompt_filter: set[int] | None,
    class_filter: set[str] | None,
    mode_filter: set[str] | None = None,
) -> list[Path]:
    """Return sorted .pt files matching the filters.

    Accepts both legacy `{idx}_{class}.pt` and new `{idx}_{cond_name}_{mode}.pt`
    naming. If mode_filter is set (e.g. {'single'}), legacy files pass through
    (mode=None) and new files are filtered by the mode suffix.
    """
    selected = []
    for pt in sorted(graphs_dir.glob("*.pt")):
        try:
            idx, cond_name, mode = parse_slug(pt.stem)
        except (ValueError, IndexError):
            continue
        if prompt_filter is not None and idx not in prompt_filter:
            continue
        if class_filter is not None and cond_name not in class_filter:
            continue
        if mode_filter is not None and mode is not None and mode not in mode_filter:
            continue
        selected.append(pt)
    return selected


def group_by_prompt(json_paths: dict[str, Path]) -> dict[int, dict[str, Path]]:
    """Group {slug: path} by prompt index. Keys = full cond_name+mode slug tail.

    Returns {idx: {slug_tail: path}} where slug_tail is the part after '{idx}_'
    (e.g. 'fiction' legacy or 'jb_fiction_multi' new). Downstream 3-way logic
    reconstructs (cond_name, mode) via parse_slug on the original stem.
    """
    by_prompt: dict[int, dict[str, Path]] = defaultdict(dict)
    for slug, path in json_paths.items():
        idx_str, _, tail = slug.partition("_")
        by_prompt[int(idx_str)][tail] = path
    return by_prompt


def group_by_prompt_structured(json_paths: dict[str, Path]) -> dict[int, dict[str, dict[str, Path]]]:
    """Group {slug: path} by (prompt_idx, cond_name, mode).

    Returns {idx: {cond_name: {mode_or_'_nomode': path}}}. For legacy-slug
    files mode_key is '_nomode'. Consumers pick paths via
        by_prompt[idx]['jb_fiction'].get('single') or by_prompt[idx]['fiction'].get('_nomode')
    """
    out: dict[int, dict[str, dict[str, Path]]] = defaultdict(lambda: defaultdict(dict))
    for slug, path in json_paths.items():
        try:
            idx, cond_name, mode = parse_slug(slug)
        except (ValueError, IndexError):
            continue
        mode_key = mode if mode is not None else "_nomode"
        out[idx][cond_name][mode_key] = path
    return out


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
    mode_filter = (
        set(GRAPH_MODES) if args.mode == "both"
        else {args.mode}
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

    # Determine which dir has our per-graph JSONs when --skip-convert is set.
    # Priority: explicit --source-graph-data, then 05_frontend/graph_data,
    # then run_dir/graph_data (where 02c_pack_graphs.py writes).
    source_graph_data_dir = None
    if args.skip_convert:
        candidates = [
            args.source_graph_data,
            graph_data_dir,
            run_dir / "graph_data",
        ]
        for cand in candidates:
            if cand is not None and cand.exists() and (
                any(cand.glob("*.json")) or any(cand.glob("*.json.gz"))
            ):
                source_graph_data_dir = cand.resolve()
                break
        if source_graph_data_dir is None:
            print("\n  ERROR: --skip-convert set but no pre-converted JSONs found.")
            print(f"    Searched: {[str(c) for c in candidates if c is not None]}")
            print("    Run 02c_pack_graphs.py first, or drop --skip-convert.")
            sys.exit(1)
        print(f"  source_graph_data:  {source_graph_data_dir}")

    if not args.skip_convert and not graphs_pt_dir.exists():
        print(f"\n  ERROR: {graphs_pt_dir} does not exist. "
              f"Did Stage 02 run with --save-graphs?")
        sys.exit(1)

    # Enumerate slugs. When converting, we enumerate from .pt files. When
    # skip-convert, we enumerate from the source JSON dir (may have .json or .json.gz).
    if args.skip_convert:
        json_like = list(source_graph_data_dir.glob("*.json")) + list(source_graph_data_dir.glob("*.json.gz"))
        # Build virtual "pt_files" from JSON stems for uniform downstream handling.
        # Use a lightweight namespace object so select_pt_files's filter logic still applies.
        class _Stem:
            def __init__(self, stem):
                self.stem = stem
                self.name = stem + ".pt"
        pt_files = []
        seen = set()
        for jl in sorted(json_like):
            stem = jl.stem[:-5] if jl.suffix == ".gz" else jl.stem  # strip .json.gz or .json
            if jl.suffix == ".gz":
                stem = jl.name.removesuffix(".json.gz")
            else:
                stem = jl.stem
            if stem in seen:
                continue
            seen.add(stem)
            try:
                idx, cond_name, mode = parse_slug(stem)
            except (ValueError, IndexError):
                continue
            if prompt_filter is not None and idx not in prompt_filter:
                continue
            if class_filter is not None and cond_name not in class_filter:
                continue
            if mode_filter is not None and mode is not None and mode not in mode_filter:
                continue
            pt_files.append(_Stem(stem))
        if not pt_files:
            print("\n  ERROR: no JSON files match the filters in source_graph_data.")
            sys.exit(1)
        print(f"\n  Selected {len(pt_files)} graphs (from JSONs, --skip-convert)")
    else:
        pt_files = select_pt_files(graphs_pt_dir, prompt_filter, class_filter, mode_filter)
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
        # Ensure destination exists + copy source JSONs into it if source ≠ dest.
        graph_data_dir.mkdir(parents=True, exist_ok=True)
        if source_graph_data_dir != graph_data_dir.resolve():
            import shutil
            n_copied = 0
            for jl in list(source_graph_data_dir.glob("*.json")) + list(source_graph_data_dir.glob("*.json.gz")):
                dst = graph_data_dir / jl.name
                if not dst.exists():
                    shutil.copy2(jl, dst)
                    n_copied += 1
            print(f"    Copied {n_copied} JSONs from {source_graph_data_dir} → {graph_data_dir}")
        # Also ensure graph-metadata.json is in the staging dir
        md_src = source_graph_data_dir / "graph-metadata.json"
        md_dst = graph_data_dir / "graph-metadata.json"
        if md_src.exists() and not md_dst.exists():
            import shutil
            shutil.copy2(md_src, md_dst)

        # If JSONs are gzipped and subsequent annotation needs plain JSON,
        # ungzip in place (annotate_overlap/annotate_subcircuits read with open()).
        for gz in list(graph_data_dir.glob("*.json.gz")):
            plain = gz.with_suffix("")  # strips .gz → .json
            if not plain.exists():
                import gzip as _gz
                with _gz.open(gz, "rb") as src, open(plain, "wb") as dst:
                    dst.write(src.read())

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
    # Step 2: annotate overlap (3-way: bare / ctrl / jb per prompt, per mode).
    # Falls back to 2-way annotation on legacy runs (no ctrl conditions).
    # ------------------------------------------------------------------
    structured = group_by_prompt_structured(json_paths)
    if args.skip_overlap:
        print("\n  Step 2: skipping overlap annotation (--skip-overlap)")
    else:
        print(f"\n  Step 2: Annotating overlap for {len(structured)} prompts...")
        n_bare = n_ctrl = n_jb_3way = n_jb_2way = n_skip = 0
        bucket_totals: dict[str, int] = defaultdict(int)
        for idx, conds in sorted(structured.items()):
            bare_modes = conds.get("bare", {})
            for mode_key, bare_path in bare_modes.items():
                annotate_bare(bare_path, idx)
                n_bare += 1

            # Annotate ctrl_{cls} graphs (per mode) vs bare. Extracts class name
            # from cond_name when it starts with 'ctrl_'.
            for cond_name, mode_map in conds.items():
                if not cond_name.startswith("ctrl_"):
                    continue
                cls = cond_name.removeprefix("ctrl_")
                for mode_key, ctrl_path in mode_map.items():
                    bare_path = bare_modes.get(mode_key)
                    annotate_ctrl(ctrl_path, bare_path, cls, idx)
                    n_ctrl += 1

            # Annotate jb_{cls} graphs. Prefer 3-way (with matched ctrl) when
            # ctrl_{cls} is available for the same mode; else fall back to 2-way.
            # Also annotate legacy-flat jb classes (e.g. 'fiction' without jb_ prefix).
            for cond_name, mode_map in conds.items():
                if cond_name == "bare" or cond_name.startswith("ctrl_"):
                    continue
                cls = cond_name.removeprefix("jb_") if cond_name.startswith("jb_") else cond_name
                ctrl_map = conds.get(f"ctrl_{cls}", {})
                for mode_key, jb_path in mode_map.items():
                    bare_path = bare_modes.get(mode_key)
                    ctrl_path = ctrl_map.get(mode_key)
                    if bare_path is None:
                        print(f"    WARN: no bare for prompt {idx:03d}/{mode_key}, skipping {cond_name}")
                        n_skip += 1
                        continue
                    if ctrl_path is not None:
                        g = annotate_overlap_3way(jb_path, bare_path, ctrl_path, cls, idx)
                        n_jb_3way += 1
                        for b, c in (g.get("metadata", {}).get("overlap_counts") or {}).items():
                            bucket_totals[b] += c
                    else:
                        annotate_overlap(jb_path, bare_path, cls, idx)
                        n_jb_2way += 1
        print(f"    Annotated {n_bare} bare + {n_ctrl} ctrl + {n_jb_3way} jb (3-way) "
              f"+ {n_jb_2way} jb (2-way legacy) graphs  (skipped {n_skip})")
        if bucket_totals:
            print("    Corpus-level overlap bucket totals (3-way jb graphs):")
            for b in ("shared_with_bare_and_ctrl", "shared_with_bare",
                      "shared_with_ctrl", "jb_unique"):
                if bucket_totals.get(b):
                    print(f"      {b:28s} {bucket_totals[b]:5d}")

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
        sample_logged = False
        for idx, conds in sorted(structured.items()):
            for cond_name, mode_map in conds.items():
                for mode_key, path in mode_map.items():
                    result = annotate_subcircuits(path, subcircuits_json)
                    n_ok += 1
                    if not sample_logged and cond_name == "bare":
                        count = result["metadata"].get("n_subcircuit_annotated", 0)
                        total_nodes = len(result.get("nodes", []))
                        print(f"    sample (prompt {idx:03d}/bare/{mode_key}): "
                              f"{count}/{total_nodes} nodes tagged")
                        sample_logged = True
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
