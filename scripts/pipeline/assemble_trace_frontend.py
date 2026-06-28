"""Assemble the 2-panel bare→comply trace site (Gemma-complement only).

Loads the 4 judge-verified flips' bare+jb graphs from the complement run,
classifies features (trace_classifier), bakes rl_trace_class onto each node,
copies the viewer + injects the recolor patch, and writes trace.html +
trace_manifest.json. No GPU, no re-fetch.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
from pathlib import Path

from trace_classifier import classify_pair, bake_trace_classes

PATCHES = Path(__file__).resolve().parent / "05_frontend_patches"
ROOT = Path(__file__).resolve().parents[2]


def load_graph(path: Path) -> dict:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(Path(path).read_text())


def build_pair_entry(idx, jb_class, request, bare_graph, jb_graph, cfg):
    out = classify_pair(bare_graph, jb_graph, top_n=cfg.get("top_n", 20),
                        delta=cfg.get("delta", 0.30),
                        model_token_gate=cfg.get("model_token_gate", False))
    baked_bare = bake_trace_classes(bare_graph, out["bare"])
    baked_jb = bake_trace_classes(jb_graph, out["jb"])
    pair = {"idx": idx, "jb_class": jb_class, "request": request,
            "bare_slug": f"{idx:03d}_bare_single",
            "jb_slug": f"{idx:03d}_{jb_class}_single",
            "evidence": out["evidence"]}
    return pair, baked_bare, baked_jb


def _inject_patch(index_html: Path):
    html = index_html.read_text()
    inj = ('<link rel="stylesheet" href="./trace-highlight.css">\n'
           '<script src="./trace-highlight.js" defer></script>\n')
    if "trace-highlight.js" in html:
        return
    marker = "<script src='./util.js'></script>"
    if marker in html:
        html = html.replace(marker, inj + marker)
    else:
        html = html.replace("</head>", inj + "</head>")
    index_html.write_text(html)


def main():
    ap = argparse.ArgumentParser(description="Assemble the bare→comply trace site")
    ap.add_argument("--config", type=Path, default=PATCHES / "trace_config.json")
    ap.add_argument("--out", type=Path, default=ROOT / "data/results/trace_bare_to_comply")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    comp = ROOT / cfg["complement_run"] / "05_frontend"
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. copy the viewer once
    viewer = args.out / "viewer"
    if viewer.exists():
        shutil.rmtree(viewer)
    shutil.copytree(comp, viewer)

    # 2. copy patch files into the viewer + inject
    for name in ("trace-highlight.css", "trace-highlight.js"):
        shutil.copy2(PATCHES / name, viewer / name)
    _inject_patch(viewer / "index.html")

    # 3. classify + bake the 8 graphs
    gd = viewer / "graph_data"
    pairs = []
    for p in cfg["pairs"]:
        idx, jbc = p["idx"], p["jb_class"]
        bare = load_graph(gd / f"{idx:03d}_bare_single.json.gz")
        jb = load_graph(gd / f"{idx:03d}_{jbc}_single.json.gz")
        pair, baked_bare, baked_jb = build_pair_entry(idx, jbc, p["request"], bare, jb, cfg)
        for slug, baked in ((pair["bare_slug"], baked_bare), (pair["jb_slug"], baked_jb)):
            with gzip.open(gd / f"{slug}.json.gz", "wt", encoding="utf-8") as fh:
                json.dump(baked, fh)
        pairs.append(pair)
        n = sum(1 for r in pair["evidence"] if r["class"] != "neutral")
        print(f"  [{idx} {jbc}] {n} classified features")

    # 4. manifest + trace.html
    manifest = {"title": cfg.get("title", "Bare→Comply Trace"), "viewer": "viewer", "pairs": pairs}
    (args.out / "trace_manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.copy2(PATCHES / "trace.html", args.out / "trace.html")

    print(f"\nAssembled {len(pairs)} pairs.")
    print(f"Serve:\n  cd {args.out}\n  python3 -m http.server 8000")
    print("  open http://localhost:8000/trace.html")


if __name__ == "__main__":
    main()
