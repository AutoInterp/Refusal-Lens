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
import shutil
from pathlib import Path

from trace_classifier import classify_pair, bake_trace_classes
from trace_propagate import (build_key_graph, upstream_contributions, delta_decompose,
                             assign_upstream_classes, bake_upstream_classes)

PATCHES = Path(__file__).resolve().parent / "05_frontend_patches"
ROOT = Path(__file__).resolve().parents[2]


def load_graph(path: Path) -> dict:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(Path(path).read_text())


def _seed_keys(evidence, cls):
    return [(r["layer"], r["feature"]) for r in evidence if r["class"] == cls]


def build_pair_entry(idx, jb_class, request, bare_graph, jb_graph, cfg):
    out = classify_pair(bare_graph, jb_graph, top_n=cfg.get("top_n", 20),
                        delta=cfg.get("delta", 0.30),
                        model_token_gate=cfg.get("model_token_gate", False))
    baked_bare = bake_trace_classes(bare_graph, out["bare"])
    baked_jb = bake_trace_classes(jb_graph, out["jb"])
    # ---- v2 propagation on the key-level digraphs ----
    k, tau, margin = cfg.get("k", 3), cfg.get("tau", 0.05), cfg.get("margin", 0.25)
    bkg, jkg = build_key_graph(baked_bare), build_key_graph(baked_jb)
    refusal_seeds = _seed_keys(out["evidence"], "refusal_centric")
    supp_seeds = _seed_keys(out["evidence"], "suppression")
    ampl_seeds = _seed_keys(out["evidence"], "amplification")
    contrib_by_class = {"refusal_centric": upstream_contributions(jkg, refusal_seeds, k=k, tau=tau)}
    delta_by_class = {
        "suppression": delta_decompose(bkg, jkg, supp_seeds, k=k, tau=tau, margin=margin),
        "amplification": delta_decompose(bkg, jkg, ampl_seeds, k=k, tau=tau, margin=margin)}
    fam = assign_upstream_classes(contrib_by_class, delta_by_class)
    bake_upstream_classes(baked_bare, fam)
    bake_upstream_classes(baked_jb, fam)
    # ---- evidence rows: seeds (hop 0) already in out["evidence"]; add hop/mechanism ----
    for r in out["evidence"]:
        r["hop"] = 0
        r["mechanism"] = "seed"
    seed_keys_set = {(r["layer"], r["feature"]) for r in out["evidence"]}
    for key, info in fam.items():
        if key in seed_keys_set:
            continue
        cls = info["upstream_class"]
        src = (contrib_by_class if cls == "refusal_centric" else delta_by_class).get(cls, {})
        score = src.get("per_feature", {}).get(key, {}).get("contrib",
                src.get("per_feature", {}).get(key, {}).get("delta", 0.0))
        out["evidence"].append({
            "layer": key[0], "feature": key[1], "class": "upstream_" + cls,
            "edge_bare": 0.0, "edge_jb": round(score, 3), "act_bare": 0.0, "act_jb": 0.0,
            "overlap_bucket": "", "hop": info["hop"], "mechanism": info["mechanism"]})
    ROLE = {"refusal_centric": "upstream_refusal", "suppression": "upstream_suppression",
            "amplification": "upstream_amplification"}
    hyps = []
    for key, info in fam.items():
        cls = info["upstream_class"]
        src = (contrib_by_class if cls == "refusal_centric" else delta_by_class).get(cls)
        d = src["per_feature"].get(key, {})
        score = d.get("contrib", d.get("delta", 0.0))
        hyps.append({"prompt_idx": idx, "jb_class": jb_class,
                     "feature": {"layer": key[0], "feature": key[1]},
                     "role": ROLE[cls], "hop": info["hop"], "mechanism": info["mechanism"],
                     "signed_contribution": score, "predicted_effect": score,
                     "coverage": src.get("coverage", 0.0), "verification_status": "unverified"})
    # ---- per-pair coverage / error_frac maps (the honesty line in trace.html) ----
    cov = {"refusal_centric": contrib_by_class["refusal_centric"]["coverage"],
           "suppression": delta_by_class["suppression"]["coverage"],
           "amplification": delta_by_class["amplification"]["coverage"]}
    ef = {"refusal_centric": contrib_by_class["refusal_centric"]["error_frac"],
          "suppression": delta_by_class["suppression"]["error_frac"],
          "amplification": delta_by_class["amplification"]["error_frac"]}
    pair = {"idx": idx, "jb_class": jb_class, "request": request,
            "bare_slug": f"{idx:03d}_bare_single", "jb_slug": f"{idx:03d}_{jb_class}_single",
            "evidence": out["evidence"], "coverage": cov, "error_frac": ef}
    return pair, baked_bare, baked_jb, hyps


def _inject_patch(index_html: Path):
    html = index_html.read_text()
    # ?v=<mtime> cache-buster so iterating on the patch JS/CSS isn't masked by a stale
    # browser cache (the viewer assets otherwise have no cache-busting).
    viewer = index_html.parent

    def _v(name):
        f = viewer / name
        return f"?v={int(f.stat().st_mtime)}" if f.exists() else ""
    inj = (f'<link rel="stylesheet" href="./trace-highlight.css{_v("trace-highlight.css")}">\n'
           f'<script src="./trace-highlight.js{_v("trace-highlight.js")}" defer></script>\n')
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
    all_hyps = []
    for p in cfg["pairs"]:
        idx, jbc = p["idx"], p["jb_class"]
        bare = load_graph(gd / f"{idx:03d}_bare_single.json.gz")
        jb = load_graph(gd / f"{idx:03d}_{jbc}_single.json.gz")
        pair, baked_bare, baked_jb, hyps = build_pair_entry(idx, jbc, p["request"], bare, jb, cfg)
        all_hyps.extend(hyps)
        for slug, baked in ((pair["bare_slug"], baked_bare), (pair["jb_slug"], baked_jb)):
            with gzip.open(gd / f"{slug}.json.gz", "wt", encoding="utf-8") as fh:
                json.dump(baked, fh)
        pairs.append(pair)
        feat_counts: dict[str, int] = {}
        for r in pair["evidence"]:
            feat_counts[r["class"]] = feat_counts.get(r["class"], 0) + 1
        total = sum(feat_counts.values())
        breakdown = ", ".join(f"{feat_counts[k]} {k}"
                              for k in ("refusal_centric", "suppression", "amplification")
                              if k in feat_counts)
        print(f"  [{idx} {jbc}] {total} features ({breakdown})")

    # 4. manifest + trace.html
    manifest = {"title": cfg.get("title", "Bare→Comply Trace"), "viewer": "viewer", "pairs": pairs}
    (args.out / "trace_manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.copy2(PATCHES / "trace.html", args.out / "trace.html")
    (args.out / "trace_hypotheses.json").write_text(json.dumps(all_hyps, indent=2))

    print(f"\nAssembled {len(pairs)} pairs.")
    print(f"Serve:\n  cd {args.out}\n  python3 -m http.server 8000")
    print("  open http://localhost:8000/trace.html")


if __name__ == "__main__":
    main()
