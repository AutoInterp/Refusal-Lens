"""Shared helpers for Stage 05 circuit visualization."""
from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

OVERLAP_BUCKETS = ("shared_with_bare", "jb_unique", "bare_only", "bare", "non_feature")

# Circuit-tracer browser frontend lives at vendor/circuit-tracer/circuit_tracer/frontend/assets/
# (NOT the parent `frontend/` dir, which contains Python helpers).
VENDOR_FRONTEND = (
    Path(__file__).resolve().parents[2]
    / "vendor" / "circuit-tracer" / "circuit_tracer" / "frontend" / "assets"
)


def feature_key_from_node(node: dict) -> str | None:
    """Extract L{layer}:F{feature_idx} from a frontend-format node dict.
    Returns None for non-feature nodes (embedding/logit/error)."""
    if node.get("feature_type") != "cross layer transcoder":
        return None
    layer = node.get("layer")
    # Node IDs look like "0_4096_5" = "{layer}_{feature}_{ctx_idx}"
    parts = str(node.get("node_id", "")).split("_")
    if len(parts) < 2:
        return None
    try:
        return f"L{int(parts[0])}:F{int(parts[1])}"
    except (ValueError, TypeError):
        return None


def convert_pt_to_frontend_json(
    pt_path: Path, slug: str, out_dir: Path,
    scan: str = "mwhanna/gemma-scope-2-4b-it//transcoder_all/width_16k_l0_small_affine",
    node_threshold: float = 0.8, edge_threshold: float = 0.98,
) -> Path:
    """Wrap circuit-tracer's create_graph_files. Returns path to <slug>.json."""
    from circuit_tracer.utils.create_graph_files import create_graph_files

    out_dir.mkdir(parents=True, exist_ok=True)
    create_graph_files(
        graph_or_path=str(pt_path),
        slug=slug,
        output_path=str(out_dir),
        scan=scan,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
    )
    return out_dir / f"{slug}.json"


def annotate_overlap(
    jb_json_path: Path, bare_json_path: Path, jb_class: str, prompt_idx: int,
) -> dict:
    """Load a JB graph JSON, annotate each feature node with overlap_bucket
    relative to the bare graph, persist in-place, and return the mutated dict."""
    with open(jb_json_path) as f:
        jb = json.load(f)
    with open(bare_json_path) as f:
        bare = json.load(f)

    bare_keys = {
        k for k in (feature_key_from_node(n) for n in bare["nodes"]) if k is not None
    }

    for node in jb["nodes"]:
        key = feature_key_from_node(node)
        if key is None:
            node["overlap_bucket"] = "non_feature"
        elif key in bare_keys:
            node["overlap_bucket"] = "shared_with_bare"
        else:
            node["overlap_bucket"] = "jb_unique"

    jb["metadata"]["jb_class"] = jb_class
    jb["metadata"]["prompt_idx"] = prompt_idx

    with open(jb_json_path, "w") as f:
        json.dump(jb, f, indent=2)
    return jb


def annotate_subcircuits(graph_json_path: Path, subcircuits_json_path: Path) -> dict:
    """Attach `subcircuits: [...]` membership array to every feature node in a graph JSON.

    Reads Stage 07's `subcircuits.json`, builds a reverse index feature_key -> [names],
    and writes each feature node's memberships in place. Non-feature nodes get `[]`.
    Persists the mutated graph JSON in place and returns the dict.
    """
    with open(subcircuits_json_path) as f:
        sc_data = json.load(f)

    feature_to_subcircuits: dict[str, list[str]] = {}
    for name, info in sc_data.get("subcircuits", {}).items():
        for key in info.get("features", []):
            feature_to_subcircuits.setdefault(key, []).append(name)

    with open(graph_json_path) as f:
        graph = json.load(f)

    n_annotated = 0
    for node in graph.get("nodes", []):
        key = feature_key_from_node(node)
        if key is None:
            node["subcircuits"] = []
            continue
        scs = feature_to_subcircuits.get(key, [])
        node["subcircuits"] = scs
        if scs:
            n_annotated += 1

    graph.setdefault("metadata", {})["n_subcircuit_annotated"] = n_annotated
    graph["metadata"]["subcircuits_source"] = subcircuits_json_path.name

    with open(graph_json_path, "w") as f:
        json.dump(graph, f, indent=2)
    return graph


def annotate_bare(bare_json_path: Path, prompt_idx: int) -> dict:
    """Tag every feature node in a bare graph JSON as 'bare'."""
    with open(bare_json_path) as f:
        bare = json.load(f)
    for node in bare["nodes"]:
        if feature_key_from_node(node) is not None:
            node["overlap_bucket"] = "bare"
        else:
            node["overlap_bucket"] = "non_feature"
    bare["metadata"]["jb_class"] = "bare"
    bare["metadata"]["prompt_idx"] = prompt_idx
    with open(bare_json_path, "w") as f:
        json.dump(bare, f, indent=2)
    return bare


def inject_feature_labels(graph_data_dir: Path, labels_dir: Path) -> dict:
    """Bake human-readable labels into each graph's CLT feature nodes.

    Reads per-layer files like `feature_labels_layer_{L}.json` (keyed by
    per-layer feature index, value `{"explanation": {"label": "...", ...}}`),
    walks every graph file in `graph_data_dir`, and writes
    `node["clerp"] = label_text` for each cross-layer-transcoder node whose
    (layer, feat_idx) has a matching entry. The vendor frontend renders
    `d.clerp` via `ppClerp` everywhere a feature title appears, so this is
    sufficient — no JS patch needed.

    Handles both `*.json` and `*.json.gz` in place. Returns a stats dict.
    """
    if not labels_dir.exists():
        return {"labeled": 0, "missed": 0, "files": 0, "skipped": True}

    layer_labels: dict[int, dict] = {}
    for lp in sorted(labels_dir.glob("feature_labels_layer_*.json")):
        try:
            layer_idx = int(lp.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        with open(lp) as f:
            layer_labels[layer_idx] = json.load(f)

    if not layer_labels:
        return {"labeled": 0, "missed": 0, "files": 0, "skipped": True}

    def label_for(layer: int, feat_idx: int) -> str | None:
        entry = layer_labels.get(layer, {}).get(str(feat_idx))
        if not entry:
            return None
        explanation = entry.get("explanation") or {}
        return explanation.get("label") or None

    def annotate_graph(graph: dict) -> tuple[int, int]:
        labeled = missed = 0
        for node in graph.get("nodes", []):
            if node.get("feature_type") != "cross layer transcoder":
                continue
            parts = str(node.get("node_id", "")).split("_")
            if len(parts) < 2:
                continue
            try:
                layer = int(parts[0])
                feat_idx = int(parts[1])
            except ValueError:
                continue
            text = label_for(layer, feat_idx)
            if text:
                node["clerp"] = text
                labeled += 1
            else:
                missed += 1
        return labeled, missed

    total_labeled = total_missed = n_files = 0
    for jp in sorted(graph_data_dir.glob("*.json")):
        if jp.name == "graph-metadata.json":
            continue
        with open(jp) as f:
            graph = json.load(f)
        l, m = annotate_graph(graph)
        with open(jp, "w") as f:
            json.dump(graph, f)
        total_labeled += l
        total_missed += m
        n_files += 1
    for gz in sorted(graph_data_dir.glob("*.json.gz")):
        with gzip.open(gz, "rt") as f:
            graph = json.load(f)
        l, m = annotate_graph(graph)
        with gzip.open(gz, "wt", compresslevel=6) as f:
            json.dump(graph, f)
        total_labeled += l
        total_missed += m
        n_files += 1

    return {"labeled": total_labeled, "missed": total_missed, "files": n_files, "skipped": False}


def _load_feature_evidence_cache(repo_root: Path) -> dict:
    """Aggregate every available `feature_labels_cache.json` into one lookup.

    Cache key is `L{layer}:F{feat_idx}`; value has `top_logits`, `examples`,
    `bottom_logits`, etc. Multiple runs may have caches; we merge with later
    runs overriding earlier ones for the same key.
    """
    out: dict[str, dict] = {}
    pattern = "data/results/pipeline_runs/*/04_labels/feature_labels_cache.json"
    for cp in sorted(repo_root.glob(pattern)):
        try:
            with open(cp) as f:
                cache = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        for key, info in cache.items():
            if info:
                out[key] = info
    return out


def inject_feature_evidence(graph_data_dir: Path, repo_root: Path) -> dict:
    """Bake top logits + activation examples onto each CLT node.

    Reads aggregated `feature_labels_cache.json` (top_logits, examples) and
    writes per-node `preview_top_logits` (5 strings) and `preview_examples`
    (≤3 dicts trimmed to the fields the frontend tooltip uses). Lets the
    user verify a feature's actual behaviour without an HF roundtrip — and
    catches misleading labels at a glance.
    """
    evidence = _load_feature_evidence_cache(repo_root)
    if not evidence:
        return {"enriched": 0, "missed": 0, "files": 0, "skipped": True}

    def trim_example(ex: dict) -> dict:
        return {
            "context": ex.get("context", ""),
            "trigger_token": ex.get("trigger_token", ""),
            "trigger_activation": ex.get("trigger_activation"),
        }

    def annotate_graph(graph: dict) -> tuple[int, int]:
        enriched = missed = 0
        for node in graph.get("nodes", []):
            if node.get("feature_type") != "cross layer transcoder":
                continue
            parts = str(node.get("node_id", "")).split("_")
            if len(parts) < 2:
                continue
            try:
                layer = int(parts[0])
                feat_idx = int(parts[1])
            except ValueError:
                continue
            info = evidence.get(f"L{layer}:F{feat_idx}")
            if not info:
                missed += 1
                continue
            node["preview_top_logits"] = (info.get("top_logits") or [])[:5]
            node["preview_examples"] = [trim_example(e) for e in (info.get("examples") or [])[:3]]
            enriched += 1
        return enriched, missed

    total_enriched = total_missed = n_files = 0
    for jp in sorted(graph_data_dir.glob("*.json")):
        if jp.name == "graph-metadata.json":
            continue
        with open(jp) as f:
            graph = json.load(f)
        e, m = annotate_graph(graph)
        with open(jp, "w") as f:
            json.dump(graph, f)
        total_enriched += e
        total_missed += m
        n_files += 1
    for gz in sorted(graph_data_dir.glob("*.json.gz")):
        with gzip.open(gz, "rt") as f:
            graph = json.load(f)
        e, m = annotate_graph(graph)
        with gzip.open(gz, "wt", compresslevel=6) as f:
            json.dump(graph, f)
        total_enriched += e
        total_missed += m
        n_files += 1

    return {"enriched": total_enriched, "missed": total_missed, "files": n_files, "skipped": False}


def gzip_json_files(graph_data_dir: Path, keep_plain: bool = False) -> dict:
    """Compress every per-graph *.json in graph_data_dir to *.json.gz.

    Skips `graph-metadata.json` — that's the small index the frontend loads
    unconditionally at page load, is served as plain JSON, and must stay
    available at <frontend>/data/graph-metadata.json. Gzipping it would
    leave the frontend with a stale or missing metadata file after staging.
    """
    results = {"compressed": {}, "total_plain": 0, "total_gz": 0, "skipped": []}
    for jp in sorted(graph_data_dir.glob("*.json")):
        if jp.name == "graph-metadata.json":
            results["skipped"].append(jp.name)
            continue
        gz_path = jp.with_suffix(".json.gz")
        with open(jp, "rb") as src, gzip.open(gz_path, "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst)
        plain = jp.stat().st_size
        gz = gz_path.stat().st_size
        results["compressed"][jp.name] = {"plain": plain, "gz": gz, "ratio": round(plain / gz, 2)}
        results["total_plain"] += plain
        results["total_gz"] += gz
        if not keep_plain:
            jp.unlink()
    return results


def stage_frontend(
    graph_data_dir: Path,
    frontend_out: Path,
    vendor_frontend: Path = VENDOR_FRONTEND,
    use_gzip: bool = False,
) -> None:
    """Copy circuit-tracer's frontend assets + arrange our graphs into the expected layout.

    Final layout:
        <frontend_out>/
            index.html, style.css, util.js, attribution_graph/..., assets/...   (from vendor)
            graph_data/<slug>.json     (moved from graph_data_dir)
            data/graph-metadata.json   (moved from graph_data_dir)
    """
    if not vendor_frontend.exists():
        raise FileNotFoundError(f"vendor frontend not found at {vendor_frontend}")
    frontend_out.mkdir(parents=True, exist_ok=True)

    # Copy frontend scaffolding (don't overwrite existing graph_data/ if already there)
    for entry in vendor_frontend.iterdir():
        if entry.name in {"graph_data", "data"}:
            continue
        dst = frontend_out / entry.name
        if entry.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(entry, dst)
        else:
            shutil.copy2(entry, dst)

    # Copy our graphs into graph_data/ (skip when caller already staged there)
    graph_data_out = frontend_out / "graph_data"
    graph_data_out.mkdir(exist_ok=True)
    same_graph_dir = graph_data_dir.resolve() == graph_data_out.resolve()
    if not same_graph_dir:
        for jp in sorted(graph_data_dir.glob("*.json")):
            if jp.name == "graph-metadata.json":
                continue
            shutil.copy2(jp, graph_data_out / jp.name)
        for gz in sorted(graph_data_dir.glob("*.json.gz")):
            shutil.copy2(gz, graph_data_out / gz.name)
    
    # Inject overlap-coloring patches into index.html
    patches_dir = Path(__file__).resolve().parent / "05_frontend_patches"
    if patches_dir.exists():
        # Copy patch files into frontend root (skip subdirs / __pycache__)
        for patch in patches_dir.iterdir():
            if patch.is_dir() or patch.name == "__pycache__":
                continue
            shutil.copy2(patch, frontend_out / patch.name)
        # Inject <link> + <script> into index.html
        index_path = frontend_out / "index.html"
        html = index_path.read_text()
        gzip_flag = '<script>window.REFUSAL_LENS_USE_GZIP = true;</script>\n' if use_gzip else ''
        injection = (
            gzip_flag
            + '<link rel="stylesheet" href="./overlap-colors.css">\n'
            '<link rel="stylesheet" href="./subcircuit-panel.css">\n'
            '<script src="./fetch-override.js" defer></script>\n'
            '<script src="./gzip-fetch.js" defer></script>\n'
            '<script src="./overlap-annotate.js" defer></script>\n'
            '<script src="./subcircuit-panel.js" defer></script>\n'
            '<script src="./feature-evidence.js" defer></script>\n'
        )
        marker = "<script src='./util.js'></script>"
        if marker in html and injection not in html:
            html = html.replace(marker, injection + marker)
            index_path.write_text(html)

    # Move graph-metadata.json to data/ (different directory — always copy)
    data_out = frontend_out / "data"
    data_out.mkdir(exist_ok=True)
    metadata_src = graph_data_dir / "graph-metadata.json"
    metadata_dst = data_out / "graph-metadata.json"
    if metadata_src.exists() and metadata_src.resolve() != metadata_dst.resolve():
        shutil.copy2(metadata_src, metadata_dst)