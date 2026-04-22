"""Shared helpers for Stage 05 circuit visualization."""
from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

OVERLAP_BUCKETS = (
    # JB-graph viewpoints (set by annotate_overlap_3way or annotate_overlap):
    "shared_with_bare_and_ctrl",   # in bare ∩ ctrl ∩ jb — most stable refusal core
    "shared_with_bare",            # in bare ∩ jb, not ctrl — JB + harmful baseline share
    #                                (also used by annotate_ctrl when feature is in bare too)
    "shared_with_ctrl",            # in ctrl ∩ jb, not bare — PREFIX-INDUCED, not JB-semantic
    "jb_unique",                   # only in the jb graph — true JB-semantic signal

    # Ctrl-graph viewpoints (set by annotate_ctrl):
    "ctrl",                        # ctrl-graph baseline tag (analog of 'bare')
    "ctrl_unique",                 # only in this ctrl graph — benign-prefix-induced

    # Bare-graph viewpoint:
    "bare",

    # Legacy / leftover:
    "bare_only",
    "non_feature",
)

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
    relative to the bare graph, persist in-place, and return the mutated dict.

    2-way (legacy) viewpoint: shared_with_bare | jb_unique | non_feature.
    For 3-way bare/ctrl/jb visualization use `annotate_overlap_3way` instead.
    """
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
    jb["metadata"]["overlap_mode"] = "2way"

    with open(jb_json_path, "w") as f:
        json.dump(jb, f, indent=2)
    return jb


def annotate_overlap_3way(
    jb_json_path: Path, bare_json_path: Path, ctrl_json_path: Path,
    jb_class: str, prompt_idx: int,
) -> dict:
    """Annotate a JB graph node-by-node against BOTH bare and ctrl_{cls} graphs.

    Buckets (jb-graph viewpoint):
        shared_with_bare_and_ctrl  — feature in bare ∩ ctrl ∩ jb  (highest stability)
        shared_with_bare           — in bare ∩ jb, NOT in ctrl
        shared_with_ctrl           — in ctrl ∩ jb, NOT in bare    (PREFIX-INDUCED — NOT JB-semantic)
        jb_unique                  — only in the jb graph         (TRUE JB-SEMANTIC signal)
        non_feature                — non-feature nodes

    `shared_with_ctrl` is the diagnostic bucket: features that appear under the
    matched benign ctrl prefix — they're recruited by the prefix itself, not
    by the JB-semantic content. Distinguishing this from `jb_unique` is the
    core novel insight of the 11-condition ctrl-balanced dataset.
    """
    with open(jb_json_path) as f:
        jb = json.load(f)
    with open(bare_json_path) as f:
        bare = json.load(f)
    with open(ctrl_json_path) as f:
        ctrl = json.load(f)

    bare_keys = {
        k for k in (feature_key_from_node(n) for n in bare["nodes"]) if k is not None
    }
    ctrl_keys = {
        k for k in (feature_key_from_node(n) for n in ctrl["nodes"]) if k is not None
    }

    counts = {b: 0 for b in
              ("shared_with_bare_and_ctrl", "shared_with_bare", "shared_with_ctrl",
               "jb_unique", "non_feature")}

    for node in jb["nodes"]:
        key = feature_key_from_node(node)
        if key is None:
            node["overlap_bucket"] = "non_feature"
        else:
            in_bare = key in bare_keys
            in_ctrl = key in ctrl_keys
            if in_bare and in_ctrl:
                node["overlap_bucket"] = "shared_with_bare_and_ctrl"
            elif in_bare:
                node["overlap_bucket"] = "shared_with_bare"
            elif in_ctrl:
                node["overlap_bucket"] = "shared_with_ctrl"
            else:
                node["overlap_bucket"] = "jb_unique"
        counts[node["overlap_bucket"]] += 1

    jb["metadata"]["jb_class"] = jb_class
    jb["metadata"]["prompt_idx"] = prompt_idx
    jb["metadata"]["overlap_mode"] = "3way"
    jb["metadata"]["overlap_counts"] = counts

    with open(jb_json_path, "w") as f:
        json.dump(jb, f, indent=2)
    return jb


def annotate_ctrl(
    ctrl_json_path: Path, bare_json_path: Path | None, ctrl_class: str, prompt_idx: int,
) -> dict:
    """Tag a ctrl graph's features with their ctrl-viewpoint buckets.

    Buckets (ctrl-graph viewpoint):
        ctrl                 — every feature in this ctrl graph (when bare_json_path is None)
        shared_with_bare     — feature appears in bare AND ctrl
        ctrl_unique          — feature only in ctrl (not bare)
        non_feature          — non-feature nodes

    Pass `bare_json_path=None` to just mark everything as `ctrl` (analog of
    the legacy `annotate_bare` flow). Passing bare enables the standalone
    2-way view of the ctrl graph.
    """
    with open(ctrl_json_path) as f:
        ctrl = json.load(f)

    if bare_json_path is not None:
        with open(bare_json_path) as f:
            bare = json.load(f)
        bare_keys = {
            k for k in (feature_key_from_node(n) for n in bare["nodes"]) if k is not None
        }
    else:
        bare_keys = None

    for node in ctrl["nodes"]:
        key = feature_key_from_node(node)
        if key is None:
            node["overlap_bucket"] = "non_feature"
        elif bare_keys is None:
            node["overlap_bucket"] = "ctrl"
        elif key in bare_keys:
            node["overlap_bucket"] = "shared_with_bare"
        else:
            node["overlap_bucket"] = "ctrl_unique"

    ctrl["metadata"]["jb_class"] = f"ctrl_{ctrl_class}"
    ctrl["metadata"]["ctrl_class"] = ctrl_class
    ctrl["metadata"]["prompt_idx"] = prompt_idx
    ctrl["metadata"]["overlap_mode"] = "ctrl" if bare_keys is None else "ctrl_vs_bare"

    with open(ctrl_json_path, "w") as f:
        json.dump(ctrl, f, indent=2)
    return ctrl


# ---------------------- subcircuit filter rules -------------------------
# Stage 07's corpus-level subcircuit memberships are aggregated across the
# union of top-50-per-condition feature sets. A feature can be a member of
# `universal_refusal_core` at the corpus level (present in bare + all 5 JB
# top-50 pools) yet still appear in a single JB graph's pruned output as
# `jb_unique` — the feature survived pruning on that JB side but not on the
# matched bare side for this particular prompt. Georg asked that the frontend
# only display memberships that are consistent with each graph's own
# `overlap_bucket`, so the UI does not paint a jb_unique node as "universal".

# universal_refusal_core: feature in bare + all jb + all ctrl (corpus level).
# In any single graph, it should appear as "bare-equivalent": bare itself,
# or shared-with-bare in a jb/ctrl graph.
_UNIVERSAL_CORE_BUCKETS = frozenset({
    "bare", "ctrl",
    "shared_with_bare", "shared_with_bare_and_ctrl",
})
# canonical_pro_refusal: in all 5 jb, not bare (corpus). Per-graph, the feature
# must NOT appear in the bare graph for this prompt. In a jb graph, that's
# jb_unique (definitely not bare) or shared_with_ctrl (in ctrl+jb, not bare).
_CANONICAL_BUCKETS = frozenset({"jb_unique", "shared_with_ctrl"})
_CLASS_EXCLUSIVE_BUCKETS = frozenset({"jb_unique", "shared_with_ctrl"})

# Ctrl-aware subcircuits (Task 10, Apr 22) — require Task 7 per_condition_top50 data.
#   ctrl_shared_refusal: bare ∩ all 5 ctrl, missing some jb. At per-graph level
#     these are "bare-ish" features, so they show up as bare / ctrl / shared
#     variants (NEVER as jb_unique — that would contradict the corpus rule).
_CTRL_SHARED_REFUSAL_BUCKETS = frozenset({
    "bare", "ctrl",
    "shared_with_bare", "shared_with_bare_and_ctrl", "shared_with_ctrl",
})
#   ctrl_only: all 5 ctrl, not bare, not any jb. At per-graph level, must be
#     "ctrl-unique" — anything touching bare or jb contradicts the corpus rule.
_CTRL_ONLY_BUCKETS = frozenset({"ctrl_unique", "ctrl"})
#   jb_{cls}_specific_vs_ctrl: in jb_{cls} top-50, NOT in ctrl_{cls} top-50.
#     Allowed only in the matching jb_{cls} graph, as jb_unique or shared_with_bare
#     (not shared_with_ctrl — that would mean feature IS in ctrl too, contradicting).
_JB_SPECIFIC_BUCKETS = frozenset({"jb_unique", "shared_with_bare"})

# These subcircuits are orthogonal to the bare/ctrl/JB axis — a feature can be a
# "sign-flip" or "dampening specialist" in any graph where it appears,
# regardless of whether the overlap bucket says bare, shared, or jb_unique.
_UNFILTERED_SUBCIRCUITS = frozenset({
    "sign_flip_convergent",
    "dampening_specialists",
    "anti_refusal_amplifiers",
    "late_wave_layer24_32",
})


def _subcircuit_allowed(
    sc_name: str, overlap_bucket: str | None, jb_class: str | None,
) -> bool:
    """Is this corpus-level subcircuit membership consistent with the per-graph
    overlap bucket + jb class? See the rules above."""
    if sc_name in _UNFILTERED_SUBCIRCUITS:
        return True
    # If overlap wasn't annotated (e.g. --skip-overlap debug run), fall back to
    # corpus behavior rather than stripping every bucket-conditional membership.
    if overlap_bucket is None:
        return True
    if sc_name == "universal_refusal_core":
        return overlap_bucket in _UNIVERSAL_CORE_BUCKETS
    if sc_name == "canonical_pro_refusal":
        return overlap_bucket in _CANONICAL_BUCKETS
    if sc_name == "ctrl_shared_refusal":
        return overlap_bucket in _CTRL_SHARED_REFUSAL_BUCKETS
    if sc_name == "ctrl_only":
        return overlap_bucket in _CTRL_ONLY_BUCKETS
    if sc_name.startswith("jb_") and sc_name.endswith("_specific_vs_ctrl"):
        # "jb_cognitive_reframe_specific_vs_ctrl" → cls is "cognitive_reframe"
        cls = sc_name.removeprefix("jb_").removesuffix("_specific_vs_ctrl")
        return overlap_bucket in _JB_SPECIFIC_BUCKETS and jb_class == cls
    if sc_name.endswith("_exclusive"):
        # "cognitive_reframe_exclusive" → class is "cognitive_reframe"
        cls = sc_name.removesuffix("_exclusive")
        return overlap_bucket in _CLASS_EXCLUSIVE_BUCKETS and jb_class == cls
    # Unknown subcircuit name — pass through so newly-added subcircuits in
    # Stage 07 don't get silently dropped until this file is updated.
    return True


def annotate_subcircuits(graph_json_path: Path, subcircuits_json_path: Path) -> dict:
    """Attach `subcircuits: [...]` membership array to every feature node in a graph JSON.

    Reads Stage 07's `subcircuits.json`, builds a reverse index feature_key -> [names],
    and writes each feature node's memberships in place. Memberships are
    filtered against the per-graph `overlap_bucket` (written earlier by
    `annotate_bare` / `annotate_overlap`) and `metadata.jb_class` so the UI
    doesn't paint membership colors inconsistent with this graph's own
    bare/JB presence. Non-feature nodes get `[]`. Persists the mutated graph
    JSON in place and returns the dict.
    """
    with open(subcircuits_json_path) as f:
        sc_data = json.load(f)

    feature_to_subcircuits: dict[str, list[str]] = {}
    for name, info in sc_data.get("subcircuits", {}).items():
        for key in info.get("features", []):
            feature_to_subcircuits.setdefault(key, []).append(name)

    with open(graph_json_path) as f:
        graph = json.load(f)
    jb_class = (graph.get("metadata") or {}).get("jb_class")

    n_annotated = 0
    n_filtered = 0
    for node in graph.get("nodes", []):
        key = feature_key_from_node(node)
        if key is None:
            node["subcircuits"] = []
            continue
        candidates = feature_to_subcircuits.get(key, [])
        overlap_bucket = node.get("overlap_bucket")
        kept = [
            sc for sc in candidates
            if _subcircuit_allowed(sc, overlap_bucket, jb_class)
        ]
        n_filtered += len(candidates) - len(kept)
        node["subcircuits"] = kept
        if kept:
            n_annotated += 1

    graph.setdefault("metadata", {})["n_subcircuit_annotated"] = n_annotated
    graph["metadata"]["n_subcircuit_filtered"] = n_filtered
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