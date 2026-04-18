"""Shared helpers for Stage 05 circuit visualization."""
from __future__ import annotations

import json
from pathlib import Path

OVERLAP_BUCKETS = ("shared_with_bare", "jb_unique", "bare_only", "bare", "non_feature")


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
    scan: str = "gemma-scope-2-4b-it",
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