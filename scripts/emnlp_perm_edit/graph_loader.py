"""Loader for packed JSON.gz attribution graphs from run_20260430_023247.

Loads a single graph file, identifies the measurement-target node (the
"logit" node representing direct_dot = h[L15, pos=-2] · r_hat), and aggregates
signed edge attributions by source-node category for the linearization audit.

Source-node categories (Phase 0 § 2.2 of EXPERIMENT_PLAN):
- feature: transcoder feature outputs (CLT decoder writes)
- embedding: token embeddings (input write)
- error_node: transcoder reconstruction error residuals

Schema discovered in run_20260430_023247 packed graphs (2026-05-19):
- top-level: {metadata, qParams, nodes, links}
- node fields: {node_id, feature, layer, ctx_idx, feature_type, is_target_logit, ...}
- edge fields: {source, target, weight}  (source/target reference node_id)
- feature_type values: cross layer transcoder | mlp reconstruction error | embedding | logit
- unique target identification: feature_type == 'logit' AND is_target_logit == True
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Literal


Category = Literal["feature", "embedding", "error_node"]


# Maps circuit-tracer's raw feature_type strings to our 3-bucket category.
DEFAULT_NODE_TYPE_TO_CATEGORY: dict[str, Category] = {
    "cross layer transcoder": "feature",
    "feature": "feature",                            # legacy alias
    "embedding": "embedding",
    "mlp reconstruction error": "error_node",
    "error": "error_node",                           # alias
    # 'logit' is the target itself, not a source category
}


def _parse_layer_feature_from_node(node: dict) -> tuple[int | None, int | None]:
    """Return (layer, feature_idx) for a packed-graph node.

    The packed graph uses node_id format `<layer>_<feature_idx>_<ctx_idx>` for
    transcoder/embedding/error nodes, where the middle component is the
    Gemma Scope per-layer feature_idx (0-16383) — same numbering as Stage 04
    feature_labels.json. The `feature` field is a different identifier
    (looks like a Neuronpedia API ID for cross_layer_transcoder nodes).

    Falls back to `node["layer"]` and `node["feature"]` if node_id doesn't
    match the 3-component format (e.g., simple test fixtures).
    """
    nid = node.get("node_id", "")
    parts = nid.split("_") if isinstance(nid, str) else []
    if len(parts) == 3:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            pass
    # Fallback to schema fields
    L_raw = node.get("layer")
    F = node.get("feature")
    L = int(L_raw) if L_raw is not None and str(L_raw).lstrip("-").isdigit() else None
    if isinstance(F, str) and F.lstrip("-").isdigit():
        F = int(F)
    return L, F


def load_packed_graph(path: Path) -> dict:
    """Load a gzipped JSON attribution graph and return the parsed dict."""
    with gzip.open(path, "rt") as f:
        return json.load(f)


def find_measurement_target_node_id(graph: dict) -> str:
    """Return the node_id of the measurement target (logit node with is_target_logit=True).

    Schema convention from run_20260430_023247 packed graphs:
      - feature_type == 'logit'
      - is_target_logit == True (only one such node per graph)
    """
    candidates = [
        n["node_id"] for n in graph["nodes"]
        if n.get("feature_type") == "logit" and n.get("is_target_logit", False)
    ]
    if len(candidates) == 0:
        # Fallback: any logit node (older/legacy schema)
        candidates = [n["node_id"] for n in graph["nodes"] if n.get("feature_type") == "logit"]
    if len(candidates) == 0:
        raise ValueError("no logit node found in graph — schema mismatch?")
    if len(candidates) > 1:
        raise ValueError(f"multiple target logit nodes found: {candidates}; schema differs from assumption")
    return candidates[0]


def aggregate_edge_attributions(
    graph: dict,
    target_node_id: str,
    node_type_to_category: dict[str, Category] = DEFAULT_NODE_TYPE_TO_CATEGORY,
) -> dict:
    """Aggregate signed weights of edges that target `target_node_id`.

    Returns a dict:
        {
            "feature":     {"pos": float, "neg": float, "signed": float},
            "embedding":   {"pos": float, "neg": float, "signed": float},
            "error_node":  {"pos": float, "neg": float, "signed": float},
            "total_signed": float,
            "n_edges_to_target": int,
        }

    Edges from sources whose `feature_type` is not in `node_type_to_category`
    raise ValueError (fail-loud so we don't silently drop signal).
    """
    node_lookup = {n["node_id"]: n for n in graph["nodes"]}
    edges_field = "links" if "links" in graph else "edges"

    sums = {cat: {"pos": 0.0, "neg": 0.0, "signed": 0.0}
            for cat in ("feature", "embedding", "error_node")}
    n_edges = 0

    for edge in graph[edges_field]:
        if edge["target"] != target_node_id:
            continue
        src_node = node_lookup.get(edge["source"])
        if src_node is None:
            # External / external-style source IDs (e.g. E_*_* token references) that
            # don't appear in the node list — skip with no fail to allow edges into
            # mid-graph nodes to use external token refs without crashing the audit.
            continue
        src_type = src_node.get("feature_type", "")
        if src_type == "logit":
            continue
        if src_type not in node_type_to_category:
            raise ValueError(
                f"unknown feature_type {src_type!r} on edge source {edge['source']!r}; "
                f"add to DEFAULT_NODE_TYPE_TO_CATEGORY in graph_loader.py"
            )
        category = node_type_to_category[src_type]
        weight = float(edge["weight"])
        sums[category]["signed"] += weight
        if weight >= 0:
            sums[category]["pos"] += weight
        else:
            sums[category]["neg"] += weight
        n_edges += 1

    sums["total_signed"] = sum(sums[c]["signed"] for c in ("feature", "embedding", "error_node"))
    sums["n_edges_to_target"] = n_edges
    return sums


def extract_edge_records_to_target(
    graph: dict,
    target_node_id: str,
    node_type_to_category: dict[str, Category] = DEFAULT_NODE_TYPE_TO_CATEGORY,
    filter_category: Category | None = None,
) -> list[dict]:
    """Return per-edge records for edges targeting `target_node_id`.

    Each record:
        {
            "source_id": str,
            "category": "feature" | "embedding" | "error_node",
            "layer": int | None,
            "feature": int | None,
            "ctx_idx": int | None,
            "signed_attribution": float,
            "activation": float,
        }
    Sorted by `signed_attribution` descending (most positive first).
    `activation` is the source node's recorded activation value (0.0 when the
    packed graph predates the field). One record per source node — the same
    (layer, feature) can appear at several ctx positions as separate records.

    Args:
        graph: parsed packed-graph dict.
        target_node_id: id of the measurement target node.
        node_type_to_category: feature_type -> category mapping.
        filter_category: if not None, only records of this category returned.
    """
    node_lookup = {n["node_id"]: n for n in graph["nodes"]}
    edges_field = "links" if "links" in graph else "edges"
    records = []
    for edge in graph[edges_field]:
        if edge["target"] != target_node_id:
            continue
        src_node = node_lookup.get(edge["source"])
        if src_node is None:
            continue  # external source ID (E_*_* style)
        src_type = src_node.get("feature_type", "")
        if src_type == "logit":
            continue
        if src_type not in node_type_to_category:
            raise ValueError(
                f"unknown feature_type {src_type!r} on edge source {edge['source']!r}"
            )
        category = node_type_to_category[src_type]
        if filter_category is not None and category != filter_category:
            continue
        # Parse node_id for (layer, feature_idx) — matches Stage 04's per-layer
        # 0-16383 indexing. Falls back to schema fields for test fixtures.
        layer, feature_idx = _parse_layer_feature_from_node(src_node)
        ctx_raw = src_node.get("ctx_idx")
        records.append({
            "source_id": edge["source"],
            "category": category,
            "layer": layer,
            "feature": feature_idx,
            "ctx_idx": int(ctx_raw) if isinstance(ctx_raw, (int, float)) else None,
            "signed_attribution": float(edge["weight"]),
            "activation": float(src_node.get("activation") or 0.0),
        })
    records.sort(key=lambda r: r["signed_attribution"], reverse=True)
    return records


def extract_feature_profile(
    graph: dict, target_node_id: str,
    node_type_to_category: dict[str, Category] = DEFAULT_NODE_TYPE_TO_CATEGORY,
) -> dict[tuple[int, int], float]:
    """Per-feature signed attribution to the measurement target.

    Returns dict keyed by (layer, feature_id) -> signed_attribution. Used by
    0f to build per-feature profiles across the 11-condition dataset (caller
    iterates over conditions and aggregates).

    Only `cross layer transcoder` source nodes (category 'feature') are
    included; embedding and error_node sources have their own profile path.
    """
    node_lookup = {n["node_id"]: n for n in graph["nodes"]}
    edges_field = "links" if "links" in graph else "edges"
    profile = {}
    for edge in graph[edges_field]:
        if edge["target"] != target_node_id:
            continue
        src_node = node_lookup.get(edge["source"])
        if src_node is None:
            continue
        src_type = src_node.get("feature_type", "")
        if src_type == "logit":
            continue
        if src_type not in node_type_to_category:
            raise ValueError(f"unknown feature_type {src_type!r}")
        if node_type_to_category[src_type] != "feature":
            continue
        # Parse node_id for (layer, feature_idx) — Stage-04-compatible indexing.
        L, F = _parse_layer_feature_from_node(src_node)
        if L is None or F is None:
            continue
        profile[(L, F)] = float(edge["weight"])
    return profile
