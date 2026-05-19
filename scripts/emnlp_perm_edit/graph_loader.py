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
            "signed_attribution": float,
        }
    Sorted by `signed_attribution` descending (most positive first).

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
        # Layer field in packed graphs is a string ("0", "13"); coerce to int when present
        layer_raw = src_node.get("layer")
        layer = int(layer_raw) if layer_raw is not None and str(layer_raw).lstrip("-").isdigit() else None
        records.append({
            "source_id": edge["source"],
            "category": category,
            "layer": layer,
            "feature": src_node.get("feature"),
            "signed_attribution": float(edge["weight"]),
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
        layer_raw = src_node.get("layer")
        F = src_node.get("feature")
        if layer_raw is None or F is None:
            continue
        L = int(layer_raw) if str(layer_raw).lstrip("-").isdigit() else None
        if L is None:
            continue
        profile[(L, F)] = float(edge["weight"])
    return profile
