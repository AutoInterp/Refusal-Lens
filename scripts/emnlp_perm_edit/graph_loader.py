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
