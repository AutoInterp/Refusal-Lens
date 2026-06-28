"""Pure classifier for the bare→comply trace view.

Given parsed circuit-tracer graph dicts (bare + jailbroken) for one prompt,
classify each feature node as refusal_centric / suppression / amplification /
neutral, using the signed edge into the refusal logit and the bare↔jb activation
delta. Pure and unit-tested; the assembler does I/O.
"""
from __future__ import annotations

FEATURE_TYPE = "cross layer transcoder"


def find_refusal_logit_id(graph) -> str | None:
    for n in graph["nodes"]:
        if n.get("feature_type") == "logit":
            return n["node_id"]
    return None


def aggregate_features(graph) -> dict:
    """key (layer:int, feature:int) -> {edge: float, act: float, node_ids: [str]}."""
    logit_id = find_refusal_logit_id(graph)
    # node_id -> (layer, feature) for feature nodes only
    nid_to_key = {}
    agg: dict = {}
    for n in graph["nodes"]:
        if n.get("feature_type") != FEATURE_TYPE:
            continue
        key = (int(n["layer"]), int(n["feature"]))
        nid_to_key[n["node_id"]] = key
        e = agg.setdefault(key, {"edge": 0.0, "act": 0.0, "node_ids": []})
        e["act"] = max(e["act"], float(n.get("activation") or 0.0))
        e["node_ids"].append(n["node_id"])
    if logit_id is not None:
        for l in graph["links"]:
            if l.get("target") != logit_id:
                continue
            key = nid_to_key.get(l.get("source"))
            if key is None:
                continue
            agg[key]["edge"] += float(l.get("weight") or 0.0)
    return agg
