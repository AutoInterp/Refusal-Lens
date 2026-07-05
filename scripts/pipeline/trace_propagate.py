"""Pure upstream-propagation for the v2 trace view.

Operates on a FEATURE-KEY-level digraph so bare and jb graphs (different prompts)
correspond by (layer, feature). Idea 2 = signed path-sum upstream of refusal_centric
seeds; idea 3 = per-edge activation-vs-weight delta split propagated along the
dominant path for suppression/amplification seeds. Pure; the assembler does I/O.
"""
from __future__ import annotations

FEATURE_TYPE = "cross layer transcoder"
ERROR_TYPE = "mlp reconstruction error"
LOGIT_KEY = ("LOGIT",)


def _key_of(node):
    ft = node.get("feature_type")
    if ft == FEATURE_TYPE:
        return (int(node["layer"]), int(node["feature"]))
    if ft == "logit":
        return LOGIT_KEY
    return None  # error / embedding / other: not a traversable feature key


def build_key_graph(graph) -> dict:
    key_of, is_error = {}, {}
    act: dict = {}
    for n in graph["nodes"]:
        nid = n["node_id"]
        k = _key_of(n)
        key_of[nid] = k
        is_error[nid] = (n.get("feature_type") == ERROR_TYPE)
        if k is not None and k != LOGIT_KEY:
            act[k] = max(act.get(k, 0.0), float(n.get("activation") or 0.0))
    parents: dict = {}
    error_into: dict = {}
    for link in graph["links"]:
        s, t, w = link.get("source"), link.get("target"), link.get("weight")
        if s is None or t is None or w is None:
            continue
        dk = key_of.get(t)
        if dk is None:
            continue
        if is_error.get(s):
            error_into[dk] = error_into.get(dk, 0.0) + abs(float(w))
            continue
        sk = key_of.get(s)
        if sk is None:
            continue
        parents.setdefault(dk, {})[sk] = parents.setdefault(dk, {}).get(sk, 0.0) + float(w)
    return {"parents": parents, "act": act, "error_into": error_into}
