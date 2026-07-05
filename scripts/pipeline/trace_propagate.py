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


def path_sums(parents, seed_key, k) -> dict:
    """{ancestor_key: (signed_contrib, shortest_hop)} over paths of length <= k."""
    contrib: dict = {}
    hop: dict = {}
    frontier = {seed_key: 1.0}
    for depth in range(1, k + 1):
        nxt: dict = {}
        for node, acc in frontier.items():
            for src, w in parents.get(node, {}).items():
                if src == seed_key:
                    continue
                c = acc * w
                contrib[src] = contrib.get(src, 0.0) + c
                if src not in hop:
                    hop[src] = depth
                nxt[src] = nxt.get(src, 0.0) + c
        frontier = nxt
        if not frontier:
            break
    return {kk: (contrib[kk], hop[kk]) for kk in contrib}


def upstream_contributions(kg, seed_keys, *, k, tau) -> dict:
    parents = kg["parents"]
    agg_c: dict = {}
    agg_h: dict = {}
    for s in seed_keys:
        for akey, (c, h) in path_sums(parents, s, k).items():
            agg_c[akey] = agg_c.get(akey, 0.0) + c
            agg_h[akey] = min(agg_h.get(akey, h), h)
    total = sum(abs(v) for v in agg_c.values())
    keep = {kk for kk, v in agg_c.items() if total > 0 and abs(v) >= tau * total}
    per_feature = {kk: {"contrib": agg_c[kk], "hop": agg_h[kk]} for kk in keep}
    kept = sum(abs(agg_c[kk]) for kk in keep)
    direct = sum(abs(w) for s in seed_keys for w in parents.get(s, {}).values())
    err = sum(kg["error_into"].get(s, 0.0) for s in seed_keys)
    return {"per_feature": per_feature,
            "coverage": (kept / total) if total > 0 else 0.0,
            "error_frac": (err / (direct + err)) if (direct + err) > 0 else 0.0}
