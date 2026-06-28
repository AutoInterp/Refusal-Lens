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


def model_token_positions(graph) -> set:
    toks = graph["metadata"].get("prompt_tokens") or []
    last = max((i for i, t in enumerate(toks) if str(t).strip() == "model"), default=None)
    if last is None:
        return set()
    return set(range(last, len(toks)))


def _signed(agg, top_n):
    """feature key -> signed edge, but only for features in the top_n by |edge|."""
    ranked = sorted(agg.items(), key=lambda kv: abs(kv[1]["edge"]), reverse=True)
    keep = {k for k, _ in ranked[:top_n] if abs(agg[k]["edge"]) > 0.0}
    return {k: agg[k]["edge"] for k in keep}


def classify_pair(bare, jb, *, top_n=20, delta=0.30, model_token_gate=False) -> dict:
    agg_b, agg_j = aggregate_features(bare), aggregate_features(jb)
    sign_b, sign_j = _signed(agg_b, top_n), _signed(agg_j, top_n)

    gate_pos = model_token_positions(bare) if model_token_gate else None
    bare_ctx = {}  # feature key -> set(ctx_idx) in bare
    for n in bare["nodes"]:
        if n.get("feature_type") == FEATURE_TYPE:
            bare_ctx.setdefault((int(n["layer"]), int(n["feature"])), set()).add(int(n["ctx_idx"]))

    bare_cls, jb_cls, evidence = {}, {}, []
    keys = set(agg_b) | set(agg_j)
    for key in keys:
        eb = agg_b.get(key, {}).get("act", 0.0)
        ej = agg_j.get(key, {}).get("act", 0.0)
        edge_b = sign_b.get(key, 0.0)
        edge_j = sign_j.get(key, 0.0)
        pro_edge = edge_b if edge_b > 0 else (edge_j if (key not in agg_b and edge_j > 0) else 0.0)
        is_refusal = pro_edge > 0.0
        if is_refusal and gate_pos is not None:
            if not (bare_ctx.get(key, set()) & gate_pos):
                is_refusal = False

        suppressed = is_refusal and (ej <= eb * (1 - delta))
        amplified = (edge_j < 0.0) and (eb == 0.0 and ej > 0.0 or ej >= eb * (1 + delta))

        cls = "neutral"
        if amplified:
            cls = "amplification"
        elif suppressed:
            cls = "suppression"
        elif is_refusal:
            cls = "refusal_centric"

        # bare-side class: refusal_centric if it qualifies (suppression is a jb-side label)
        bcls = "refusal_centric" if is_refusal else "neutral"
        for nid in agg_b.get(key, {}).get("node_ids", []):
            bare_cls[nid] = bcls
        for nid in agg_j.get(key, {}).get("node_ids", []):
            jb_cls[nid] = cls if cls != "neutral" else ("refusal_centric" if is_refusal else "neutral")

        if cls != "neutral" or is_refusal:
            evidence.append({"layer": key[0], "feature": key[1], "class": cls,
                             "edge_bare": edge_b, "edge_jb": edge_j,
                             "act_bare": eb, "act_jb": ej})
    evidence.sort(key=lambda r: ({"amplification": 0, "suppression": 1,
                                  "refusal_centric": 2, "neutral": 3}[r["class"]],
                                 -max(r["act_bare"], r["act_jb"])))
    return {"bare": bare_cls, "jb": jb_cls, "evidence": evidence}
