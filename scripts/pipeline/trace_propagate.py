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


def normalized_parents(kg) -> dict:
    """Per-target normalized parent weights: w / (Σ|feature incoming w| + error_into[t])."""
    out: dict = {}
    for dst, srcs in kg["parents"].items():
        denom = sum(abs(w) for w in srcs.values()) + kg["error_into"].get(dst, 0.0)
        if denom > 0:
            out[dst] = {s: w / denom for s, w in srcs.items()}
    return out


def upstream_contributions(kg, seed_keys, *, k, tau) -> dict:
    nparents = normalized_parents(kg)
    agg_c: dict = {}
    agg_h: dict = {}
    for s in seed_keys:
        for akey, (c, h) in path_sums(nparents, s, k).items():
            if abs(c) > abs(agg_c.get(akey, 0.0)):      # aggregate by MAX-|contrib| (keep sign)
                agg_c[akey] = c
            agg_h[akey] = min(agg_h.get(akey, h), h)
    total = sum(abs(v) for v in agg_c.values())
    keep = {kk for kk, v in agg_c.items() if abs(v) >= tau}   # ABSOLUTE floor
    per_feature = {kk: {"contrib": agg_c[kk], "hop": agg_h[kk]} for kk in keep}
    kept = sum(abs(agg_c[kk]) for kk in keep)
    direct = sum(abs(w) for s in seed_keys for w in kg["parents"].get(s, {}).values())
    err = sum(kg["error_into"].get(s, 0.0) for s in seed_keys)
    return {"per_feature": per_feature,
            "coverage": (kept / total) if total > 0 else 0.0,
            "error_frac": (err / (direct + err)) if (direct + err) > 0 else 0.0}


def edge_delta_label(w_bare, w_jb, a_bare, a_jb, *, margin, eps=1e-9) -> dict:
    delta = w_jb - w_bare
    if a_bare <= eps:
        # source was inactive in bare; any jb connection is a new (active) mechanism
        return {"delta": delta, "act_term": 0.0, "edge_term": delta, "label": "active"}
    act_term = w_bare * (a_jb / a_bare - 1.0)
    edge_term = delta - act_term
    a, e = abs(act_term), abs(edge_term)
    if a >= (1 + margin) * e:
        label = "passive"
    elif e >= (1 + margin) * a:
        label = "active"
    else:
        label = "ambiguous"
    return {"delta": delta, "act_term": act_term, "edge_term": edge_term, "label": label}


def dominant_path(parents_jb, seed_key, k) -> dict:
    """For each ancestor (<=k hops), the edge list of its max Π|w| path to the seed,
    ordered SEED-ADJACENT edge FIRST. Double-buffered so a path grows at most one edge
    per round -> length capped at k."""
    best = {seed_key: (1.0, [])}          # node -> (max_abs_product, [edges seed-adjacent-first])
    for _ in range(k):
        prev = dict(best)                 # read last round, write this round: +1 edge/round
        for dst, srcs in parents_jb.items():
            if dst not in prev:
                continue
            prod_dst, path_dst = prev[dst]
            for src, w in srcs.items():
                if src == seed_key:
                    continue
                cand = abs(w) * prod_dst
                if cand > best.get(src, (0.0, None))[0] + 1e-15:
                    best[src] = (cand, path_dst + [(src, dst)])   # append -> seed-adjacent stays first
    return {kk: v[1] for kk, v in best.items() if kk != seed_key}


def _mechanism_for_path(edges, bare_kg, jb_kg, margin):
    """edges seed-adjacent-first. Scan from seed outward; first 'active' -> active_inhibitor;
    an 'ambiguous' before any active -> mixed; all 'passive' -> passive_cascade."""
    seen_ambiguous = False
    for (src, dst) in edges:
        wb = bare_kg["parents"].get(dst, {}).get(src, 0.0)
        wj = jb_kg["parents"].get(dst, {}).get(src, 0.0)
        ab = bare_kg["act"].get(src, 0.0)
        aj = jb_kg["act"].get(src, 0.0)
        lab = edge_delta_label(wb, wj, ab, aj, margin=margin)["label"]
        if lab == "active":
            return "active_inhibitor"
        if lab == "ambiguous":
            seen_ambiguous = True
    return "mixed" if seen_ambiguous else "passive_cascade"


def delta_decompose(bare_kg, jb_kg, seed_keys, *, k, tau, margin) -> dict:
    npar_j = normalized_parents(jb_kg)          # NORMALIZED: delta = change in fractional share
    npar_b = normalized_parents(bare_kg)
    agg_d: dict = {}
    agg_h: dict = {}
    for s in seed_keys:
        cj = path_sums(npar_j, s, k)
        cb = path_sums(npar_b, s, k)
        for a in set(cj) | set(cb):
            d = cj.get(a, (0.0, None))[0] - cb.get(a, (0.0, None))[0]
            if abs(d) > abs(agg_d.get(a, 0.0)):     # aggregate by MAX-|delta| (keep sign)
                agg_d[a] = d
            h = cj.get(a, (None, 10**9))[1]
            hb = cb.get(a, (None, 10**9))[1]
            agg_h[a] = min(agg_h.get(a, 10**9), h, hb)
    total = sum(abs(v) for v in agg_d.values())
    keep = {kk for kk, v in agg_d.items() if abs(v) >= tau}     # ABSOLUTE floor
    # dominant path (mechanism) on the NORMALIZED jb parents; first seed to reach a feature wins:
    dpaths: dict = {}
    for s in seed_keys:
        for a, edges in dominant_path(npar_j, s, k).items():
            if a not in dpaths:
                dpaths[a] = edges
    per_feature = {}
    for kk in keep:
        edges = dpaths.get(kk, [])
        mech = _mechanism_for_path(edges, bare_kg, jb_kg, margin) if edges else "none"
        per_feature[kk] = {"delta": agg_d[kk], "hop": agg_h[kk], "mechanism": mech}
    kept = sum(abs(agg_d[kk]) for kk in keep)
    direct = sum(abs(w) for s in seed_keys for w in jb_kg["parents"].get(s, {}).values())
    err = sum(jb_kg["error_into"].get(s, 0.0) for s in seed_keys)
    return {"per_feature": per_feature,
            "coverage": (kept / total) if total > 0 else 0.0,
            "error_frac": (err / (direct + err)) if (direct + err) > 0 else 0.0}


SEED_CLASSES = ("refusal_centric", "suppression", "amplification")


def assign_upstream_classes(contrib_by_class, delta_by_class) -> dict:
    # gather per-feature score/hop/mechanism per seed-class
    per_key: dict = {}
    def add(cls, key, score, hop, mech):
        per_key.setdefault(key, {})[cls] = {"score": score, "hop": hop, "mechanism": mech}
    for cls, out in contrib_by_class.items():
        for key, d in out["per_feature"].items():
            add(cls, key, d["contrib"], d["hop"], "none")
    for cls, out in delta_by_class.items():
        for key, d in out["per_feature"].items():
            add(cls, key, d["delta"], d["hop"], d["mechanism"])
    # per-class max|score| so idea-2 (level) and idea-3 (delta) are commensurable
    def _maxabs(out):
        vals = [abs(d.get("contrib", d.get("delta", 0.0))) for d in out["per_feature"].values()]
        return max(vals) if vals else 1.0
    class_norm = {}
    for cls, out in contrib_by_class.items(): class_norm[cls] = _maxabs(out) or 1.0
    for cls, out in delta_by_class.items():   class_norm[cls] = _maxabs(out) or 1.0
    result = {}
    for key, byc in per_key.items():
        dom = max(byc, key=lambda c: abs(byc[c]["score"]) / (class_norm.get(c, 1.0) or 1.0))
        result[key] = {"upstream_class": dom, "hop": byc[dom]["hop"],
                       "mechanism": byc[dom]["mechanism"]}
    return result


def bake_upstream_classes(graph, feature_class_map):
    for n in graph["nodes"]:
        if n.get("feature_type") != FEATURE_TYPE:
            continue
        if n.get("rl_trace_class") in SEED_CLASSES:
            n["rl_trace_hop"] = 0
            n["rl_trace_mechanism"] = "seed"
            continue
        key = (int(n["layer"]), int(n["feature"]))
        info = feature_class_map.get(key)
        if info is not None:
            n["rl_trace_upstream_class"] = info["upstream_class"]
            n["rl_trace_hop"] = info["hop"]
            n["rl_trace_mechanism"] = info["mechanism"]
    return graph
