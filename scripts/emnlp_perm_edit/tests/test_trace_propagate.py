"""CPU unit tests for v2 upstream propagation (no GPU, no HF)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3] / "scripts/pipeline"))

from trace_propagate import build_key_graph  # noqa: E402
from trace_propagate import path_sums, normalized_parents, upstream_contributions  # noqa: E402
from trace_propagate import edge_delta_label  # noqa: E402
from trace_propagate import dominant_path, delta_decompose  # noqa: E402


def _feat(node_id, feature, layer, activation):
    return {"node_id": node_id, "feature": feature, "layer": str(layer),
            "ctx_idx": 0, "feature_type": "cross layer transcoder", "activation": activation}


def _graph(nodes, links):
    return {"metadata": {}, "nodes": nodes, "links": links}


def test_build_key_graph_sums_positions_and_tracks_error():
    nodes = [_feat("1_10_2", 10, 1, 5.0), _feat("1_10_6", 10, 1, 9.0),   # same key (1,10)
             _feat("2_20_2", 20, 2, 4.0),
             {"node_id": "E1", "feature_type": "mlp reconstruction error"},
             {"node_id": "L", "feature_type": "logit"}]
    links = [{"source": "1_10_2", "target": "2_20_2", "weight": 3.0},
             {"source": "1_10_6", "target": "2_20_2", "weight": 1.0},   # -> summed 4.0
             {"source": "E1", "target": "2_20_2", "weight": -7.0},       # error leakage
             {"source": "2_20_2", "target": "L", "weight": 2.0}]
    kg = build_key_graph(_graph(nodes, links))
    assert kg["parents"][(2, 20)][(1, 10)] == 4.0
    assert kg["parents"][("LOGIT",)][(2, 20)] == 2.0
    assert kg["act"][(1, 10)] == 9.0 and kg["act"][(2, 20)] == 4.0
    assert kg["error_into"][(2, 20)] == 7.0


def test_path_sums_products_and_hops():
    # (0,5) --5--> (1,10) --4--> seed(2,20) ; also (0,5) --(-3)--> seed
    parents = {(2, 20): {(1, 10): 4.0, (0, 5): -3.0}, (1, 10): {(0, 5): 5.0}}
    ps = path_sums(parents, (2, 20), k=3)
    assert ps[(1, 10)][0] == 4.0 and ps[(1, 10)][1] == 1
    # (0,5): direct -3  +  via (1,10): 4*5=20  => 17 ; shortest hop = 1
    assert ps[(0, 5)][0] == 17.0 and ps[(0, 5)][1] == 1


def test_path_sums_respects_depth_cap():
    parents = {(2, 20): {(1, 10): 4.0}, (1, 10): {(0, 5): 5.0}}
    assert (0, 5) not in path_sums(parents, (2, 20), k=1)   # hop-2 path excluded at k=1


def test_normalized_parents_shares_and_error_dilution():
    # (2,20) incoming |3|+|-1| = 4 -> shares 0.75 / -0.25
    kg = {"parents": {(2, 20): {(1, 10): 3.0, (0, 5): -1.0}}, "act": {}, "error_into": {}}
    npar = normalized_parents(kg)
    assert npar[(2, 20)][(1, 10)] == 0.75 and npar[(2, 20)][(0, 5)] == -0.25
    # error leakage dilutes the feature shares: 3 / (3 + 1) = 0.75
    kg2 = {"parents": {(2, 20): {(1, 10): 3.0}}, "act": {}, "error_into": {(2, 20): 1.0}}
    assert normalized_parents(kg2)[(2, 20)][(1, 10)] == 0.75


def test_upstream_contributions_absolute_floor_and_coverage():
    kg = {"parents": {(2, 20): {(1, 10): 3.0, (0, 5): -1.0}, (1, 10): {(0, 5): 2.0}},
          "act": {}, "error_into": {}}
    # normalized: (2,20)->(1,10)=0.75, (0,5)=-0.25 ; (1,10)->(0,5)=1.0
    # path-sums: (1,10)=0.75 ; (0,5) = -0.25 + 0.75*1.0 = 0.5
    out = upstream_contributions(kg, [(2, 20)], k=3, tau=0.6)   # absolute floor
    pf = out["per_feature"]
    assert set(pf) == {(1, 10)}                                  # keep 0.75, drop 0.5
    assert abs(pf[(1, 10)]["contrib"] - 0.75) < 1e-9
    assert abs(out["coverage"] - 0.75 / 1.25) < 1e-9             # kept / (0.75+0.5)


def test_upstream_contributions_error_frac():
    kg = {"parents": {(2, 20): {(1, 10): 3.0}}, "act": {}, "error_into": {(2, 20): 1.0}}
    out = upstream_contributions(kg, [(2, 20)], k=3, tau=0.1)
    assert abs(out["error_frac"] - 0.25) < 1e-9                  # 1 / (3 + 1)


def test_edge_delta_label_passive_active_ambiguous_newlyactive():
    # passive: activation up, weight barely moves
    r = edge_delta_label(10.0, 12.0, 2.0, 3.0, margin=0.25)
    assert r["act_term"] == 5.0 and r["label"] == "passive"     # |5| >= 1.25*|-3|=3.75
    # active: activation flat, weight drops
    r = edge_delta_label(10.0, 4.0, 2.0, 2.0, margin=0.25)
    assert r["act_term"] == 0.0 and r["label"] == "active"
    # newly active source
    r = edge_delta_label(0.0, 8.0, 0.0, 1.0, margin=0.25)
    assert r["label"] == "active" and r["delta"] == 8.0
    # ambiguous: neither term dominates by margin
    r = edge_delta_label(10.0, 19.5, 2.0, 3.0, margin=0.25)   # act=5, edge=4.5
    assert r["label"] == "ambiguous"


def test_dominant_path_picks_largest_product():
    # (0,5)->(1,10)->S and a weaker direct (0,5)->S ; dominant is the 2-hop (5*4=20 > 1)
    parents = {(2, 20): {(1, 10): 4.0, (0, 5): 1.0}, (1, 10): {(0, 5): 5.0}}
    dp = dominant_path(parents, (2, 20), k=3)
    assert dp[(0, 5)] == [((1, 10), (2, 20)), ((0, 5), (1, 10))]  # seed-adjacent edge first


def test_delta_decompose_active_redistribution():
    # seed (2,20) inputs REWIRED: (1,10) share 0.75->0.25 (lost), (1,11) 0.25->0.75 (gained).
    # activations flat -> the change is weight-driven -> active_inhibitor.
    bare = {"parents": {(2, 20): {(1, 10): 3.0, (1, 11): 1.0}},
            "act": {(1, 10): 2.0, (1, 11): 2.0}, "error_into": {}}
    jb = {"parents": {(2, 20): {(1, 10): 1.0, (1, 11): 3.0}},
          "act": {(1, 10): 2.0, (1, 11): 2.0}, "error_into": {}}
    out = delta_decompose(bare, jb, [(2, 20)], k=3, tau=0.1, margin=0.25)
    pf = out["per_feature"]
    assert abs(pf[(1, 10)]["delta"] - (-0.5)) < 1e-9   # 0.25 - 0.75
    assert abs(pf[(1, 11)]["delta"] - (0.5)) < 1e-9    # 0.75 - 0.25
    assert pf[(1, 10)]["mechanism"] == "active_inhibitor" and pf[(1, 10)]["hop"] == 1


def test_delta_decompose_passive_redistribution():
    # (1,10) gains share (0.5->0.75) via activation 1->3, weight tracks it 2->6 -> passive.
    bare = {"parents": {(2, 20): {(1, 10): 2.0, (1, 11): 2.0}},
            "act": {(1, 10): 1.0, (1, 11): 1.0}, "error_into": {}}
    jb = {"parents": {(2, 20): {(1, 10): 6.0, (1, 11): 2.0}},
          "act": {(1, 10): 3.0, (1, 11): 1.0}, "error_into": {}}
    out = delta_decompose(bare, jb, [(2, 20)], k=3, tau=0.1, margin=0.25)
    pf = out["per_feature"]
    assert abs(pf[(1, 10)]["delta"] - 0.25) < 1e-9     # 0.75 - 0.5
    assert pf[(1, 10)]["mechanism"] == "passive_cascade"
