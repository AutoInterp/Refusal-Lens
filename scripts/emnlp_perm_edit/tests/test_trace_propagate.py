"""CPU unit tests for v2 upstream propagation (no GPU, no HF)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3] / "scripts/pipeline"))

from trace_propagate import build_key_graph  # noqa: E402
from trace_propagate import path_sums, upstream_contributions  # noqa: E402


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


def test_upstream_contributions_threshold_and_coverage():
    parents = {(2, 20): {(1, 10): 4.0, (0, 5): -3.0, (9, 9): 0.5}, (1, 10): {(0, 5): 5.0}}
    kg = {"parents": parents, "act": {}, "error_into": {(2, 20): 0.0}}
    out = upstream_contributions(kg, [(2, 20)], k=3, tau=0.10)
    # contribs: (1,10)=4, (0,5)=17, (9,9)=0.5 ; total=21.5 ; tau*total=2.15 -> drop (9,9)
    pf = out["per_feature"]
    assert set(pf) == {(1, 10), (0, 5)}
    assert pf[(0, 5)]["contrib"] == 17.0
    assert abs(out["coverage"] - 21.0 / 21.5) < 1e-9
