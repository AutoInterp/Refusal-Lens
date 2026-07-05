"""CPU unit tests for v2 upstream propagation (no GPU, no HF)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3] / "scripts/pipeline"))

from trace_propagate import build_key_graph  # noqa: E402


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
