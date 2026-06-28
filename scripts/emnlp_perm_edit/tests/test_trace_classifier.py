"""CPU unit tests for the bare→comply trace classifier (no GPU, no HF)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "pipeline"))  # scripts/pipeline

from trace_classifier import find_refusal_logit_id, aggregate_features  # noqa: E402


def _feat(node_id, feature, layer, ctx_idx, activation):
    return {"node_id": node_id, "feature": feature, "layer": str(layer),
            "ctx_idx": ctx_idx, "feature_type": "cross layer transcoder",
            "activation": activation}


def _graph(nodes, links, tokens=None):
    return {"metadata": {"prompt_tokens": tokens or ["<bos>", "model", "\n"]},
            "nodes": nodes, "links": links}


def test_find_refusal_logit_id():
    g = _graph(
        [_feat("1_10_2", 10, 1, 2, 5.0),
         {"node_id": "L", "feature_type": "logit", "clerp": "Output refusal"}],
        [])
    assert find_refusal_logit_id(g) == "L"


def test_aggregate_sums_signed_edges_and_max_act():
    # feature (layer1, feat10) has two nodes both feeding the logit: +3 and +2 -> +5
    nodes = [_feat("1_10_2", 10, 1, 2, 5.0), _feat("1_10_6", 10, 1, 6, 9.0),
             _feat("3_20_2", 20, 3, 2, 4.0),
             {"node_id": "L", "feature_type": "logit"}]
    links = [{"source": "1_10_2", "target": "L", "weight": 3.0},
             {"source": "1_10_6", "target": "L", "weight": 2.0},
             {"source": "3_20_2", "target": "L", "weight": -4.0}]
    agg = aggregate_features(_graph(nodes, links))
    assert agg[(1, 10)]["edge"] == 5.0
    assert agg[(1, 10)]["act"] == 9.0
    assert sorted(agg[(1, 10)]["node_ids"]) == ["1_10_2", "1_10_6"]
    assert agg[(3, 20)]["edge"] == -4.0


def test_aggregate_feature_without_logit_link_has_zero_edge():
    nodes = [_feat("1_10_2", 10, 1, 2, 5.0), {"node_id": "L", "feature_type": "logit"}]
    agg = aggregate_features(_graph(nodes, []))
    assert agg[(1, 10)]["edge"] == 0.0
    assert agg[(1, 10)]["act"] == 5.0
