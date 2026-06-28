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


from trace_classifier import classify_pair, model_token_positions  # noqa: E402


def test_model_token_positions():
    g = _graph([], [], tokens=["<bos>", "user", "model", "\n"])
    # last 'model' at index 2 -> positions {2, 3}
    assert model_token_positions(g) == {2, 3}


def test_classify_suppression_amplification_and_neutral():
    L = {"node_id": "L", "feature_type": "logit"}
    # A: pro-refusal (edge +50), active in bare (10) drops in jb (1) -> suppression on jb side
    # B: anti-refusal (edge -40), absent in bare, active in jb (8) -> amplification
    # F: jb-only, small edge (-5) ranked OUT by top_n=2 -> unsigned -> neutral (tests the gate)
    bare = _graph(
        [_feat("1_10_2", 10, 1, 2, 10.0), L],
        [{"source": "1_10_2", "target": "L", "weight": 50.0}])
    jb = _graph(
        [_feat("1_10_5", 10, 1, 5, 1.0), _feat("3_20_5", 20, 3, 5, 8.0),
         _feat("4_40_5", 40, 4, 5, 8.0), L],
        [{"source": "1_10_5", "target": "L", "weight": 50.0},
         {"source": "3_20_5", "target": "L", "weight": -40.0},
         {"source": "4_40_5", "target": "L", "weight": -5.0}])
    out = classify_pair(bare, jb, top_n=2, delta=0.30)
    assert out["bare"]["1_10_2"] == "refusal_centric"
    assert out["jb"]["1_10_5"] == "suppression"
    assert out["jb"]["3_20_5"] == "amplification"
    assert out["jb"]["4_40_5"] == "neutral"          # ranked out of top_n -> unsigned
    # evidence has one row per non-neutral feature, with both edges/acts
    feats = {(r["layer"], r["feature"]): r for r in out["evidence"]}
    assert feats[(1, 10)]["class"] == "suppression"
    assert feats[(1, 10)]["act_bare"] == 10.0 and feats[(1, 10)]["act_jb"] == 1.0
    assert feats[(3, 20)]["class"] == "amplification"


from trace_classifier import bake_trace_classes  # noqa: E402


def test_bake_sets_class_and_defaults_neutral():
    g = _graph([_feat("1_10_2", 10, 1, 2, 5.0), _feat("1_11_2", 11, 1, 2, 5.0),
                {"node_id": "L", "feature_type": "logit"}], [])
    bake_trace_classes(g, {"1_10_2": "refusal_centric"})
    by_id = {n["node_id"]: n for n in g["nodes"]}
    assert by_id["1_10_2"]["rl_trace_class"] == "refusal_centric"
    assert by_id["1_11_2"]["rl_trace_class"] == "neutral"      # default
    assert "rl_trace_class" not in by_id["L"]                  # logit untouched
