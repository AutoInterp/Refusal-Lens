"""Local unit tests for the Qwen subcircuits + Top-K sweep orchestration.

Pure-Python (no GPU, no model downloads). Run from repo root:
    PYTHONPATH=scripts .venv/bin/python3 scripts/emnlp_perm_edit/tests/test_qwen_subcircuit_orchestration.py

Covers:
  1. select_topk ranking semantics (pos/neg/abs/activation) + K clamping
  2. aggregate_feature_records (position-instance summing, activation max,
     non-feature categories excluded, descending sort)
  3. build_zero_interventions (slice(None) positions, 0.0 value, dedup)
  4. proxy delta conversion math (delta = delta_norm * ||r_unnorm|| preserves
     the normalized-basis edit; mirrors 00_edge_ablation_runtime_qwen.py)
  5. compare_features replica vs hand-computed buckets
  6. aggregator flip-rate logic (break/recovery denominators) + Wilson CI + knees
  7. graph_loader edge-record activation/ctx fields on a synthetic packed graph
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import graph_loader  # noqa: E402
import qwen_rebuild_attribution_index as rebuild  # noqa: E402
import qwen_subcircuits_aggregate as agg  # noqa: E402


def _load_sweep_module():
    spec = importlib.util.spec_from_file_location(
        "topk_sweep", SCRIPT_DIR / "00_topk_circuit_sweep_qwen.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_select_topk_and_clamp(m):
    recs = [
        {"layer": 1, "feature": 10, "signed_attribution": 3.0, "activation": 0.5},
        {"layer": 2, "feature": 20, "signed_attribution": 1.0, "activation": 9.0},
        {"layer": 3, "feature": 30, "signed_attribution": -2.0, "activation": 4.0},
        {"layer": 4, "feature": 40, "signed_attribution": -5.0, "activation": 1.0},
    ]
    recs.sort(key=lambda r: r["signed_attribution"], reverse=True)
    assert [r["feature"] for r in m.select_topk(recs, "pos", 2)] == [10, 20]
    assert [r["feature"] for r in m.select_topk(recs, "neg", 2)] == [40, 30]
    assert [r["feature"] for r in m.select_topk(recs, "abs", 2)] == [40, 10]
    assert [r["feature"] for r in m.select_topk(recs, "activation", 2)] == [20, 30]
    assert len(m.select_topk(recs, "pos", 99)) == 4          # clamps to available
    assert m.select_topk([], "pos", 5) == []
    s = m.summarize_selection(m.select_topk(recs, "pos", 2))
    assert s["delta_norm_basis"] == 4.0 and s["n_used"] == 2
    print("  [1] select_topk + clamping OK")


def test_aggregate_feature_records(m):
    agg_recs = m.aggregate_feature_records([
        {"category": "feature", "layer": 1, "feature": 10, "signed_attribution": 1.0, "activation": 2.0},
        {"category": "feature", "layer": 1, "feature": 10, "signed_attribution": -0.25, "activation": 5.0},
        {"category": "feature", "layer": 2, "feature": 20, "signed_attribution": 0.5, "activation": 1.0},
        {"category": "embedding", "layer": 0, "feature": None, "signed_attribution": 9.9, "activation": 0.0},
        {"category": "error_node", "layer": 1, "feature": 7, "signed_attribution": 4.0, "activation": 0.0},
    ])
    assert len(agg_recs) == 2                                  # embedding/error excluded
    assert agg_recs[0]["signed_attribution"] == 0.75           # summed across instances
    assert agg_recs[0]["activation"] == 5.0                    # max across instances
    assert agg_recs[0]["signed_attribution"] >= agg_recs[1]["signed_attribution"]  # desc
    print("  [2] aggregate_feature_records OK")


def test_zero_interventions(m):
    chosen = [
        {"layer": 4, "feature": 40, "signed_attribution": -5.0, "activation": 1.0},
        {"layer": 4, "feature": 40, "signed_attribution": -1.0, "activation": 0.5},  # dup
        {"layer": 3, "feature": 30, "signed_attribution": -2.0, "activation": 4.0},
    ]
    iv = m.build_zero_interventions(chosen)
    assert len(iv) == 2                                        # deduped
    L, pos, F, val = iv[0]
    assert (L, F, val) == (4, 40, 0.0) and pos == slice(None)  # Stage-08 'all' convention
    print("  [3] build_zero_interventions OK")


def test_delta_conversion():
    """h_new·r̂_normalized must equal h·r̂_normalized − delta_norm when the hook
    receives (r_unnorm, delta_norm × ||r_unnorm||) — the v2 Qwen convention."""
    import torch
    sys.path.insert(0, str(SCRIPT_DIR))
    from edge_ablation_hook import make_scalar_rhat_subtraction_hook
    torch.manual_seed(0)
    r_unnorm = torch.randn(64) * 2.0
    r_norm = r_unnorm.norm().item()
    r_hat_unit = r_unnorm / r_norm
    h = torch.randn(1, 3, 64)
    delta_norm = 7.3
    hook = make_scalar_rhat_subtraction_hook(r_unnorm, delta_norm * r_norm, position_mode="all")
    h_new = hook(None, None, h.clone())
    before = (h[0, -1] @ r_hat_unit).item()
    after = (h_new[0, -1] @ r_hat_unit).item()
    assert abs((before - after) - delta_norm) < 1e-3, (before, after, delta_norm)
    print("  [4] proxy delta conversion math OK")


def test_compare_features_replica():
    bare = {"L1:F1": {"attribution": 2.0},   # sign flip vs cls
            "L1:F2": {"attribution": 1.0},   # dampened (delta -0.5)
            "L1:F3": {"attribution": -1.0},  # amplified_anti (delta -0.5)
            "L1:F4": {"attribution": 0.5}}   # bare only
    cls = {"L1:F1": {"attribution": -2.0},
           "L1:F2": {"attribution": 0.5},
           "L1:F3": {"attribution": -1.5},
           "L1:F5": {"attribution": 3.0}}    # cls only
    c = rebuild.compare_features(bare, cls)
    assert c["n_shared"] == 3 and c["n_bare_only"] == 1 and c["n_cls_only"] == 1
    assert c["n_sign_flipped"] == 1 and c["top_sign_flipped"][0]["key"] == "L1:F1"
    assert c["n_dampened"] == 1 and c["top_dampened"][0]["key"] == "L1:F2"
    assert c["n_amplified_anti"] == 1 and c["top_amplified_anti"][0]["key"] == "L1:F3"
    print("  [5] compare_features replica OK")


def test_aggregator_rates_and_knees():
    baseline = {(0, "bare"): "REFUSE", (1, "bare"): "COMPLY",   # prompt 1 ineligible for break
                (0, "jb_fiction"): "COMPLY", (1, "jb_fiction"): "REFUSE"}  # prompt 1 ineligible for recovery
    records = [
        {"prompt_idx": 0, "condition": "bare", "classification": "COMPLY", "coherent": True},
        {"prompt_idx": 1, "condition": "bare", "classification": "COMPLY", "coherent": True},
        {"prompt_idx": 0, "condition": "jb_fiction", "classification": "REFUSE", "coherent": True},
        {"prompt_idx": 1, "condition": "jb_fiction", "classification": "REFUSE", "coherent": False},
    ]
    rates = agg.cell_rates(records, baseline)
    assert rates["break"]["n"] == 1 and rates["break"]["k"] == 1 and rates["break"]["rate"] == 1.0
    assert rates["recovery"]["n"] == 1 and rates["recovery"]["rate"] == 1.0
    assert rates["n_incoherent"] == 1
    r, lo, hi = agg.wilson_ci(8, 10)
    assert 0.49 < lo < r < hi <= 1.0
    assert agg.wilson_ci(0, 0) == (0.0, 0.0, 0.0)
    assert agg.find_knee([(1, 0.1), (5, 0.4), (10, 0.8)], 0.5) == 10
    assert agg.find_knee([(1, 0.1)], 0.5) is None
    print("  [6] aggregator rates + Wilson + knees OK")


def test_graph_loader_activation_fields():
    graph = {
        "nodes": [
            {"node_id": "5_123_2", "feature_type": "cross layer transcoder",
             "activation": 3.5, "ctx_idx": 2},
            {"node_id": "T", "feature_type": "logit", "is_target_logit": True},
        ],
        "links": [{"source": "5_123_2", "target": "T", "weight": -1.25}],
    }
    tid = graph_loader.find_measurement_target_node_id(graph)
    recs = graph_loader.extract_edge_records_to_target(graph, tid)
    assert len(recs) == 1
    r = recs[0]
    assert r["layer"] == 5 and r["feature"] == 123 and r["ctx_idx"] == 2
    assert r["activation"] == 3.5 and r["signed_attribution"] == -1.25
    print("  [7] graph_loader activation/ctx fields OK")


def main():
    m = _load_sweep_module()
    test_select_topk_and_clamp(m)
    test_aggregate_feature_records(m)
    test_zero_interventions(m)
    test_delta_conversion()
    test_compare_features_replica()
    test_aggregator_rates_and_knees()
    test_graph_loader_activation_fields()
    print("\nALL QWEN SUBCIRCUIT ORCHESTRATION TESTS PASS")


if __name__ == "__main__":
    main()
