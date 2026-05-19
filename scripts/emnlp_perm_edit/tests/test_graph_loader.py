"""Tests for the packed-graph loader library."""
from __future__ import annotations

import gzip
import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graph_loader import (  # noqa: E402
    DEFAULT_NODE_TYPE_TO_CATEGORY,
    aggregate_edge_attributions,
    find_measurement_target_node_id,
    load_packed_graph,
)


def _write_packed(path: Path, data: dict) -> None:
    with gzip.open(path, "wt") as f:
        json.dump(data, f)


def test_load_packed_graph_roundtrip():
    data = {"nodes": [{"node_id": "n0", "feature_type": "embedding"}],
            "links": [{"source": "n0", "target": "n0", "weight": 1.0}]}
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test.json.gz"
        _write_packed(path, data)
        loaded = load_packed_graph(path)
    assert loaded == data


def test_aggregate_edge_attributions_basic():
    graph = {
        "nodes": [
            {"node_id": "f1", "feature_type": "cross layer transcoder"},
            {"node_id": "f2", "feature_type": "cross layer transcoder"},
            {"node_id": "e1", "feature_type": "embedding"},
            {"node_id": "err1", "feature_type": "mlp reconstruction error"},
            {"node_id": "target", "feature_type": "logit", "is_target_logit": True},
        ],
        "links": [
            {"source": "f1", "target": "target", "weight": +1.5},
            {"source": "f2", "target": "target", "weight": -0.5},
            {"source": "e1", "target": "target", "weight": +0.2},
            {"source": "err1", "target": "target", "weight": -0.1},
            # An edge that does NOT target the measurement node — should be ignored
            {"source": "f1", "target": "f2", "weight": 99.0},
        ],
    }
    sums = aggregate_edge_attributions(graph, target_node_id="target")
    assert sums["feature"]["pos"] == pytest.approx(1.5)
    assert sums["feature"]["neg"] == pytest.approx(-0.5)
    assert sums["feature"]["signed"] == pytest.approx(1.0)
    assert sums["embedding"]["signed"] == pytest.approx(0.2)
    assert sums["error_node"]["signed"] == pytest.approx(-0.1)
    assert sums["total_signed"] == pytest.approx(1.0 + 0.2 - 0.1)


def test_aggregate_skips_edges_not_targeting_measurement_node():
    graph = {
        "nodes": [
            {"node_id": "f1", "feature_type": "cross layer transcoder"},
            {"node_id": "target", "feature_type": "logit", "is_target_logit": True},
        ],
        "links": [
            {"source": "f1", "target": "f1", "weight": 10.0},  # self-loop, not to target
        ],
    }
    sums = aggregate_edge_attributions(graph, target_node_id="target")
    assert sums["feature"]["signed"] == 0.0
    assert sums["total_signed"] == 0.0


def test_aggregate_raises_on_unknown_feature_type():
    graph = {
        "nodes": [
            {"node_id": "mystery", "feature_type": "WHATSIT"},
            {"node_id": "target", "feature_type": "logit", "is_target_logit": True},
        ],
        "links": [{"source": "mystery", "target": "target", "weight": 1.0}],
    }
    with pytest.raises(ValueError, match="unknown feature_type"):
        aggregate_edge_attributions(graph, target_node_id="target")


def test_aggregate_skips_external_source_ids():
    """E_*_* style external token-embedding refs that aren't in the node list are skipped."""
    graph = {
        "nodes": [
            {"node_id": "target", "feature_type": "logit", "is_target_logit": True},
        ],
        "links": [
            {"source": "E_2_0", "target": "target", "weight": 1.0},  # external — no node
        ],
    }
    sums = aggregate_edge_attributions(graph, target_node_id="target")
    assert sums["n_edges_to_target"] == 0  # skipped silently


def test_find_measurement_target_uses_is_target_logit():
    """When multiple logit nodes exist, return the one with is_target_logit=True."""
    graph = {
        "nodes": [
            {"node_id": "n1", "feature_type": "cross layer transcoder"},
            {"node_id": "target", "feature_type": "logit", "is_target_logit": True},
        ],
        "links": [],
    }
    assert find_measurement_target_node_id(graph) == "target"


def test_find_measurement_target_no_logit_raises():
    graph = {
        "nodes": [{"node_id": "n1", "feature_type": "cross layer transcoder"}],
        "links": [],
    }
    with pytest.raises(ValueError, match="no logit node"):
        find_measurement_target_node_id(graph)


def test_find_measurement_target_multiple_target_logits_raises():
    graph = {
        "nodes": [
            {"node_id": "t1", "feature_type": "logit", "is_target_logit": True},
            {"node_id": "t2", "feature_type": "logit", "is_target_logit": True},
        ],
        "links": [],
    }
    with pytest.raises(ValueError, match="multiple target logit"):
        find_measurement_target_node_id(graph)


# ===== extract_edge_records_to_target =====

def test_extract_edge_records_returns_signed_per_source_sorted_desc():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from graph_loader import extract_edge_records_to_target  # noqa: E402
    graph = {
        "nodes": [
            {"node_id": "f1", "feature_type": "cross layer transcoder", "layer": "13", "feature": 427},
            {"node_id": "f2", "feature_type": "cross layer transcoder", "layer": "11", "feature": 99},
            {"node_id": "e1", "feature_type": "embedding", "layer": "0", "feature": 12345},
            {"node_id": "target", "feature_type": "logit", "is_target_logit": True},
        ],
        "links": [
            {"source": "f1", "target": "target", "weight": +1.5},
            {"source": "f2", "target": "target", "weight": -0.5},
            {"source": "e1", "target": "target", "weight": +0.2},
            {"source": "f1", "target": "f2", "weight": 99.0},
        ],
    }
    records = extract_edge_records_to_target(graph, target_node_id="target",
                                             filter_category="feature")
    assert len(records) == 2
    # Sorted descending by signed_attribution
    assert records[0]["source_id"] == "f1"
    assert records[0]["signed_attribution"] == pytest.approx(1.5)
    assert records[0]["category"] == "feature"
    assert records[0]["layer"] == 13
    assert records[0]["feature"] == 427
    assert records[1]["source_id"] == "f2"
    assert records[1]["signed_attribution"] == pytest.approx(-0.5)


def test_extract_edge_records_no_filter_returns_all_categories():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from graph_loader import extract_edge_records_to_target  # noqa: E402
    graph = {
        "nodes": [
            {"node_id": "f1", "feature_type": "cross layer transcoder", "layer": "1", "feature": 0},
            {"node_id": "e1", "feature_type": "embedding", "layer": "0", "feature": 0},
            {"node_id": "err1", "feature_type": "mlp reconstruction error", "layer": "15", "feature": 0},
            {"node_id": "target", "feature_type": "logit", "is_target_logit": True},
        ],
        "links": [
            {"source": "f1", "target": "target", "weight": 1.0},
            {"source": "e1", "target": "target", "weight": 2.0},
            {"source": "err1", "target": "target", "weight": 3.0},
        ],
    }
    records = extract_edge_records_to_target(graph, target_node_id="target", filter_category=None)
    assert len(records) == 3
    categories = {r["category"] for r in records}
    assert categories == {"feature", "embedding", "error_node"}


# ===== extract_feature_profile =====

def test_extract_feature_profile_returns_only_features():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from graph_loader import extract_feature_profile  # noqa: E402
    graph = {
        "nodes": [
            {"node_id": "f1", "feature_type": "cross layer transcoder", "layer": "13", "feature": 427},
            {"node_id": "e1", "feature_type": "embedding", "layer": "0", "feature": 1},
            {"node_id": "err1", "feature_type": "mlp reconstruction error", "layer": "15", "feature": 0},
            {"node_id": "target", "feature_type": "logit", "is_target_logit": True},
        ],
        "links": [
            {"source": "f1", "target": "target", "weight": +1.5},
            {"source": "e1", "target": "target", "weight": +0.2},
            {"source": "err1", "target": "target", "weight": -0.1},
        ],
    }
    profile = extract_feature_profile(graph, "target")
    # Test fixtures use simple node_ids ("f1") that don't match the parse pattern,
    # so the loader falls back to schema fields layer=13, feature=427.
    assert profile == {(13, 427): 1.5}


def test_extract_feature_profile_parses_real_node_id_format():
    """In production packed graphs, node_id format is '<layer>_<feature_idx>_<ctx_idx>'.

    The middle component is the Gemma Scope feature_idx (Stage-04-compatible),
    NOT the `feature` field (which is a Neuronpedia API ID).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from graph_loader import extract_feature_profile  # noqa: E402
    graph = {
        "nodes": [
            # Real-format node: feature_idx=144 from node_id; `feature` field is the
            # Neuronpedia API ID (10584) for the same feature. Loader must prefer node_id.
            {"node_id": "0_144_14", "feature_type": "cross layer transcoder",
             "layer": "0", "feature": 10584},
            {"node_id": "35_262208_18", "feature_type": "logit", "is_target_logit": True},
        ],
        "links": [
            {"source": "0_144_14", "target": "35_262208_18", "weight": +1.0},
        ],
    }
    profile = extract_feature_profile(graph, "35_262208_18")
    # Must use feature_idx=144 (from node_id), NOT feature=10584 (Neuronpedia ID)
    assert profile == {(0, 144): 1.0}, f"got {profile}, expected (0, 144): 1.0"
    assert (0, 10584) not in profile, "should NOT use the `feature` field (Neuronpedia ID)"


def test_extract_edge_records_parses_real_node_id_format():
    """Same Stage-04-compatible feature_idx parsing for edge records."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from graph_loader import extract_edge_records_to_target  # noqa: E402
    graph = {
        "nodes": [
            {"node_id": "13_427_18", "feature_type": "cross layer transcoder",
             "layer": "13", "feature": 99999},  # Neuronpedia ID different from node_id
            {"node_id": "35_262208_18", "feature_type": "logit", "is_target_logit": True},
        ],
        "links": [
            {"source": "13_427_18", "target": "35_262208_18", "weight": +1.5},
        ],
    }
    records = extract_edge_records_to_target(graph, "35_262208_18", filter_category="feature")
    assert len(records) == 1
    assert records[0]["layer"] == 13
    assert records[0]["feature"] == 427, f"expected 427 from node_id, got {records[0]['feature']}"
