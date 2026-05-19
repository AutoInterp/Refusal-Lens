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
