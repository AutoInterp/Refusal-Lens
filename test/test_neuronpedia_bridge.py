"""Tests for the Neuronpedia bridge additions to supernode_analyzer.py.

Covers Feature namedtuple, parse_neuronpedia_url, get_transcoder_for_feature,
supernodes_to_features, and the hardcoded feature constants.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from refusal_lens.supernode_analyzer import (
    Feature,
    NeuronPattern,
    SupernodeData,
    NEURONPEDIA_GRAPH_URL,
    REFUSAL_FEATURES,
    ROLEPLAY_FEATURES,
    get_transcoder_for_feature,
    parse_neuronpedia_url,
    supernodes_to_features,
)


# ---------------------------------------------------------------------------
# Feature NamedTuple
# ---------------------------------------------------------------------------

class TestFeature:
    def test_creation(self):
        f = Feature(layer=18, feature_idx=84158, token_pos=59)
        assert f.layer == 18
        assert f.feature_idx == 84158
        assert f.token_pos == 59

    def test_is_tuple(self):
        f = Feature(layer=0, feature_idx=1, token_pos=2)
        assert isinstance(f, tuple)
        assert len(f) == 3

    def test_unpacking(self):
        layer, idx, pos = Feature(layer=5, feature_idx=100, token_pos=10)
        assert (layer, idx, pos) == (5, 100, 10)

    def test_equality(self):
        a = Feature(18, 84158, 59)
        b = Feature(18, 84158, 59)
        assert a == b

    def test_inequality(self):
        a = Feature(18, 84158, 59)
        b = Feature(18, 84158, 60)
        assert a != b

    def test_hashable(self):
        """Features should work as dict keys / set members."""
        s = {Feature(1, 2, 3), Feature(1, 2, 3), Feature(4, 5, 6)}
        assert len(s) == 2


# ---------------------------------------------------------------------------
# Hardcoded feature constants
# ---------------------------------------------------------------------------

class TestFeatureConstants:
    def test_roleplay_count(self):
        assert len(ROLEPLAY_FEATURES) == 5

    def test_refusal_count(self):
        assert len(REFUSAL_FEATURES) == 4

    def test_all_are_feature_instances(self):
        for f in ROLEPLAY_FEATURES + REFUSAL_FEATURES:
            assert isinstance(f, Feature)

    def test_roleplay_layers(self):
        layers = {f.layer for f in ROLEPLAY_FEATURES}
        assert layers == {16, 17, 18, 19}

    def test_refusal_layers(self):
        layers = {f.layer for f in REFUSAL_FEATURES}
        assert layers == {0, 16, 17, 18}

    def test_no_overlap(self):
        rp_set = set(ROLEPLAY_FEATURES)
        ref_set = set(REFUSAL_FEATURES)
        assert rp_set.isdisjoint(ref_set)

    def test_specific_roleplay_features(self):
        """Verify exact values match notebook's Cell 13."""
        assert Feature(18, 84158, 59) in ROLEPLAY_FEATURES
        assert Feature(16, 37041, 59) in ROLEPLAY_FEATURES
        assert Feature(19, 4137, 59) in ROLEPLAY_FEATURES

    def test_specific_refusal_features(self):
        assert Feature(16, 9947, 59) in REFUSAL_FEATURES
        assert Feature(0, 3393, 58) in REFUSAL_FEATURES
        assert Feature(18, 10166, 59) in REFUSAL_FEATURES


# ---------------------------------------------------------------------------
# parse_neuronpedia_url
# ---------------------------------------------------------------------------

class TestParseNeuronpediaUrl:
    def test_parses_graph_url(self):
        result = parse_neuronpedia_url(NEURONPEDIA_GRAPH_URL)
        assert "Role Playing Features" in result
        assert "Refuse" in result

    def test_roleplay_group_count(self):
        result = parse_neuronpedia_url(NEURONPEDIA_GRAPH_URL)
        assert len(result["Role Playing Features"]) == 5

    def test_refuse_group_count(self):
        result = parse_neuronpedia_url(NEURONPEDIA_GRAPH_URL)
        assert len(result["Refuse"]) == 4

    def test_parsed_matches_hardcoded_roleplay(self):
        result = parse_neuronpedia_url(NEURONPEDIA_GRAPH_URL)
        for i, f in enumerate(result["Role Playing Features"]):
            assert f == ROLEPLAY_FEATURES[i], f"RP mismatch at index {i}"

    def test_parsed_matches_hardcoded_refusal(self):
        result = parse_neuronpedia_url(NEURONPEDIA_GRAPH_URL)
        for i, f in enumerate(result["Refuse"]):
            assert f == REFUSAL_FEATURES[i], f"Refusal mismatch at index {i}"

    def test_all_parsed_are_features(self):
        result = parse_neuronpedia_url(NEURONPEDIA_GRAPH_URL)
        for group in result.values():
            for f in group:
                assert isinstance(f, Feature)

    def test_missing_supernodes_param_raises(self):
        with pytest.raises(ValueError, match="no 'supernodes' query parameter"):
            parse_neuronpedia_url("https://example.com/graph?foo=bar")

    def test_minimal_url(self):
        """Construct a minimal URL with one supernode and one feature."""
        import json
        from urllib.parse import quote

        data = json.dumps([["TestGroup", "5_100_10"]])
        url = f"https://example.com/graph?supernodes={quote(data)}"
        result = parse_neuronpedia_url(url)
        assert "TestGroup" in result
        assert result["TestGroup"] == [Feature(5, 100, 10)]

    def test_multiple_groups(self):
        import json
        from urllib.parse import quote

        data = json.dumps([
            ["GroupA", "1_2_3", "4_5_6"],
            ["GroupB", "7_8_9"],
        ])
        url = f"https://example.com/graph?supernodes={quote(data)}"
        result = parse_neuronpedia_url(url)
        assert len(result) == 2
        assert len(result["GroupA"]) == 2
        assert len(result["GroupB"]) == 1


# ---------------------------------------------------------------------------
# get_transcoder_for_feature
# ---------------------------------------------------------------------------

class TestGetTranscoderForFeature:
    """Use SimpleNamespace as a fake transcoder with a d_sae attribute."""

    def test_wide_feature_uses_262k(self):
        tc_wide = SimpleNamespace(d_sae=262144)
        tc_16k = SimpleNamespace(d_sae=16384)
        feat = Feature(layer=18, feature_idx=84158, token_pos=59)
        tc, label = get_transcoder_for_feature(
            feat,
            transcoders_16k={18: tc_16k},
            transcoders_wide={18: tc_wide},
        )
        assert tc is tc_wide
        assert label == "262k"

    def test_narrow_feature_uses_16k(self):
        tc_16k = SimpleNamespace(d_sae=16384)
        feat = Feature(layer=16, feature_idx=9947, token_pos=59)
        tc, label = get_transcoder_for_feature(
            feat,
            transcoders_16k={16: tc_16k},
            transcoders_wide={},
        )
        assert tc is tc_16k
        assert label == "16k"

    def test_missing_layer_returns_none(self):
        feat = Feature(layer=99, feature_idx=100, token_pos=0)
        tc, label = get_transcoder_for_feature(
            feat,
            transcoders_16k={},
            transcoders_wide={},
        )
        assert tc is None
        assert label is None

    def test_wide_feature_no_wide_tc_falls_back_to_16k(self):
        """If feature_idx >= 16384 but no wide transcoder, falls back to 16k
        only if the 16k transcoder's d_sae is large enough."""
        tc_16k = SimpleNamespace(d_sae=100000)  # large enough
        feat = Feature(layer=18, feature_idx=20000, token_pos=59)
        tc, label = get_transcoder_for_feature(
            feat,
            transcoders_16k={18: tc_16k},
            transcoders_wide={},
        )
        assert tc is tc_16k
        assert label == "16k"

    def test_feature_idx_exceeds_d_sae_returns_none(self):
        """If feature_idx >= d_sae for both transcoders, return None."""
        tc_16k = SimpleNamespace(d_sae=100)
        tc_wide = SimpleNamespace(d_sae=100)
        feat = Feature(layer=18, feature_idx=84158, token_pos=59)
        tc, label = get_transcoder_for_feature(
            feat,
            transcoders_16k={18: tc_16k},
            transcoders_wide={18: tc_wide},
        )
        assert tc is None
        assert label is None

    def test_custom_wide_threshold(self):
        tc_wide = SimpleNamespace(d_sae=50000)
        feat = Feature(layer=5, feature_idx=500, token_pos=0)
        tc, label = get_transcoder_for_feature(
            feat,
            transcoders_16k={},
            transcoders_wide={5: tc_wide},
            wide_threshold=100,  # low threshold
        )
        assert tc is tc_wide
        assert label == "262k"

    def test_boundary_at_threshold(self):
        """feature_idx == wide_threshold should use wide."""
        tc_wide = SimpleNamespace(d_sae=262144)
        tc_16k = SimpleNamespace(d_sae=16384)
        feat = Feature(layer=18, feature_idx=16384, token_pos=59)
        tc, label = get_transcoder_for_feature(
            feat,
            transcoders_16k={18: tc_16k},
            transcoders_wide={18: tc_wide},
        )
        assert tc is tc_wide
        assert label == "262k"

    def test_below_threshold_uses_16k(self):
        tc_wide = SimpleNamespace(d_sae=262144)
        tc_16k = SimpleNamespace(d_sae=16384)
        feat = Feature(layer=18, feature_idx=16383, token_pos=59)
        tc, label = get_transcoder_for_feature(
            feat,
            transcoders_16k={18: tc_16k},
            transcoders_wide={18: tc_wide},
        )
        assert tc is tc_16k
        assert label == "16k"


# ---------------------------------------------------------------------------
# supernodes_to_features
# ---------------------------------------------------------------------------

class TestSupernodesToFeatures:
    def test_basic_conversion(self):
        supernodes = {
            1: SupernodeData(
                supernode_id=1,
                neurons=[
                    NeuronPattern(
                        neuron_id=100,
                        activation_strength=0.9,
                        features=[],
                        metadata={"layer": 18, "token_pos": 59},
                    ),
                ],
            ),
        }
        features = supernodes_to_features(supernodes)
        assert len(features) == 1
        assert features[0] == Feature(layer=18, feature_idx=100, token_pos=59)

    def test_defaults_when_metadata_missing(self):
        """layer defaults to supernode_id, token_pos defaults to 0."""
        supernodes = {
            7: SupernodeData(
                supernode_id=7,
                neurons=[
                    NeuronPattern(
                        neuron_id=42,
                        activation_strength=0.5,
                        features=[],
                        metadata={},
                    ),
                ],
            ),
        }
        features = supernodes_to_features(supernodes)
        assert features[0].layer == 7  # falls back to supernode_id
        assert features[0].token_pos == 0

    def test_multiple_supernodes(self):
        supernodes = {
            1: SupernodeData(
                supernode_id=1,
                neurons=[
                    NeuronPattern(neuron_id=10, activation_strength=0.9, features=[], metadata={"layer": 1}),
                    NeuronPattern(neuron_id=20, activation_strength=0.8, features=[], metadata={"layer": 1}),
                ],
            ),
            2: SupernodeData(
                supernode_id=2,
                neurons=[
                    NeuronPattern(neuron_id=30, activation_strength=0.7, features=[], metadata={"layer": 2}),
                ],
            ),
        }
        features = supernodes_to_features(supernodes)
        assert len(features) == 3

    def test_empty_supernodes(self):
        assert supernodes_to_features({}) == []

    def test_supernode_with_no_neurons(self):
        supernodes = {
            1: SupernodeData(supernode_id=1, neurons=[]),
        }
        assert supernodes_to_features(supernodes) == []


# ---------------------------------------------------------------------------
# NEURONPEDIA_GRAPH_URL constant
# ---------------------------------------------------------------------------

class TestNeuronpediaGraphUrl:
    def test_is_string(self):
        assert isinstance(NEURONPEDIA_GRAPH_URL, str)

    def test_contains_model_name(self):
        assert "gemma-3-4b-it" in NEURONPEDIA_GRAPH_URL

    def test_contains_supernodes_param(self):
        assert "supernodes=" in NEURONPEDIA_GRAPH_URL

    def test_parseable(self):
        """The URL should be parseable by our function."""
        result = parse_neuronpedia_url(NEURONPEDIA_GRAPH_URL)
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------

class TestPackageExports:
    def test_feature_exported(self):
        import refusal_lens
        assert hasattr(refusal_lens, "Feature")

    def test_constants_exported(self):
        import refusal_lens
        assert hasattr(refusal_lens, "ROLEPLAY_FEATURES")
        assert hasattr(refusal_lens, "REFUSAL_FEATURES")
        assert hasattr(refusal_lens, "NEURONPEDIA_GRAPH_URL")

    def test_functions_exported(self):
        import refusal_lens
        assert hasattr(refusal_lens, "parse_neuronpedia_url")
        assert hasattr(refusal_lens, "get_transcoder_for_feature")
        assert hasattr(refusal_lens, "supernodes_to_features")

    def test_all_in_all(self):
        import refusal_lens
        for name in [
            "Feature", "ROLEPLAY_FEATURES", "REFUSAL_FEATURES",
            "NEURONPEDIA_GRAPH_URL", "parse_neuronpedia_url",
            "get_transcoder_for_feature", "supernodes_to_features",
        ]:
            assert name in refusal_lens.__all__, f"{name} not in __all__"
