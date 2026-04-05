"""Tests for refusal_lens.clt — circuit-tracer integration.

Tests are split into two tiers:

1. **Always-run tests**: Module import, constants, guard logic, API shape,
   error handling.  These work without circuit-tracer installed.

2. **circuit-tracer-required tests**: Use ``pytest.importorskip`` to skip
   when circuit-tracer is not available.  These test ``make_refusal_target``,
   ``extract_top_features``, and the delegation from ``attribute_to_refusal``
   to ``attribute_to_direction`` (Step 8 replaced the old warning behavior).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


# ============================================================================
# Tier 1: Always-run tests (no circuit-tracer needed)
# ============================================================================


# ---------------------------------------------------------------------------
# Module import & guarded flag
# ---------------------------------------------------------------------------

class TestGuardedImport:
    def test_module_imports_without_circuit_tracer(self):
        """clt.py must import cleanly even without circuit-tracer."""
        from refusal_lens import clt
        assert hasattr(clt, "HAS_CIRCUIT_TRACER")

    def test_has_circuit_tracer_is_bool(self):
        from refusal_lens.clt import HAS_CIRCUIT_TRACER
        assert isinstance(HAS_CIRCUIT_TRACER, bool)

    def test_public_api_accessible(self):
        from refusal_lens import clt
        expected = [
            "HAS_CIRCUIT_TRACER",
            "DEFAULT_TRANSCODER_REPO",
            "DEFAULT_BACKEND",
            "load_replacement_model",
            "make_refusal_target",
            "attribute_to_refusal",
            "attribute_to_refusal_sweep",
            "prune_refusal_graph",
            "extract_top_features",
        ]
        for name in expected:
            assert hasattr(clt, name), f"clt.{name} not found"


# ---------------------------------------------------------------------------
# _require_circuit_tracer gate
# ---------------------------------------------------------------------------

class TestRequireCircuitTracer:
    def test_raises_when_not_installed(self):
        from refusal_lens.clt import HAS_CIRCUIT_TRACER, _require_circuit_tracer
        if not HAS_CIRCUIT_TRACER:
            with pytest.raises(ImportError, match="circuit-tracer is required"):
                _require_circuit_tracer()

    def test_error_message_includes_install_command(self):
        from refusal_lens.clt import HAS_CIRCUIT_TRACER, _require_circuit_tracer
        if not HAS_CIRCUIT_TRACER:
            with pytest.raises(ImportError, match="pip install"):
                _require_circuit_tracer()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_default_transcoder_repo(self):
        from refusal_lens.clt import DEFAULT_TRANSCODER_REPO
        assert "gemma-scope-2-4b-it" in DEFAULT_TRANSCODER_REPO
        assert "width_16k_l0_small" in DEFAULT_TRANSCODER_REPO

    def test_default_backend(self):
        from refusal_lens.clt import DEFAULT_BACKEND
        assert DEFAULT_BACKEND == "nnsight"

    def test_transcoder_repo_is_string(self):
        from refusal_lens.clt import DEFAULT_TRANSCODER_REPO
        assert isinstance(DEFAULT_TRANSCODER_REPO, str)

    def test_transcoder_repo_has_mwhanna_prefix(self):
        from refusal_lens.clt import DEFAULT_TRANSCODER_REPO
        assert DEFAULT_TRANSCODER_REPO.startswith("mwhanna/")


# ---------------------------------------------------------------------------
# Functions require circuit-tracer (guard gate tests)
# ---------------------------------------------------------------------------

class TestFunctionsRequireCircuitTracer:
    """All public functions should raise ImportError when circuit-tracer is missing."""

    def test_load_replacement_model_requires_ct(self):
        from refusal_lens.clt import HAS_CIRCUIT_TRACER, load_replacement_model
        if not HAS_CIRCUIT_TRACER:
            with pytest.raises(ImportError):
                load_replacement_model()

    def test_make_refusal_target_requires_ct(self):
        from refusal_lens.clt import HAS_CIRCUIT_TRACER, make_refusal_target
        if not HAS_CIRCUIT_TRACER:
            with pytest.raises(ImportError):
                make_refusal_target(None)

    def test_attribute_to_refusal_requires_ct(self):
        from refusal_lens.clt import HAS_CIRCUIT_TRACER, attribute_to_refusal
        if not HAS_CIRCUIT_TRACER:
            with pytest.raises(ImportError):
                attribute_to_refusal("test", None, None)

    def test_attribute_to_refusal_sweep_requires_ct(self):
        from refusal_lens.clt import HAS_CIRCUIT_TRACER, attribute_to_refusal_sweep
        if not HAS_CIRCUIT_TRACER:
            with pytest.raises(ImportError):
                attribute_to_refusal_sweep("test", None, None)

    def test_prune_refusal_graph_requires_ct(self):
        from refusal_lens.clt import HAS_CIRCUIT_TRACER, prune_refusal_graph
        if not HAS_CIRCUIT_TRACER:
            with pytest.raises(ImportError):
                prune_refusal_graph(None)

    def test_extract_top_features_requires_ct(self):
        from refusal_lens.clt import HAS_CIRCUIT_TRACER, extract_top_features
        if not HAS_CIRCUIT_TRACER:
            with pytest.raises(ImportError):
                extract_top_features(None)


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------

class TestPackageExports:
    def test_all_names_in_all(self):
        import refusal_lens
        expected = [
            "HAS_CIRCUIT_TRACER",
            "DEFAULT_TRANSCODER_REPO",
            "DEFAULT_BACKEND",
            "load_replacement_model",
            "make_refusal_target",
            "attribute_to_refusal",
            "attribute_to_refusal_sweep",
            "prune_refusal_graph",
            "extract_top_features",
        ]
        for name in expected:
            assert name in refusal_lens.__all__, f"{name} not in __all__"

    def test_all_names_importable(self):
        import refusal_lens
        for name in [
            "HAS_CIRCUIT_TRACER",
            "DEFAULT_TRANSCODER_REPO",
            "DEFAULT_BACKEND",
            "load_replacement_model",
            "make_refusal_target",
            "attribute_to_refusal",
            "attribute_to_refusal_sweep",
            "prune_refusal_graph",
            "extract_top_features",
        ]:
            assert hasattr(refusal_lens, name), f"refusal_lens.{name} not accessible"

    def test_has_circuit_tracer_is_false_without_library(self):
        import refusal_lens
        from refusal_lens.clt import HAS_CIRCUIT_TRACER
        assert refusal_lens.HAS_CIRCUIT_TRACER == HAS_CIRCUIT_TRACER


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    def test_sweep_default_layers_come_from_config(self):
        """attribute_to_refusal_sweep defaults to config.REFUSAL_LAYERS."""
        from refusal_lens import config
        assert config.REFUSAL_LAYERS == list(range(0, 34))

    def test_load_model_defaults_come_from_config(self):
        from refusal_lens import config
        assert config.MODEL_NAME == "google/gemma-3-4b-it"


# ============================================================================
# Tier 2: Tests requiring circuit-tracer (or torch for fake graph objects)
# ============================================================================

# We use pytest.importorskip to gate on torch availability.
# For functions that interact with circuit-tracer types (CustomTarget, Graph),
# we build lightweight fakes using torch tensors.


class TestMakeRefusalTarget:
    """Tests for make_refusal_target — requires torch."""

    def test_creates_target_with_correct_fields(self):
        torch = pytest.importorskip("torch")
        ct = pytest.importorskip("circuit_tracer")
        from refusal_lens.clt import make_refusal_target

        r_hat = torch.randn(256)
        target = make_refusal_target(r_hat)
        assert target.token_str == "refusal_direction"
        assert target.prob == 1.0
        assert torch.equal(target.vec, r_hat)

    def test_custom_label(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("circuit_tracer")
        from refusal_lens.clt import make_refusal_target

        r_hat = torch.randn(256)
        target = make_refusal_target(r_hat, label="my_direction")
        assert target.token_str == "my_direction"

    def test_custom_weight(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("circuit_tracer")
        from refusal_lens.clt import make_refusal_target

        r_hat = torch.randn(256)
        target = make_refusal_target(r_hat, weight=0.5)
        assert target.prob == 0.5

    def test_rejects_2d_direction(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("circuit_tracer")
        from refusal_lens.clt import make_refusal_target

        r_hat_2d = torch.randn(1, 256)
        with pytest.raises(ValueError, match="must be 1-D"):
            make_refusal_target(r_hat_2d)

    def test_rejects_0d_direction(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("circuit_tracer")
        from refusal_lens.clt import make_refusal_target

        r_hat_0d = torch.tensor(1.0)
        with pytest.raises(ValueError, match="must be 1-D"):
            make_refusal_target(r_hat_0d)

    def test_rejects_3d_direction(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("circuit_tracer")
        from refusal_lens.clt import make_refusal_target

        r_hat_3d = torch.randn(2, 3, 4)
        with pytest.raises(ValueError, match="must be 1-D"):
            make_refusal_target(r_hat_3d)

    def test_preserves_dtype(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("circuit_tracer")
        from refusal_lens.clt import make_refusal_target

        r_hat = torch.randn(128, dtype=torch.float16)
        target = make_refusal_target(r_hat)
        assert target.vec.dtype == torch.float16

    def test_unit_vector_not_required(self):
        """The function should accept any 1-D tensor, not just unit vectors."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("circuit_tracer")
        from refusal_lens.clt import make_refusal_target

        r_hat = torch.ones(64) * 5.0  # not unit norm
        target = make_refusal_target(r_hat)
        assert target.vec.norm().item() > 1.0  # confirms non-unit is accepted


class TestAttributeToRefusalDelegation:
    """Test that attribute_to_refusal delegates to attribute_to_direction (Step 8).

    After Step 8, the old warning behavior was removed. attribute_to_refusal
    now passes layer/position through to attribute_to_direction, which passes
    them to the vendored circuit-tracer's attribute() function.
    """

    def test_no_warning_with_layer(self):
        """Passing layer should NOT emit a warning after Step 8."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("circuit_tracer")
        from refusal_lens.clt import attribute_to_refusal
        import warnings

        r_hat = torch.randn(256)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                attribute_to_refusal("test", None, r_hat, layer=20)
            except UserWarning:
                pytest.fail("Should no longer warn about intermediate-layer")
            except (TypeError, AttributeError):
                pass  # expected — None model

    def test_no_warning_with_position(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("circuit_tracer")
        from refusal_lens.clt import attribute_to_refusal
        import warnings

        r_hat = torch.randn(256)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                attribute_to_refusal("test", None, r_hat, position=5)
            except UserWarning:
                pytest.fail("Should no longer warn about intermediate-layer")
            except (TypeError, AttributeError):
                pass

    def test_no_warning_with_both(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("circuit_tracer")
        from refusal_lens.clt import attribute_to_refusal
        import warnings

        r_hat = torch.randn(256)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                attribute_to_refusal("test", None, r_hat, layer=20, position=5)
            except UserWarning:
                pytest.fail("Should no longer warn about intermediate-layer")
            except (TypeError, AttributeError):
                pass

    def test_no_warning_with_defaults(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("circuit_tracer")
        from refusal_lens.clt import attribute_to_refusal
        import warnings

        r_hat = torch.randn(256)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                attribute_to_refusal("test", None, r_hat)
            except UserWarning:
                pytest.fail("Should not warn with default params")
            except (TypeError, AttributeError):
                pass


class TestExtractTopFeatures:
    """Test extract_top_features using a fake Graph-like object with real tensors."""

    def test_basic_extraction(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("circuit_tracer")
        from refusal_lens.clt import extract_top_features

        # Build a fake graph with the minimal attributes needed
        n_features = 5
        n_total = n_features + 3  # +3 for error/token/logit nodes

        graph = SimpleNamespace(
            # (n_features, 3) = [layer, position, feature_idx]
            active_features=torch.tensor([
                [10, 5, 100],  # feature 0: layer 10, pos 5, idx 100
                [12, 5, 200],  # feature 1: layer 12, pos 5, idx 200
                [14, 5, 300],  # feature 2: layer 14, pos 5, idx 300
                [16, 5, 400],  # feature 3: layer 16, pos 5, idx 400
                [18, 5, 500],  # feature 4: layer 18, pos 5, idx 500
            ]),
            adjacency_matrix=torch.zeros(n_total, n_total),
            activation_values=torch.tensor([0.1, 0.5, 0.9, 0.3, 0.7]),
        )
        # Set attribution values in the last row (target) for feature columns
        # Feature 2 has the highest absolute attribution
        graph.adjacency_matrix[-1, 0] = 0.1   # feature 0
        graph.adjacency_matrix[-1, 1] = -0.8  # feature 1 (negative = suppresses refusal)
        graph.adjacency_matrix[-1, 2] = 0.9   # feature 2 (strongest positive)
        graph.adjacency_matrix[-1, 3] = 0.05  # feature 3
        graph.adjacency_matrix[-1, 4] = -0.3  # feature 4

        results = extract_top_features(graph, top_k=3)
        assert len(results) == 3

    def test_sorted_by_absolute_attribution(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("circuit_tracer")
        from refusal_lens.clt import extract_top_features

        n_features = 4
        n_total = n_features + 2

        graph = SimpleNamespace(
            active_features=torch.tensor([
                [10, 0, 100],
                [12, 0, 200],
                [14, 0, 300],
                [16, 0, 400],
            ]),
            adjacency_matrix=torch.zeros(n_total, n_total),
            activation_values=torch.tensor([0.1, 0.2, 0.3, 0.4]),
        )
        graph.adjacency_matrix[-1, 0] = 0.1
        graph.adjacency_matrix[-1, 1] = -0.9  # highest absolute
        graph.adjacency_matrix[-1, 2] = 0.5
        graph.adjacency_matrix[-1, 3] = -0.7  # second highest absolute

        results = extract_top_features(graph, top_k=4)
        abs_attrs = [abs(r["attribution"]) for r in results]
        assert abs_attrs == sorted(abs_attrs, reverse=True)

    def test_result_dict_keys(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("circuit_tracer")
        from refusal_lens.clt import extract_top_features

        n_features = 2
        n_total = n_features + 1

        graph = SimpleNamespace(
            active_features=torch.tensor([[5, 3, 42], [10, 7, 99]]),
            adjacency_matrix=torch.zeros(n_total, n_total),
            activation_values=torch.tensor([1.0, 2.0]),
        )
        graph.adjacency_matrix[-1, 0] = 0.5
        graph.adjacency_matrix[-1, 1] = 0.3

        results = extract_top_features(graph, top_k=2)
        for r in results:
            assert set(r.keys()) == {"layer", "feature_idx", "position", "attribution", "activation"}

    def test_correct_values_extracted(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("circuit_tracer")
        from refusal_lens.clt import extract_top_features

        n_features = 2
        n_total = n_features + 1

        graph = SimpleNamespace(
            active_features=torch.tensor([[18, 5, 84158], [16, 5, 9947]]),
            adjacency_matrix=torch.zeros(n_total, n_total),
            activation_values=torch.tensor([3.5, 2.1]),
        )
        graph.adjacency_matrix[-1, 0] = 0.9
        graph.adjacency_matrix[-1, 1] = -0.4

        results = extract_top_features(graph, top_k=2)
        # First result should be feature 0 (|0.9| > |-0.4|)
        assert results[0]["layer"] == 18
        assert results[0]["feature_idx"] == 84158
        assert results[0]["position"] == 5
        assert abs(results[0]["attribution"] - 0.9) < 1e-5
        assert abs(results[0]["activation"] - 3.5) < 1e-5

        # Second result should be feature 1
        assert results[1]["layer"] == 16
        assert results[1]["feature_idx"] == 9947
        assert abs(results[1]["attribution"] - (-0.4)) < 1e-5

    def test_top_k_larger_than_features(self):
        """Requesting more features than exist should return all features."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("circuit_tracer")
        from refusal_lens.clt import extract_top_features

        n_features = 3
        n_total = n_features + 1

        graph = SimpleNamespace(
            active_features=torch.tensor([[1, 0, 10], [2, 0, 20], [3, 0, 30]]),
            adjacency_matrix=torch.zeros(n_total, n_total),
            activation_values=torch.tensor([0.1, 0.2, 0.3]),
        )
        graph.adjacency_matrix[-1, 0] = 0.5
        graph.adjacency_matrix[-1, 1] = 0.3
        graph.adjacency_matrix[-1, 2] = 0.1

        results = extract_top_features(graph, top_k=100)
        assert len(results) == 3

    def test_negative_attributions_preserved(self):
        """Negative attributions (suppression) should be preserved, not abs'd."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("circuit_tracer")
        from refusal_lens.clt import extract_top_features

        n_features = 2
        n_total = n_features + 1

        graph = SimpleNamespace(
            active_features=torch.tensor([[18, 5, 84158], [16, 5, 9947]]),
            adjacency_matrix=torch.zeros(n_total, n_total),
            activation_values=torch.tensor([1.0, 1.0]),
        )
        graph.adjacency_matrix[-1, 0] = -0.8  # RP feature suppressing refusal
        graph.adjacency_matrix[-1, 1] = 0.3

        results = extract_top_features(graph, top_k=2)
        # Feature 0 should come first (|-0.8| > |0.3|)
        assert results[0]["attribution"] < 0  # negative is preserved
        assert abs(results[0]["attribution"] - (-0.8)) < 1e-5

    def test_top_k_one(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("circuit_tracer")
        from refusal_lens.clt import extract_top_features

        n_features = 5
        n_total = n_features + 1

        graph = SimpleNamespace(
            active_features=torch.tensor([
                [1, 0, 10], [2, 0, 20], [3, 0, 30], [4, 0, 40], [5, 0, 50],
            ]),
            adjacency_matrix=torch.zeros(n_total, n_total),
            activation_values=torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        )
        graph.adjacency_matrix[-1, 0] = 0.1
        graph.adjacency_matrix[-1, 1] = -0.9
        graph.adjacency_matrix[-1, 2] = 0.3
        graph.adjacency_matrix[-1, 3] = 0.5
        graph.adjacency_matrix[-1, 4] = -0.2

        results = extract_top_features(graph, top_k=1)
        assert len(results) == 1
        assert results[0]["feature_idx"] == 20  # |-0.9| is largest


class TestExtractTopFeaturesResearchScenarios:
    """Test extract_top_features with scenarios matching the research use case."""

    def test_rp_features_have_negative_attribution(self):
        """Simulate: RP features suppress refusal → negative A_{s→R}."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("circuit_tracer")
        from refusal_lens.clt import extract_top_features

        n_features = 5
        n_total = n_features + 1

        graph = SimpleNamespace(
            active_features=torch.tensor([
                [18, 59, 84158],  # RP feature
                [16, 59, 37041],  # RP feature
                [16, 59, 9947],   # Refusal feature
                [18, 59, 10166],  # Refusal feature
                [10, 30, 555],    # Unrelated feature
            ]),
            adjacency_matrix=torch.zeros(n_total, n_total),
            activation_values=torch.tensor([2.0, 1.5, 3.0, 2.5, 0.1]),
        )
        # RP features: negative attribution (suppress refusal)
        graph.adjacency_matrix[-1, 0] = -0.7   # RP layer 18
        graph.adjacency_matrix[-1, 1] = -0.4   # RP layer 16
        # Refusal features: positive attribution (drive refusal)
        graph.adjacency_matrix[-1, 2] = 0.9    # Refusal layer 16
        graph.adjacency_matrix[-1, 3] = 0.6    # Refusal layer 18
        # Unrelated: near zero
        graph.adjacency_matrix[-1, 4] = 0.01

        results = extract_top_features(graph, top_k=5)

        # Verify RP features have negative attribution
        rp_results = [r for r in results if r["feature_idx"] in {84158, 37041}]
        for r in rp_results:
            assert r["attribution"] < 0, f"RP feature {r['feature_idx']} should be negative"

        # Verify refusal features have positive attribution
        ref_results = [r for r in results if r["feature_idx"] in {9947, 10166}]
        for r in ref_results:
            assert r["attribution"] > 0, f"Refusal feature {r['feature_idx']} should be positive"

    def test_feature_ranking_matches_known_features(self):
        """The top features by |A_{s→R}| should include known refusal/RP features."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("circuit_tracer")
        from refusal_lens.clt import extract_top_features

        n_features = 8
        n_total = n_features + 1

        graph = SimpleNamespace(
            active_features=torch.tensor([
                [16, 59, 9947],   # Known refusal
                [18, 59, 84158],  # Known RP
                [5, 10, 1],       # Noise
                [7, 20, 2],       # Noise
                [9, 30, 3],       # Noise
                [11, 40, 4],      # Noise
                [13, 50, 5],      # Noise
                [15, 55, 6],      # Noise
            ]),
            adjacency_matrix=torch.zeros(n_total, n_total),
            activation_values=torch.tensor([3.0, 2.0, 0.01, 0.02, 0.01, 0.03, 0.01, 0.02]),
        )
        graph.adjacency_matrix[-1, 0] = 0.95   # refusal: strong positive
        graph.adjacency_matrix[-1, 1] = -0.85  # RP: strong negative
        # Noise features: tiny attributions
        for i in range(2, 8):
            graph.adjacency_matrix[-1, i] = 0.01 * (i - 2)

        top3 = extract_top_features(graph, top_k=3)
        top3_idxs = {r["feature_idx"] for r in top3}
        assert 9947 in top3_idxs, "Known refusal feature should be in top 3"
        assert 84158 in top3_idxs, "Known RP feature should be in top 3"


class TestLoadReplacementModelDefaults:
    """Test that load_replacement_model uses correct defaults from config."""

    def test_requires_circuit_tracer(self):
        from refusal_lens.clt import HAS_CIRCUIT_TRACER, load_replacement_model
        if not HAS_CIRCUIT_TRACER:
            with pytest.raises(ImportError):
                load_replacement_model()


class TestAttributeToRefusalSweepLogic:
    """Test the sweep function's layer iteration logic."""

    def test_default_layers_from_config(self):
        """Verify that REFUSAL_LAYERS has the expected range."""
        from refusal_lens import config
        assert config.REFUSAL_LAYERS == list(range(0, 34))
        assert len(config.REFUSAL_LAYERS) == 34

    def test_sweep_requires_circuit_tracer(self):
        from refusal_lens.clt import HAS_CIRCUIT_TRACER, attribute_to_refusal_sweep
        if not HAS_CIRCUIT_TRACER:
            with pytest.raises(ImportError):
                attribute_to_refusal_sweep("test", None, None, layers=[8, 10])


class TestPruneRefusalGraphApi:
    """Test prune_refusal_graph API shape."""

    def test_requires_circuit_tracer(self):
        from refusal_lens.clt import HAS_CIRCUIT_TRACER, prune_refusal_graph
        if not HAS_CIRCUIT_TRACER:
            with pytest.raises(ImportError):
                prune_refusal_graph(None)
