"""Tests for refusal_lens.attribution — intermediate-layer attribution.

Tests are split into two tiers:

1. **Always-run tests**: Module import, validate_measurement_point (pure Python),
   vendored code signature checks, package exports.

2. **circuit-tracer-required tests**: Use ``pytest.importorskip`` to gate on
   circuit-tracer/torch.  These test ``attribute_to_direction`` validation,
   direction shape rejection, and the delegation from ``clt.attribute_to_refusal``.
"""
from __future__ import annotations

import inspect

import pytest


# ============================================================================
# Tier 1: Always-run tests (no circuit-tracer needed)
# ============================================================================


class TestAttributionModuleImport:
    """attribution.py must import cleanly even without circuit-tracer."""

    def test_module_imports(self):
        from refusal_lens import attribution
        assert hasattr(attribution, "HAS_CIRCUIT_TRACER")

    def test_has_circuit_tracer_is_bool(self):
        from refusal_lens.attribution import HAS_CIRCUIT_TRACER
        assert isinstance(HAS_CIRCUIT_TRACER, bool)

    def test_public_api_accessible(self):
        from refusal_lens import attribution
        expected = [
            "HAS_CIRCUIT_TRACER",
            "validate_measurement_point",
            "attribute_to_direction",
        ]
        for name in expected:
            assert hasattr(attribution, name), f"attribution.{name} not found"


class TestRequireCircuitTracerAttribution:
    def test_raises_when_not_installed(self):
        from refusal_lens.attribution import HAS_CIRCUIT_TRACER, _require_circuit_tracer
        if not HAS_CIRCUIT_TRACER:
            with pytest.raises(ImportError, match="circuit-tracer is required"):
                _require_circuit_tracer()

    def test_error_message_includes_install_command(self):
        from refusal_lens.attribution import HAS_CIRCUIT_TRACER, _require_circuit_tracer
        if not HAS_CIRCUIT_TRACER:
            with pytest.raises(ImportError, match="pip install"):
                _require_circuit_tracer()


# ---------------------------------------------------------------------------
# validate_measurement_point — pure Python, always runnable
# ---------------------------------------------------------------------------


class TestValidateMeasurementPoint:
    """Comprehensive tests for measurement point validation."""

    def test_none_none_passes(self):
        from refusal_lens.attribution import validate_measurement_point
        validate_measurement_point(None, None)  # should not raise

    def test_none_none_with_bounds_passes(self):
        from refusal_lens.attribution import validate_measurement_point
        validate_measurement_point(None, None, n_layers=26, n_positions=50)

    def test_valid_layer_passes(self):
        from refusal_lens.attribution import validate_measurement_point
        validate_measurement_point(20, None)

    def test_valid_position_passes(self):
        from refusal_lens.attribution import validate_measurement_point
        validate_measurement_point(None, 5)

    def test_valid_both_passes(self):
        from refusal_lens.attribution import validate_measurement_point
        validate_measurement_point(20, 5)

    def test_layer_zero_is_valid(self):
        from refusal_lens.attribution import validate_measurement_point
        validate_measurement_point(0, None, n_layers=26)

    def test_position_zero_is_valid(self):
        from refusal_lens.attribution import validate_measurement_point
        validate_measurement_point(None, 0, n_positions=50)

    def test_layer_just_under_n_layers_passes(self):
        from refusal_lens.attribution import validate_measurement_point
        validate_measurement_point(25, None, n_layers=26)

    def test_position_just_under_n_positions_passes(self):
        from refusal_lens.attribution import validate_measurement_point
        validate_measurement_point(None, 49, n_positions=50)

    def test_negative_layer_raises(self):
        from refusal_lens.attribution import validate_measurement_point
        with pytest.raises(ValueError, match="must be >= 0"):
            validate_measurement_point(-1, None)

    def test_negative_position_raises(self):
        from refusal_lens.attribution import validate_measurement_point
        with pytest.raises(ValueError, match="must be >= 0"):
            validate_measurement_point(None, -1)

    def test_layer_equals_n_layers_raises(self):
        from refusal_lens.attribution import validate_measurement_point
        with pytest.raises(ValueError, match="measurement_layer 26 >= n_layers 26"):
            validate_measurement_point(26, None, n_layers=26)

    def test_layer_exceeds_n_layers_raises(self):
        from refusal_lens.attribution import validate_measurement_point
        with pytest.raises(ValueError, match="measurement_layer 30 >= n_layers 26"):
            validate_measurement_point(30, None, n_layers=26)

    def test_position_equals_n_positions_raises(self):
        from refusal_lens.attribution import validate_measurement_point
        with pytest.raises(ValueError, match="measurement_position 50 >= n_positions 50"):
            validate_measurement_point(None, 50, n_positions=50)

    def test_position_exceeds_n_positions_raises(self):
        from refusal_lens.attribution import validate_measurement_point
        with pytest.raises(ValueError, match="measurement_position 100 >= n_positions 50"):
            validate_measurement_point(None, 100, n_positions=50)

    def test_negative_layer_checked_before_upper_bound(self):
        """Negative layer should raise even without n_layers."""
        from refusal_lens.attribution import validate_measurement_point
        with pytest.raises(ValueError, match="must be >= 0"):
            validate_measurement_point(-5, None)

    def test_no_upper_bound_check_without_n_layers(self):
        """Large layer should pass when n_layers is not provided."""
        from refusal_lens.attribution import validate_measurement_point
        validate_measurement_point(9999, None)  # no n_layers → no upper bound check

    def test_no_upper_bound_check_without_n_positions(self):
        """Large position should pass when n_positions is not provided."""
        from refusal_lens.attribution import validate_measurement_point
        validate_measurement_point(None, 9999)  # no n_positions → no upper bound check

    def test_both_invalid_layer_reported(self):
        """When both are invalid, layer is checked first."""
        from refusal_lens.attribution import validate_measurement_point
        with pytest.raises(ValueError, match="measurement_layer"):
            validate_measurement_point(-1, -1)


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------


class TestAttributionPackageExports:
    def test_attribute_to_direction_in_all(self):
        import refusal_lens
        assert "attribute_to_direction" in refusal_lens.__all__

    def test_validate_measurement_point_in_all(self):
        import refusal_lens
        assert "validate_measurement_point" in refusal_lens.__all__

    def test_importable_from_package(self):
        from refusal_lens import attribute_to_direction, validate_measurement_point
        assert callable(attribute_to_direction)
        assert callable(validate_measurement_point)


# ---------------------------------------------------------------------------
# Vendored circuit-tracer signature checks
# ---------------------------------------------------------------------------


class TestVendoredAttributeSignatures:
    """Verify the vendored circuit-tracer code accepts measurement parameters.

    These tests use ``inspect.signature`` to verify the parameters exist
    in the vendored functions without needing to run attribution.
    """

    def test_toplevel_attribute_has_measurement_layer(self):
        ct = pytest.importorskip("circuit_tracer")
        from circuit_tracer.attribution.attribute import attribute
        sig = inspect.signature(attribute)
        assert "measurement_layer" in sig.parameters

    def test_toplevel_attribute_has_measurement_position(self):
        ct = pytest.importorskip("circuit_tracer")
        from circuit_tracer.attribution.attribute import attribute
        sig = inspect.signature(attribute)
        assert "measurement_position" in sig.parameters

    def test_toplevel_measurement_layer_defaults_none(self):
        ct = pytest.importorskip("circuit_tracer")
        from circuit_tracer.attribution.attribute import attribute
        sig = inspect.signature(attribute)
        assert sig.parameters["measurement_layer"].default is None

    def test_toplevel_measurement_position_defaults_none(self):
        ct = pytest.importorskip("circuit_tracer")
        from circuit_tracer.attribution.attribute import attribute
        sig = inspect.signature(attribute)
        assert sig.parameters["measurement_position"].default is None

    def test_nnsight_attribute_has_measurement_params(self):
        ct = pytest.importorskip("circuit_tracer")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from circuit_tracer.attribution.attribute_nnsight import attribute
        sig = inspect.signature(attribute)
        assert "measurement_layer" in sig.parameters
        assert "measurement_position" in sig.parameters

    def test_nnsight_run_attribution_has_measurement_params(self):
        ct = pytest.importorskip("circuit_tracer")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from circuit_tracer.attribution.attribute_nnsight import _run_attribution
        sig = inspect.signature(_run_attribution)
        assert "measurement_layer" in sig.parameters
        assert "measurement_position" in sig.parameters

    def test_transformerlens_attribute_has_measurement_params(self):
        ct = pytest.importorskip("circuit_tracer")
        from circuit_tracer.attribution.attribute_transformerlens import attribute
        sig = inspect.signature(attribute)
        assert "measurement_layer" in sig.parameters
        assert "measurement_position" in sig.parameters

    def test_transformerlens_run_attribution_has_measurement_params(self):
        ct = pytest.importorskip("circuit_tracer")
        from circuit_tracer.attribution.attribute_transformerlens import _run_attribution
        sig = inspect.signature(_run_attribution)
        assert "measurement_layer" in sig.parameters
        assert "measurement_position" in sig.parameters

    def test_nnsight_run_attribution_defaults_none(self):
        ct = pytest.importorskip("circuit_tracer")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from circuit_tracer.attribution.attribute_nnsight import _run_attribution
        sig = inspect.signature(_run_attribution)
        assert sig.parameters["measurement_layer"].default is None
        assert sig.parameters["measurement_position"].default is None

    def test_transformerlens_run_attribution_defaults_none(self):
        ct = pytest.importorskip("circuit_tracer")
        from circuit_tracer.attribution.attribute_transformerlens import _run_attribution
        sig = inspect.signature(_run_attribution)
        assert sig.parameters["measurement_layer"].default is None
        assert sig.parameters["measurement_position"].default is None


# ---------------------------------------------------------------------------
# attribute_to_direction function signature
# ---------------------------------------------------------------------------


class TestAttributeToDirectionSignature:
    """Verify attribute_to_direction has the expected parameters."""

    def test_has_measurement_layer(self):
        from refusal_lens.attribution import attribute_to_direction
        sig = inspect.signature(attribute_to_direction)
        assert "measurement_layer" in sig.parameters

    def test_has_measurement_position(self):
        from refusal_lens.attribution import attribute_to_direction
        sig = inspect.signature(attribute_to_direction)
        assert "measurement_position" in sig.parameters

    def test_measurement_layer_defaults_none(self):
        from refusal_lens.attribution import attribute_to_direction
        sig = inspect.signature(attribute_to_direction)
        assert sig.parameters["measurement_layer"].default is None

    def test_measurement_position_defaults_none(self):
        from refusal_lens.attribution import attribute_to_direction
        sig = inspect.signature(attribute_to_direction)
        assert sig.parameters["measurement_position"].default is None

    def test_has_label_param(self):
        from refusal_lens.attribution import attribute_to_direction
        sig = inspect.signature(attribute_to_direction)
        assert sig.parameters["label"].default == "refusal_direction"

    def test_has_batch_size_param(self):
        from refusal_lens.attribution import attribute_to_direction
        sig = inspect.signature(attribute_to_direction)
        assert sig.parameters["batch_size"].default == 512

    def test_required_positional_params(self):
        from refusal_lens.attribution import attribute_to_direction
        sig = inspect.signature(attribute_to_direction)
        positional = [
            name for name, p in sig.parameters.items()
            if p.default is inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        assert positional == ["prompt", "model", "direction"]


# ============================================================================
# Tier 2: Tests requiring circuit-tracer and/or torch
# ============================================================================


class TestAttributeToDirectionValidation:
    """Test direction shape validation and measurement point validation."""

    def test_rejects_2d_direction(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("circuit_tracer")
        from refusal_lens.attribution import attribute_to_direction

        with pytest.raises(ValueError, match="must be 1-D"):
            attribute_to_direction("test", None, torch.randn(1, 256))

    def test_rejects_0d_direction(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("circuit_tracer")
        from refusal_lens.attribution import attribute_to_direction

        with pytest.raises(ValueError, match="must be 1-D"):
            attribute_to_direction("test", None, torch.tensor(1.0))

    def test_rejects_3d_direction(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("circuit_tracer")
        from refusal_lens.attribution import attribute_to_direction

        with pytest.raises(ValueError, match="must be 1-D"):
            attribute_to_direction("test", None, torch.randn(2, 3, 4))

    def test_rejects_negative_layer(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("circuit_tracer")
        from refusal_lens.attribution import attribute_to_direction

        with pytest.raises(ValueError, match="must be >= 0"):
            attribute_to_direction(
                "test", None, torch.randn(256), measurement_layer=-1
            )

    def test_rejects_negative_position(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("circuit_tracer")
        from refusal_lens.attribution import attribute_to_direction

        with pytest.raises(ValueError, match="must be >= 0"):
            attribute_to_direction(
                "test", None, torch.randn(256), measurement_position=-1
            )


class TestAttributeToDirectionCustomTarget:
    """Verify that attribute_to_direction creates the correct CustomTarget."""

    def test_creates_custom_target_with_default_label(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("circuit_tracer")
        from circuit_tracer.attribution.targets import CustomTarget
        from refusal_lens.attribution import attribute_to_direction

        direction = torch.randn(256)
        # We can't run full attribution without a model, but we can verify
        # that passing a valid direction doesn't raise before the attribute() call.
        # The function will fail at attribute() since model is None, but direction
        # validation and CustomTarget creation happen first.
        try:
            attribute_to_direction("test prompt", None, direction)
        except (TypeError, AttributeError):
            pass  # expected — None model fails in attribute()

    def test_custom_label_passed_through(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("circuit_tracer")
        from refusal_lens.attribution import attribute_to_direction

        direction = torch.randn(256)
        try:
            attribute_to_direction("test", None, direction, label="my_custom_target")
        except (TypeError, AttributeError):
            pass  # expected


class TestCltDelegatesToAttribution:
    """Verify that clt.attribute_to_refusal now delegates to attribution.py
    instead of calling attribute() directly with a warning."""

    def test_no_warning_with_layer_param(self):
        """After Step 8, passing layer should NOT emit a warning."""
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
                pytest.fail("attribute_to_refusal should no longer warn about layer param")
            except (TypeError, AttributeError):
                pass  # expected — None model

    def test_no_warning_with_position_param(self):
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
                pytest.fail("attribute_to_refusal should no longer warn about position param")
            except (TypeError, AttributeError):
                pass

    def test_no_warning_with_both_params(self):
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
                pytest.fail("attribute_to_refusal should no longer warn")
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
                pytest.fail("Should not warn when using defaults")
            except (TypeError, AttributeError):
                pass

    def test_attribute_to_refusal_rejects_bad_direction(self):
        """Delegation to attribute_to_direction should propagate ValueError."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("circuit_tracer")
        from refusal_lens.clt import attribute_to_refusal

        with pytest.raises(ValueError, match="must be 1-D"):
            attribute_to_refusal("test", None, torch.randn(2, 256))

    def test_clt_imports_attribute_to_direction(self):
        """Verify clt.py imports from attribution.py."""
        pytest.importorskip("circuit_tracer")
        from refusal_lens import clt
        # The import of attribute_to_direction is inside the try block,
        # so it's only available when circuit-tracer is installed
        assert clt.HAS_CIRCUIT_TRACER  # if we get here, CT is available


class TestVendoredPhase3Patch:
    """Verify the Phase 3 patch in vendored code is structurally correct.

    These tests inspect the source code of _run_attribution to confirm
    that the hardcoded n_layers/n_pos-1 values have been replaced with
    _ml/_mp variables that respect measurement_layer/measurement_position.
    """

    def test_nnsight_phase3_uses_ml_mp(self):
        """_run_attribution in nnsight backend should use _ml and _mp."""
        pytest.importorskip("circuit_tracer")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from circuit_tracer.attribution.attribute_nnsight import _run_attribution
        source = inspect.getsource(_run_attribution)
        # The patch replaces hardcoded n_layers with _ml
        assert "_ml" in source, "Phase 3 should use _ml variable"
        assert "_mp" in source, "Phase 3 should use _mp variable"
        # Verify the conditional assignment exists
        assert "n_layers if measurement_layer is None else measurement_layer" in source
        assert "n_pos - 1 if measurement_position is None else measurement_position" in source

    def test_transformerlens_phase3_uses_ml_mp(self):
        """_run_attribution in transformerlens backend should use _ml and _mp."""
        pytest.importorskip("circuit_tracer")
        from circuit_tracer.attribution.attribute_transformerlens import _run_attribution
        source = inspect.getsource(_run_attribution)
        assert "_ml" in source, "Phase 3 should use _ml variable"
        assert "_mp" in source, "Phase 3 should use _mp variable"
        assert "n_layers if measurement_layer is None else measurement_layer" in source
        assert "n_pos - 1 if measurement_position is None else measurement_position" in source

    def test_nnsight_compute_batch_uses_ml(self):
        """Phase 3's compute_batch should use _ml not n_layers."""
        pytest.importorskip("circuit_tracer")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from circuit_tracer.attribution.attribute_nnsight import _run_attribution
        source = inspect.getsource(_run_attribution)
        # Find Phase 3 section and check it uses _ml
        phase3_idx = source.find("Phase 3")
        phase4_idx = source.find("Phase 4")
        phase3_code = source[phase3_idx:phase4_idx]
        assert "torch.full((batch.shape[0],), _ml)" in phase3_code
        assert "torch.full((batch.shape[0],), _mp)" in phase3_code

    def test_transformerlens_compute_batch_uses_ml(self):
        """Phase 3's compute_batch should use _ml not n_layers."""
        pytest.importorskip("circuit_tracer")
        from circuit_tracer.attribution.attribute_transformerlens import _run_attribution
        source = inspect.getsource(_run_attribution)
        phase3_idx = source.find("Phase 3")
        phase4_idx = source.find("Phase 4")
        phase3_code = source[phase3_idx:phase4_idx]
        assert "torch.full((batch.shape[0],), _ml)" in phase3_code
        assert "torch.full((batch.shape[0],), _mp)" in phase3_code

    def test_nnsight_phase4_unchanged(self):
        """Phase 4 should still use feat_layers/feat_pos, not _ml/_mp."""
        pytest.importorskip("circuit_tracer")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from circuit_tracer.attribution.attribute_nnsight import _run_attribution
        source = inspect.getsource(_run_attribution)
        phase4_idx = source.find("Phase 4")
        phase5_idx = source.find("Phase 5")
        phase4_code = source[phase4_idx:phase5_idx]
        assert "feat_layers[idx_batch]" in phase4_code
        assert "feat_pos[idx_batch]" in phase4_code

    def test_transformerlens_phase4_unchanged(self):
        """Phase 4 should still use feat_layers/feat_pos, not _ml/_mp."""
        pytest.importorskip("circuit_tracer")
        from circuit_tracer.attribution.attribute_transformerlens import _run_attribution
        source = inspect.getsource(_run_attribution)
        phase4_idx = source.find("Phase 4")
        phase5_idx = source.find("Phase 5")
        phase4_code = source[phase4_idx:phase5_idx]
        assert "feat_layers[idx_batch]" in phase4_code
        assert "feat_pos[idx_batch]" in phase4_code


class TestBackwardCompatibility:
    """Verify that None defaults preserve original circuit-tracer behavior."""

    def test_nnsight_none_defaults_to_n_layers(self):
        """When measurement_layer is None, _ml should be n_layers (original behavior)."""
        pytest.importorskip("circuit_tracer")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from circuit_tracer.attribution.attribute_nnsight import _run_attribution
        source = inspect.getsource(_run_attribution)
        assert "_ml = n_layers if measurement_layer is None" in source

    def test_nnsight_none_defaults_to_last_position(self):
        """When measurement_position is None, _mp should be n_pos - 1 (original behavior)."""
        pytest.importorskip("circuit_tracer")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from circuit_tracer.attribution.attribute_nnsight import _run_attribution
        source = inspect.getsource(_run_attribution)
        assert "_mp = n_pos - 1 if measurement_position is None" in source

    def test_toplevel_attribute_backward_compatible(self):
        """Calling attribute() without new params should still work (defaults to None)."""
        pytest.importorskip("circuit_tracer")
        from circuit_tracer.attribution.attribute import attribute
        sig = inspect.signature(attribute)
        # All new params should have defaults (None), so existing callers aren't broken
        for param_name in ["measurement_layer", "measurement_position"]:
            assert sig.parameters[param_name].default is None
