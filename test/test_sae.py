"""
Tests for the sae module.

Two test tiers:
  1. Mock-based tests (always run) -- validate logic without real packages.
  2. Real-package tests (skip if torch missing) -- validate with actual tensors.
  3. Real HuggingFace download tests (skip if no torch/network) -- validate loading.

Note: We use create=True on patch() calls for torch-dependent attributes
because these only exist on the module when torch is installed.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Detect if real torch is available
try:
    import torch
    import torch.nn as nn

    _HAS_REAL_TORCH = True
except ImportError:
    _HAS_REAL_TORCH = False

requires_torch = pytest.mark.skipif(
    not _HAS_REAL_TORCH, reason="torch not installed"
)

# Module path shorthand
_SAE = "refusal_lens.sae"


# ---------------------------------------------------------------------------
# Patch helpers (same pattern as test_model_loader.py)
# ---------------------------------------------------------------------------


def _sae_patches(
    mock_torch,
    mock_nn=None,
    mock_F=None,
    mock_hf_download=None,
    mock_load_file=None,
):
    """Build patch list for sae module's torch-dependent attributes."""
    patches = [
        patch(f"{_SAE}.HAS_TORCH", True),
        patch(f"{_SAE}.torch", mock_torch, create=True),
    ]
    if mock_nn is not None:
        patches.append(patch(f"{_SAE}.nn", mock_nn, create=True))
    if mock_F is not None:
        patches.append(patch(f"{_SAE}.F", mock_F, create=True))
    if mock_hf_download is not None:
        patches.append(
            patch(f"{_SAE}.hf_hub_download", mock_hf_download, create=True)
        )
    if mock_load_file is not None:
        patches.append(
            patch(f"{_SAE}.load_file", mock_load_file, create=True)
        )
    return patches


class _PatchContext:
    """Context manager that applies a list of patches."""

    def __init__(self, patches):
        self._patches = patches
        self._active = []

    def __enter__(self):
        for p in self._patches:
            self._active.append(p.__enter__())
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.__exit__(*exc)
        self._active.clear()
        return False


def apply_patches(*patches):
    """Combine multiple patch context managers into one."""
    return _PatchContext(list(patches))


# ---------------------------------------------------------------------------
# Test: Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Test that module-level constants have expected values."""

    def test_default_scope_repo(self):
        from refusal_lens.sae import DEFAULT_SCOPE_REPO

        assert DEFAULT_SCOPE_REPO == "google/gemma-scope-2-4b-it"

    def test_neuronpedia_layers(self):
        from refusal_lens.sae import NEURONPEDIA_LAYERS

        assert NEURONPEDIA_LAYERS == [0, 16, 17, 18, 19]

    def test_analysis_layers(self):
        from refusal_lens.sae import ANALYSIS_LAYERS

        assert ANALYSIS_LAYERS == [6, 10, 13, 17, 20, 22]

    def test_width_constants(self):
        from refusal_lens.sae import WIDTH_16K, WIDTH_262K

        assert WIDTH_16K == "16k"
        assert WIDTH_262K == "262k"

    def test_l0_constant(self):
        from refusal_lens.sae import L0

        assert L0 == "small"

    def test_wide_layers(self):
        from refusal_lens.sae import WIDE_LAYERS

        assert WIDE_LAYERS == [16, 18]


# ---------------------------------------------------------------------------
# Test: Guarded import
# ---------------------------------------------------------------------------


class TestGuardedImport:
    """Test that the module handles missing torch gracefully."""

    def test_module_imports_without_torch(self):
        from refusal_lens import sae

        assert hasattr(sae, "HAS_TORCH")

    def test_has_torch_flag_reflects_availability(self):
        from refusal_lens import sae

        if "torch" not in sys.modules or sys.modules.get("torch") is None:
            assert sae.HAS_TORCH is False


# ---------------------------------------------------------------------------
# Test: _require_torch
# ---------------------------------------------------------------------------


class TestRequireTorch:
    """Test the _require_torch gate function."""

    def test_raises_without_torch(self):
        from refusal_lens.sae import _require_torch

        with patch(f"{_SAE}.HAS_TORCH", False):
            with pytest.raises(ImportError, match="torch, safetensors"):
                _require_torch()

    def test_passes_with_torch(self):
        from refusal_lens.sae import _require_torch

        with patch(f"{_SAE}.HAS_TORCH", True):
            _require_torch()


# ---------------------------------------------------------------------------
# Test: load_sae (mocked)
# ---------------------------------------------------------------------------


class TestLoadSae:
    """Test load_sae filename construction and HF download call.

    These tests mock JumpReLUSAE itself (along with hf_hub_download and
    load_file) so they work regardless of whether real torch is installed.
    The goal is to test the loading *logic*, not the SAE class.
    """

    def _loading_patches(self, mock_hf, mock_load):
        """Patches that replace the SAE class + HF download + load_file."""
        mock_sae_instance = MagicMock()
        mock_sae_instance.to.return_value = mock_sae_instance
        mock_sae_instance.eval.return_value = mock_sae_instance
        mock_sae_class = MagicMock(return_value=mock_sae_instance)
        mock_sae_instance.load_state_dict = MagicMock()

        return [
            patch(f"{_SAE}.HAS_TORCH", True),
            patch(f"{_SAE}.hf_hub_download", mock_hf, create=True),
            patch(f"{_SAE}.load_file", mock_load, create=True),
            patch(f"{_SAE}.JumpReLUSAE", mock_sae_class, create=True),
            patch(f"{_SAE}.torch", MagicMock(), create=True),
        ], mock_sae_class

    def test_filename_construction_no_affine(self):
        from refusal_lens.sae import load_sae

        mock_hf = MagicMock(return_value="/tmp/params.safetensors")
        mock_load = MagicMock(return_value={
            "w_enc": MagicMock(shape=(256, 1024)),
        })

        patches, _ = self._loading_patches(mock_hf, mock_load)
        with apply_patches(*patches):
            load_sae("resid_post", 10, "16k", "small")

            mock_hf.assert_called_once()
            call_kwargs = mock_hf.call_args[1]
            assert call_kwargs["repo_id"] == "google/gemma-scope-2-4b-it"
            assert call_kwargs["filename"] == (
                "resid_post/layer_10_width_16k_l0_small/params.safetensors"
            )

    def test_filename_construction_with_affine(self):
        from refusal_lens.sae import load_sae

        mock_hf = MagicMock(return_value="/tmp/params.safetensors")
        mock_load = MagicMock(return_value={
            "w_enc": MagicMock(shape=(256, 1024)),
        })

        patches, _ = self._loading_patches(mock_hf, mock_load)
        with apply_patches(*patches):
            load_sae("transcoder", 18, "262k", "small", affine=True)

            call_kwargs = mock_hf.call_args[1]
            assert call_kwargs["filename"] == (
                "transcoder/layer_18_width_262k_l0_small_affine/"
                "params.safetensors"
            )

    def test_custom_repo(self):
        from refusal_lens.sae import load_sae

        mock_hf = MagicMock(return_value="/tmp/params.safetensors")
        mock_load = MagicMock(return_value={
            "w_enc": MagicMock(shape=(256, 1024)),
        })

        patches, _ = self._loading_patches(mock_hf, mock_load)
        with apply_patches(*patches):
            load_sae("resid_post", 5, repo="custom/repo")

            call_kwargs = mock_hf.call_args[1]
            assert call_kwargs["repo_id"] == "custom/repo"

    def test_constructs_sae_with_correct_dims(self):
        from refusal_lens.sae import load_sae

        mock_hf = MagicMock(return_value="/tmp/params.safetensors")
        mock_load = MagicMock(return_value={
            "w_enc": MagicMock(shape=(512, 2048)),
        })

        patches, mock_sae_class = self._loading_patches(mock_hf, mock_load)
        with apply_patches(*patches):
            load_sae("resid_post", 10)

            mock_sae_class.assert_called_once_with(
                512, 2048, affine_skip_connection=False
            )

    def test_requires_torch(self):
        from refusal_lens.sae import load_sae

        with patch(f"{_SAE}.HAS_TORCH", False):
            with pytest.raises(ImportError):
                load_sae("resid_post", 0)


# ---------------------------------------------------------------------------
# Test: load_sae_flexible (mocked)
# ---------------------------------------------------------------------------


class TestLoadSaeFlexible:
    """Test that load_sae_flexible tries fallback combinations.

    We mock load_sae itself here to isolate the fallback logic.
    """

    def test_tries_category_all_fallback(self):
        from refusal_lens.sae import load_sae_flexible

        call_log = []

        def mock_load_sae(cat, layer, width, l0, *, affine, repo, device):
            call_log.append(cat)
            if cat == "transcoder":
                msg = "Not found"
                raise Exception(msg)  # noqa: TRY002
            return MagicMock()  # success on transcoder_all

        with patch(f"{_SAE}.HAS_TORCH", True), \
             patch(f"{_SAE}.load_sae", side_effect=mock_load_sae):
            load_sae_flexible("transcoder", 10)
            assert "transcoder" in call_log
            assert "transcoder_all" in call_log

    def test_raises_runtime_error_on_all_failures(self):
        from refusal_lens.sae import load_sae_flexible

        with patch(f"{_SAE}.HAS_TORCH", True), \
             patch(f"{_SAE}.load_sae", side_effect=Exception("fail")):
            with pytest.raises(RuntimeError, match="Could not load"):
                load_sae_flexible("resid_post", 99)

    def test_affine_true_tries_both(self):
        """When affine=True, should try True first, then False as fallback."""
        from refusal_lens.sae import load_sae_flexible

        mock_load = MagicMock(side_effect=Exception("fail"))

        with patch(f"{_SAE}.HAS_TORCH", True), \
             patch(f"{_SAE}.load_sae", mock_load):
            with pytest.raises(RuntimeError):
                load_sae_flexible("transcoder", 0, affine=True)

            # Should have tried: (transcoder, True), (transcoder, False),
            #                    (transcoder_all, True), (transcoder_all, False)
            assert mock_load.call_count == 4

    def test_requires_torch(self):
        from refusal_lens.sae import load_sae_flexible

        with patch(f"{_SAE}.HAS_TORCH", False):
            with pytest.raises(ImportError):
                load_sae_flexible("resid_post", 0)


# ---------------------------------------------------------------------------
# Test: load_sae_set (mocked)
# ---------------------------------------------------------------------------


class TestLoadSaeSet:
    """Test batch loading with mixed success/failure.

    We mock load_sae_flexible to isolate the batch logic.
    """

    def test_skips_failed_layers(self):
        from refusal_lens.sae import load_sae_set

        def mock_flexible(cat, layer, width, l0, *, affine, repo, device):
            if layer == 99:
                msg = "Not found"
                raise RuntimeError(msg)
            return MagicMock()

        with patch(f"{_SAE}.HAS_TORCH", True), \
             patch(f"{_SAE}.load_sae_flexible", side_effect=mock_flexible):
            result = load_sae_set("resid_post", [10, 99, 17])
            assert 10 in result
            assert 17 in result
            assert 99 not in result

    def test_returns_empty_dict_on_all_failures(self):
        from refusal_lens.sae import load_sae_set

        with patch(f"{_SAE}.HAS_TORCH", True), \
             patch(f"{_SAE}.load_sae_flexible", side_effect=Exception("fail")):
            result = load_sae_set("resid_post", [0, 1, 2])
            assert result == {}

    def test_requires_torch(self):
        from refusal_lens.sae import load_sae_set

        with patch(f"{_SAE}.HAS_TORCH", False):
            with pytest.raises(ImportError):
                load_sae_set("resid_post", [0])


# ---------------------------------------------------------------------------
# Test: __init__.py exports
# ---------------------------------------------------------------------------


class TestPackageExports:
    """Test that __init__.py correctly exports sae names."""

    def test_all_contains_sae_names(self):
        import refusal_lens

        expected = {
            "JumpReLUSAE",
            "load_sae",
            "load_sae_flexible",
            "load_sae_set",
            "analyze_width_mismatch",
            "DEFAULT_SCOPE_REPO",
            "NEURONPEDIA_LAYERS",
            "ANALYSIS_LAYERS",
            "WIDTH_16K",
            "WIDTH_262K",
            "WIDE_LAYERS",
        }
        actual = set(refusal_lens.__all__)
        assert expected.issubset(actual), (
            f"Missing from __all__: {expected - actual}"
        )

    def test_sae_names_importable(self):
        from refusal_lens import (
            ANALYSIS_LAYERS,
            DEFAULT_SCOPE_REPO,
            JumpReLUSAE,
            NEURONPEDIA_LAYERS,
            WIDE_LAYERS,
            WIDTH_16K,
            WIDTH_262K,
            analyze_width_mismatch,
            load_sae,
            load_sae_flexible,
            load_sae_set,
        )

        assert callable(load_sae)
        assert callable(load_sae_flexible)
        assert callable(load_sae_set)
        assert callable(analyze_width_mismatch)
        assert DEFAULT_SCOPE_REPO == "google/gemma-scope-2-4b-it"
        assert WIDTH_16K == "16k"
        assert WIDTH_262K == "262k"
        assert isinstance(NEURONPEDIA_LAYERS, list)
        assert isinstance(ANALYSIS_LAYERS, list)
        assert isinstance(WIDE_LAYERS, list)


# ===========================================================================
# TIER 2: Real-package tests (skipped if torch is not installed)
# ===========================================================================


@requires_torch
class TestJumpReLUSAERealTorch:
    """Test JumpReLUSAE with real tensors."""

    def test_init_basic(self):
        from refusal_lens.sae import JumpReLUSAE

        sae = JumpReLUSAE(64, 256)
        assert sae.d_in == 64
        assert sae.d_sae == 256
        assert sae.affine_skip_connection is None

    def test_init_with_affine(self):
        from refusal_lens.sae import JumpReLUSAE

        sae = JumpReLUSAE(64, 256, affine_skip_connection=True)
        assert sae.affine_skip_connection is not None
        assert sae.affine_skip_connection.shape == (64, 64)

    def test_parameter_shapes(self):
        from refusal_lens.sae import JumpReLUSAE

        sae = JumpReLUSAE(32, 128)
        assert sae.w_enc.shape == (32, 128)
        assert sae.w_dec.shape == (128, 32)
        assert sae.threshold.shape == (128,)
        assert sae.b_enc.shape == (128,)
        assert sae.b_dec.shape == (32,)

    def test_encode_output_shape(self):
        from refusal_lens.sae import JumpReLUSAE

        sae = JumpReLUSAE(32, 128)
        x = torch.randn(1, 32)
        acts = sae.encode(x)
        assert acts.shape == (1, 128)

    def test_encode_is_sparse(self):
        """JumpReLU should produce sparse activations (mostly zeros)."""
        from refusal_lens.sae import JumpReLUSAE

        sae = JumpReLUSAE(32, 128)
        # Set threshold high so most activations are zero
        with torch.no_grad():
            sae.threshold.fill_(10.0)
        x = torch.randn(1, 32)
        acts = sae.encode(x)
        # With high threshold, almost all activations should be zero
        assert (acts == 0).sum() > 100

    def test_encode_nonnegative(self):
        """JumpReLU output should always be non-negative."""
        from refusal_lens.sae import JumpReLUSAE

        sae = JumpReLUSAE(32, 128)
        x = torch.randn(10, 32)
        acts = sae.encode(x)
        assert (acts >= 0).all()

    def test_decode_output_shape(self):
        from refusal_lens.sae import JumpReLUSAE

        sae = JumpReLUSAE(32, 128)
        acts = torch.randn(1, 128)
        recon = sae.decode(acts)
        assert recon.shape == (1, 32)

    def test_forward_output_shape(self):
        from refusal_lens.sae import JumpReLUSAE

        sae = JumpReLUSAE(32, 128)
        x = torch.randn(1, 32)
        out = sae.forward(x)
        assert out.shape == (1, 32)

    def test_forward_with_affine_output_shape(self):
        from refusal_lens.sae import JumpReLUSAE

        sae = JumpReLUSAE(32, 128, affine_skip_connection=True)
        x = torch.randn(1, 32)
        out = sae.forward(x)
        assert out.shape == (1, 32)

    def test_forward_without_affine_equals_decode_encode(self):
        """forward(x) should equal decode(encode(x)) when no affine skip."""
        from refusal_lens.sae import JumpReLUSAE

        sae = JumpReLUSAE(32, 128)
        x = torch.randn(1, 32)
        expected = sae.decode(sae.encode(x))
        actual = sae.forward(x)
        assert torch.allclose(actual, expected)

    def test_forward_with_affine_adds_skip(self):
        """forward(x) with affine should differ from decode(encode(x))."""
        from refusal_lens.sae import JumpReLUSAE

        sae = JumpReLUSAE(32, 128, affine_skip_connection=True)
        # Set non-zero affine weights so the skip actually contributes
        with torch.no_grad():
            sae._affine_skip.fill_(0.1)
        x = torch.randn(1, 32)
        no_skip = sae.decode(sae.encode(x))
        with_skip = sae.forward(x)
        # They should differ because of the affine skip
        assert not torch.allclose(no_skip, with_skip)

    def test_batch_input(self):
        """SAE should handle batched input."""
        from refusal_lens.sae import JumpReLUSAE

        sae = JumpReLUSAE(64, 256)
        x = torch.randn(8, 64)
        acts = sae.encode(x)
        assert acts.shape == (8, 256)
        recon = sae.forward(x)
        assert recon.shape == (8, 64)

    def test_eval_mode(self):
        """SAE should support eval mode."""
        from refusal_lens.sae import JumpReLUSAE

        sae = JumpReLUSAE(32, 128).eval()
        assert not sae.training

    def test_parameters_are_trainable(self):
        """All parameters should be registered properly."""
        from refusal_lens.sae import JumpReLUSAE

        sae = JumpReLUSAE(32, 128)
        param_names = {name for name, _ in sae.named_parameters()}
        assert "w_enc" in param_names
        assert "w_dec" in param_names
        assert "threshold" in param_names
        assert "b_enc" in param_names
        assert "b_dec" in param_names

    def test_affine_parameter_registered(self):
        """Affine skip should be a registered parameter."""
        from refusal_lens.sae import JumpReLUSAE

        sae = JumpReLUSAE(32, 128, affine_skip_connection=True)
        param_names = {name for name, _ in sae.named_parameters()}
        assert "_affine_skip" in param_names

    def test_load_state_dict_round_trip(self):
        """State dict should round-trip correctly."""
        from refusal_lens.sae import JumpReLUSAE

        sae1 = JumpReLUSAE(32, 128)
        with torch.no_grad():
            sae1.w_enc.normal_()
            sae1.w_dec.normal_()
            sae1.b_enc.fill_(0.5)

        state = sae1.state_dict()
        sae2 = JumpReLUSAE(32, 128)
        sae2.load_state_dict(state)

        assert torch.equal(sae1.w_enc, sae2.w_enc)
        assert torch.equal(sae1.w_dec, sae2.w_dec)
        assert torch.equal(sae1.b_enc, sae2.b_enc)


@requires_torch
class TestAnalyzeWidthMismatchRealTorch:
    """Test analyze_width_mismatch with real tensors."""

    def test_returns_expected_keys(self):
        from refusal_lens.sae import JumpReLUSAE, analyze_width_mismatch

        tc_16k = JumpReLUSAE(64, 100)
        tc_wide = JumpReLUSAE(64, 500)
        with torch.no_grad():
            tc_16k.w_dec.normal_()
            tc_wide.w_dec.normal_()

        result = analyze_width_mismatch(0, tc_16k, tc_wide, k=5)

        assert "top_indices" in result
        assert "top_cosines" in result
        assert "n_high_overlap" in result
        assert len(result["top_indices"]) == 5
        assert len(result["top_cosines"]) == 5
        assert isinstance(result["n_high_overlap"], int)

    def test_cosines_in_valid_range(self):
        from refusal_lens.sae import JumpReLUSAE, analyze_width_mismatch

        tc_16k = JumpReLUSAE(64, 100)
        tc_wide = JumpReLUSAE(64, 500)
        with torch.no_grad():
            tc_16k.w_dec.normal_()
            tc_wide.w_dec.normal_()

        result = analyze_width_mismatch(5, tc_16k, tc_wide, k=10)

        for cos in result["top_cosines"]:
            assert 0.0 <= cos <= 1.0, f"Cosine {cos} out of [0, 1] range"

    def test_identical_decoder_gives_high_cosine(self):
        """If a wide feature has the same decoder as the 16k feature, cosine ≈ 1."""
        from refusal_lens.sae import JumpReLUSAE, analyze_width_mismatch

        tc_16k = JumpReLUSAE(64, 100)
        tc_wide = JumpReLUSAE(64, 500)
        with torch.no_grad():
            tc_16k.w_dec.normal_()
            tc_wide.w_dec.normal_()
            # Copy 16k feature 0 decoder into wide feature 42
            tc_wide.w_dec[42] = tc_16k.w_dec[0].clone()

        result = analyze_width_mismatch(0, tc_16k, tc_wide, k=5)

        assert 42 in result["top_indices"]
        # The cosine for the matching feature should be very high
        idx = result["top_indices"].index(42)
        assert result["top_cosines"][idx] > 0.99

    def test_n_high_overlap_counts_above_threshold(self):
        from refusal_lens.sae import JumpReLUSAE, analyze_width_mismatch

        tc_16k = JumpReLUSAE(64, 10)
        tc_wide = JumpReLUSAE(64, 20)
        with torch.no_grad():
            tc_16k.w_dec.normal_()
            # Make all wide decoders orthogonal to 16k feature 0
            tc_wide.w_dec.zero_()
            # Set one to be parallel
            tc_wide.w_dec[3] = tc_16k.w_dec[0].clone()

        result = analyze_width_mismatch(0, tc_16k, tc_wide, k=10)

        # Only feature 3 should have high overlap
        assert result["n_high_overlap"] >= 1

    def test_sorted_by_cosine_descending(self):
        from refusal_lens.sae import JumpReLUSAE, analyze_width_mismatch

        tc_16k = JumpReLUSAE(64, 50)
        tc_wide = JumpReLUSAE(64, 200)
        with torch.no_grad():
            tc_16k.w_dec.normal_()
            tc_wide.w_dec.normal_()

        result = analyze_width_mismatch(0, tc_16k, tc_wide, k=10)

        cosines = result["top_cosines"]
        assert cosines == sorted(cosines, reverse=True)


@requires_torch
class TestHasTorchFlag:
    """Verify HAS_TORCH is True when torch is installed."""

    def test_has_torch_is_true(self):
        from refusal_lens.sae import HAS_TORCH

        assert HAS_TORCH is True

    def test_require_torch_does_not_raise(self):
        from refusal_lens.sae import _require_torch

        _require_torch()


# ===========================================================================
# TIER 3: Real HuggingFace download tests
# ===========================================================================

# These tests actually download SAE weights from HuggingFace.
# They require torch + network access and are slower.
# The downloaded files are cleaned up after each test.

requires_hf = pytest.mark.skipif(
    not _HAS_REAL_TORCH, reason="torch not installed (needed for HF loading)"
)


@requires_hf
class TestRealSAELoading:
    """Test actual SAE loading from HuggingFace Gemma Scope 2."""

    def _cleanup_hf_cache(self, repo: str) -> None:
        """Remove cached HuggingFace files for a repo to save disk space."""
        try:
            from huggingface_hub import scan_cache_dir

            cache_info = scan_cache_dir()
            for repo_info in cache_info.repos:
                if repo_info.repo_id == repo:
                    for revision in repo_info.revisions:
                        strategy = cache_info.delete_revisions(revision.commit_hash)
                        strategy.execute()
                    break
        except Exception:  # noqa: BLE001
            pass  # Cache cleanup is best-effort

    def test_load_residual_sae_16k(self):
        """Download and verify a 16k residual SAE from Gemma Scope 2."""
        from refusal_lens.sae import (
            DEFAULT_SCOPE_REPO,
            JumpReLUSAE,
            load_sae,
        )

        try:
            sae = load_sae(
                "resid_post", 17, "16k", "small", device="cpu"
            )

            # Verify it's a proper JumpReLUSAE
            assert isinstance(sae, JumpReLUSAE)
            assert not sae.training  # should be in eval mode

            # Verify dimensions make sense for Gemma 3 4B
            # Hidden dim is 2560 for Gemma 3 4B
            assert sae.d_in > 0
            assert sae.d_sae > 0
            assert sae.d_sae > sae.d_in  # SAE is overcomplete

            # Verify we can run a forward pass
            x = torch.randn(1, sae.d_in)
            acts = sae.encode(x)
            assert acts.shape == (1, sae.d_sae)
            assert (acts >= 0).all()  # JumpReLU is non-negative

            recon = sae.forward(x)
            assert recon.shape == (1, sae.d_in)

            print(f"Residual SAE layer 17: d_in={sae.d_in}, d_sae={sae.d_sae}")

        finally:
            self._cleanup_hf_cache(DEFAULT_SCOPE_REPO)

    def test_load_transcoder_16k_with_flexible(self):
        """Download and verify a 16k transcoder using load_sae_flexible."""
        from refusal_lens.sae import (
            DEFAULT_SCOPE_REPO,
            JumpReLUSAE,
            load_sae_flexible,
        )

        try:
            tc = load_sae_flexible(
                "transcoder", 17, "16k", "small",
                affine=True, device="cpu",
            )

            assert isinstance(tc, JumpReLUSAE)
            assert not tc.training
            assert tc.d_in > 0
            assert tc.d_sae > 0

            # Run forward pass
            x = torch.randn(1, tc.d_in)
            out = tc.forward(x)
            assert out.shape == (1, tc.d_in)

            # Check if affine skip was loaded
            has_affine = tc.affine_skip_connection is not None
            print(
                f"Transcoder layer 17: d_in={tc.d_in}, d_sae={tc.d_sae}, "
                f"affine={has_affine}"
            )

        finally:
            self._cleanup_hf_cache(DEFAULT_SCOPE_REPO)

    def test_load_sae_flexible_bad_layer_raises(self):
        """Requesting a non-existent layer should raise RuntimeError."""
        from refusal_lens.sae import load_sae_flexible

        with pytest.raises(RuntimeError, match="Could not load"):
            load_sae_flexible("resid_post", 999, "16k", "small", device="cpu")
