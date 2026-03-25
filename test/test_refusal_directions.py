"""Tests for refusal_lens.refusal_directions — real torch tests, no mocks."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from refusal_lens.refusal_directions import (
    HAS_TORCH,
    RefusalDirectionResult,
    RefusalDirectionSet,
    collect_resid_acts_multipos,
    compute_refusal_directions,
    find_best_layer,
    gather_residual_activations,
    get_model_layers,
    load_directions,
    save_directions,
)


# ---------------------------------------------------------------------------
# Helpers: lightweight fake model for hook-based tests
# ---------------------------------------------------------------------------

class FakeLayer(torch.nn.Module):
    """A transformer block that returns a fixed hidden state."""

    def __init__(self, d_model: int, seq_len: int = 10) -> None:
        super().__init__()
        self.hidden = torch.randn(1, seq_len, d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (self.hidden,)


class FakeModel(torch.nn.Module):
    """Minimal model with Gemma/LLaMA-style .model.layers structure."""

    def __init__(self, n_layers: int = 3, d_model: int = 32, seq_len: int = 10) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self._layers = torch.nn.ModuleList(
            [FakeLayer(d_model, seq_len) for _ in range(n_layers)]
        )
        # Expose as model.model.layers (Gemma/LLaMA pattern)
        self.model = torch.nn.Module()
        self.model.layers = self._layers

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = torch.randn(1, self.seq_len, self.d_model)
        for layer in self._layers:
            x = layer(x)[0]
        return x


def _make_synthetic_acts(
    layers: list[int],
    n_prompts: int = 20,
    n_pos: int = 3,
    d_model: int = 64,
    offset: float = 0.0,
) -> dict[int, torch.Tensor]:
    """Create synthetic activation tensors for testing."""
    return {
        l: torch.randn(n_prompts, n_pos, d_model) + offset
        for l in layers
    }


# ---------------------------------------------------------------------------
# HAS_TORCH flag
# ---------------------------------------------------------------------------

class TestHasTorch:
    def test_has_torch_is_true(self):
        assert HAS_TORCH is True


# ---------------------------------------------------------------------------
# get_model_layers
# ---------------------------------------------------------------------------

class TestGetModelLayers:
    def test_gemma_llama_style(self):
        model = FakeModel(n_layers=4)
        layers = get_model_layers(model)
        assert len(layers) == 4
        assert isinstance(layers[0], FakeLayer)

    def test_gpt2_style(self):
        class GPT2Like(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = torch.nn.Module()
                self.transformer.h = torch.nn.ModuleList(
                    [FakeLayer(32) for _ in range(2)]
                )
        layers = get_model_layers(GPT2Like())
        assert len(layers) == 2

    def test_gpt_neox_style(self):
        class NeoXLike(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gpt_neox = torch.nn.Module()
                self.gpt_neox.layers = torch.nn.ModuleList(
                    [FakeLayer(32) for _ in range(3)]
                )
        layers = get_model_layers(NeoXLike())
        assert len(layers) == 3

    def test_unknown_architecture_raises(self):
        class Unknown(torch.nn.Module):
            pass
        with pytest.raises(ValueError, match="Cannot find transformer blocks"):
            get_model_layers(Unknown())


# ---------------------------------------------------------------------------
# gather_residual_activations
# ---------------------------------------------------------------------------

class TestGatherResidualActivations:
    def test_returns_correct_shape(self):
        model = FakeModel(n_layers=3, d_model=16, seq_len=8)
        input_ids = torch.zeros(1, 8, dtype=torch.long)
        result = gather_residual_activations(model, 0, input_ids)
        assert result.shape == (1, 8, 16)

    def test_captures_specific_layer(self):
        model = FakeModel(n_layers=3, d_model=16, seq_len=8)
        input_ids = torch.zeros(1, 8, dtype=torch.long)
        # Each FakeLayer has a fixed .hidden tensor — layer 0 and layer 1 differ
        act0 = gather_residual_activations(model, 0, input_ids)
        act1 = gather_residual_activations(model, 1, input_ids)
        assert not torch.allclose(act0, act1)

    def test_result_is_detached(self):
        model = FakeModel(n_layers=2, d_model=8, seq_len=4)
        input_ids = torch.zeros(1, 4, dtype=torch.long)
        result = gather_residual_activations(model, 0, input_ids)
        assert not result.requires_grad

    def test_hook_is_removed_after_call(self):
        model = FakeModel(n_layers=2, d_model=8, seq_len=4)
        input_ids = torch.zeros(1, 4, dtype=torch.long)
        hooks_before = len(model.model.layers[0]._forward_hooks)
        gather_residual_activations(model, 0, input_ids)
        hooks_after = len(model.model.layers[0]._forward_hooks)
        assert hooks_after == hooks_before

    def test_hook_removed_on_error(self):
        """Hook must be removed even if forward pass fails."""
        class FailingLayer(torch.nn.Module):
            def forward(self, x):
                raise RuntimeError("intentional failure")

        model = FakeModel(n_layers=2)
        model.model.layers[0] = FailingLayer()
        input_ids = torch.zeros(1, 4, dtype=torch.long)

        with pytest.raises(RuntimeError, match="intentional failure"):
            gather_residual_activations(model, 0, input_ids)
        # Hook should still be cleaned up
        assert len(model.model.layers[0]._forward_hooks) == 0


# ---------------------------------------------------------------------------
# compute_refusal_directions — mean method
# ---------------------------------------------------------------------------

class TestComputeRefusalDirectionsMean:
    def test_returns_unit_norm(self):
        harmful = _make_synthetic_acts([0, 1], offset=2.0)
        harmless = _make_synthetic_acts([0, 1], offset=-2.0)
        dirs = compute_refusal_directions(harmful, harmless, method="mean")
        for l in dirs:
            assert abs(dirs[l].direction.norm().item() - 1.0) < 1e-5

    def test_direction_shape(self):
        harmful = _make_synthetic_acts([5], d_model=128)
        harmless = _make_synthetic_acts([5], d_model=128)
        dirs = compute_refusal_directions(harmful, harmless, method="mean")
        assert dirs[5].direction.shape == (128,)

    def test_separation_is_positive(self):
        harmful = _make_synthetic_acts([0], offset=1.0)
        harmless = _make_synthetic_acts([0], offset=-1.0)
        dirs = compute_refusal_directions(harmful, harmless, method="mean")
        assert dirs[0].separation > 0

    def test_large_offset_gives_large_separation(self):
        harmful_big = _make_synthetic_acts([0], offset=10.0)
        harmless_big = _make_synthetic_acts([0], offset=-10.0)
        harmful_small = _make_synthetic_acts([0], offset=0.1)
        harmless_small = _make_synthetic_acts([0], offset=-0.1)

        dirs_big = compute_refusal_directions(harmful_big, harmless_big)
        dirs_small = compute_refusal_directions(harmful_small, harmless_small)
        assert dirs_big[0].separation > dirs_small[0].separation

    def test_method_stored(self):
        harmful = _make_synthetic_acts([0])
        harmless = _make_synthetic_acts([0])
        dirs = compute_refusal_directions(harmful, harmless, method="mean")
        assert dirs[0].method == "mean"

    def test_layer_stored(self):
        harmful = _make_synthetic_acts([7, 12])
        harmless = _make_synthetic_acts([7, 12])
        dirs = compute_refusal_directions(harmful, harmless)
        assert dirs[7].layer == 7
        assert dirs[12].layer == 12

    def test_multiple_layers(self):
        layers = [0, 5, 10, 15]
        harmful = _make_synthetic_acts(layers)
        harmless = _make_synthetic_acts(layers)
        dirs = compute_refusal_directions(harmful, harmless)
        assert set(dirs.keys()) == set(layers)

    def test_direction_points_harmful_minus_harmless(self):
        """Direction should have positive cosine with (harmful_mean - harmless_mean)."""
        d_model = 32
        # Create clearly separated data
        harmful = {0: torch.ones(10, 1, d_model) * 5}
        harmless = {0: torch.ones(10, 1, d_model) * -5}
        dirs = compute_refusal_directions(harmful, harmless, method="mean")
        # The difference is all-positive, so direction should have positive entries
        cos = torch.nn.functional.cosine_similarity(
            dirs[0].direction.unsqueeze(0),
            (torch.ones(d_model) * 10).unsqueeze(0),
        ).item()
        assert cos > 0.99


# ---------------------------------------------------------------------------
# compute_refusal_directions — PCA method
# ---------------------------------------------------------------------------

class TestComputeRefusalDirectionsPCA:
    def test_pca_returns_direction(self):
        harmful = _make_synthetic_acts([0], n_prompts=30, offset=2.0)
        harmless = _make_synthetic_acts([0], n_prompts=30, offset=-2.0)
        dirs = compute_refusal_directions(harmful, harmless, method="pca")
        assert dirs[0].direction.shape[0] == 64  # d_model

    def test_pca_method_stored(self):
        harmful = _make_synthetic_acts([0], n_prompts=20)
        harmless = _make_synthetic_acts([0], n_prompts=20)
        dirs = compute_refusal_directions(harmful, harmless, method="pca")
        assert dirs[0].method == "pca"

    def test_pca_and_mean_roughly_agree_on_clear_signal(self):
        """When the signal has high variance along the mean-difference direction,
        PCA's top component should align with the mean direction."""
        torch.manual_seed(42)
        d_model = 32
        n = 100
        # Per-sample variable offsets so PCA sees variance in dim 0
        # Harmful: dim 0 ~ Uniform(3, 7), Harmless: dim 0 ~ Uniform(-7, -3)
        harmful = {0: torch.randn(n, 1, d_model) * 0.1}
        harmful[0][:, :, 0] += torch.rand(n, 1) * 4 + 3  # [3, 7]
        harmless = {0: torch.randn(n, 1, d_model) * 0.1}
        harmless[0][:, :, 0] -= torch.rand(n, 1) * 4 + 3  # [-7, -3]

        dirs_mean = compute_refusal_directions(harmful, harmless, method="mean")
        dirs_pca = compute_refusal_directions(harmful, harmless, method="pca")

        cos = torch.nn.functional.cosine_similarity(
            dirs_mean[0].direction.unsqueeze(0),
            dirs_pca[0].direction.unsqueeze(0),
        ).abs().item()
        # Should be close (abs because PCA sign is arbitrary)
        assert cos > 0.9

    def test_unknown_method_raises(self):
        harmful = _make_synthetic_acts([0])
        harmless = _make_synthetic_acts([0])
        with pytest.raises(ValueError, match="Unknown method"):
            compute_refusal_directions(harmful, harmless, method="invalid")


# ---------------------------------------------------------------------------
# find_best_layer
# ---------------------------------------------------------------------------

class TestFindBestLayer:
    def test_selects_highest_separation(self):
        dirs = {
            0: RefusalDirectionResult(torch.randn(32), 0, "mean", 1.0),
            5: RefusalDirectionResult(torch.randn(32), 5, "mean", 3.0),
            10: RefusalDirectionResult(torch.randn(32), 10, "mean", 2.0),
        }
        best_layer, best_result = find_best_layer(dirs)
        assert best_layer == 5
        assert best_result.separation == 3.0

    def test_single_layer(self):
        dirs = {
            7: RefusalDirectionResult(torch.randn(16), 7, "mean", 1.5),
        }
        best_layer, _ = find_best_layer(dirs)
        assert best_layer == 7

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No directions provided"):
            find_best_layer({})


# ---------------------------------------------------------------------------
# save_directions / load_directions roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoadDirections:
    def test_roundtrip(self, tmp_path):
        d_model = 48
        dirs = {
            6: RefusalDirectionResult(torch.randn(d_model), 6, "mean", 1.5),
            10: RefusalDirectionResult(torch.randn(d_model), 10, "mean", 2.3),
            17: RefusalDirectionResult(torch.randn(d_model), 17, "pca", 1.8),
        }
        save_directions(dirs, tmp_path)
        loaded = load_directions(tmp_path)

        assert loaded.best_layer == 10  # highest separation
        assert set(loaded.directions.keys()) == {6, 10, 17}
        for l in dirs:
            assert torch.allclose(dirs[l].direction, loaded.directions[l].direction)
            assert loaded.directions[l].method == dirs[l].method
            assert abs(loaded.directions[l].separation - dirs[l].separation) < 1e-4

    def test_creates_pt_files(self, tmp_path):
        dirs = {
            3: RefusalDirectionResult(torch.randn(16), 3, "mean", 1.0),
            15: RefusalDirectionResult(torch.randn(16), 15, "mean", 2.0),
        }
        save_directions(dirs, tmp_path)
        assert (tmp_path / "layer_03.pt").exists()
        assert (tmp_path / "layer_15.pt").exists()
        assert (tmp_path / "summary.json").exists()

    def test_summary_json_content(self, tmp_path):
        dirs = {
            8: RefusalDirectionResult(torch.randn(16), 8, "mean", 1.234567),
        }
        save_directions(dirs, tmp_path)
        with open(tmp_path / "summary.json") as f:
            summary = json.load(f)
        assert summary["best_layer"] == 8
        assert "8" in summary["layers"]
        assert summary["layers"]["8"]["method"] == "mean"
        assert abs(summary["layers"]["8"]["separation"] - 1.234567) < 1e-5

    def test_best_layer_override(self, tmp_path):
        dirs = {
            0: RefusalDirectionResult(torch.randn(16), 0, "mean", 5.0),
            1: RefusalDirectionResult(torch.randn(16), 1, "mean", 1.0),
        }
        # Override: pick layer 1 even though layer 0 has higher separation
        save_directions(dirs, tmp_path, best_layer=1)
        loaded = load_directions(tmp_path)
        assert loaded.best_layer == 1

    def test_creates_output_dir(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        dirs = {0: RefusalDirectionResult(torch.randn(8), 0, "mean", 1.0)}
        save_directions(dirs, nested)
        assert nested.exists()
        assert (nested / "summary.json").exists()

    def test_load_missing_summary_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="summary.json"):
            load_directions(tmp_path)

    def test_loaded_set_has_metadata(self, tmp_path):
        dirs = {0: RefusalDirectionResult(torch.randn(8), 0, "mean", 1.0)}
        save_directions(dirs, tmp_path)
        loaded = load_directions(tmp_path)
        assert isinstance(loaded, RefusalDirectionSet)
        assert "best_layer" in loaded.metadata
        assert "layers" in loaded.metadata


# ---------------------------------------------------------------------------
# End-to-end: compute → find_best → save → load
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_full_pipeline(self, tmp_path):
        """Simulate the full refusal direction workflow with synthetic data."""
        layers = [6, 10, 13, 17]
        d_model = 64
        n_prompts = 25

        # Layer 13 gets the strongest signal
        harmful = {}
        harmless = {}
        for l in layers:
            offset = 5.0 if l == 13 else 1.0
            harmful[l] = torch.randn(n_prompts, 3, d_model) + offset
            harmless[l] = torch.randn(n_prompts, 3, d_model) - offset

        # Compute
        dirs = compute_refusal_directions(harmful, harmless, method="mean")
        assert len(dirs) == 4

        # Find best
        best_layer, best_result = find_best_layer(dirs)
        assert best_layer == 13
        assert best_result.separation > dirs[6].separation

        # Save
        out_dir = tmp_path / "computed_directions"
        save_directions(dirs, out_dir)

        # Load
        loaded = load_directions(out_dir)
        assert loaded.best_layer == 13
        assert len(loaded.directions) == 4

        # Verify loaded directions are unit-norm (mean method)
        for l in loaded.directions:
            norm = loaded.directions[l].direction.norm().item()
            assert abs(norm - 1.0) < 1e-4

    def test_pca_pipeline(self, tmp_path):
        """Same pipeline with PCA method."""
        layers = [0, 1]
        harmful = _make_synthetic_acts(layers, n_prompts=30, offset=3.0)
        harmless = _make_synthetic_acts(layers, n_prompts=30, offset=-3.0)

        dirs = compute_refusal_directions(harmful, harmless, method="pca")
        save_directions(dirs, tmp_path)
        loaded = load_directions(tmp_path)

        for l in loaded.directions:
            assert loaded.directions[l].method == "pca"


# ---------------------------------------------------------------------------
# collect_resid_acts_multipos with FakeModel
# ---------------------------------------------------------------------------

class TestCollectResidActsMultipos:
    """Test activation collection using FakeModel (no real LLM needed)."""

    def _make_fake_tokenizer(self, seq_len: int = 10):
        """Create a minimal tokenizer-like object matching model_loader.tokenize_prompt."""
        class FakeTokenizer:
            def apply_chat_template(self, messages, tokenize=True,
                                     return_tensors=None,
                                     add_generation_prompt=True):
                return torch.zeros(1, seq_len, dtype=torch.long)

            @property
            def chat_template(self):
                return "fake_template"
        return FakeTokenizer()

    def test_last_position(self):
        model = FakeModel(n_layers=2, d_model=16, seq_len=8)
        tokenizer = self._make_fake_tokenizer(seq_len=8)
        prompts = ["hello", "world"]

        # Patch tokenize_prompt to use our fake tokenizer directly
        acts = collect_resid_acts_multipos(
            model, tokenizer, prompts, [0, 1], positions="last",
        )
        assert set(acts.keys()) == {0, 1}
        # 2 prompts, 1 position ("last"), 16 d_model
        assert acts[0].shape == (2, 1, 16)

    def test_max_prompts(self):
        model = FakeModel(n_layers=1, d_model=8, seq_len=4)
        tokenizer = self._make_fake_tokenizer(seq_len=4)
        prompts = ["a", "b", "c", "d", "e"]

        acts = collect_resid_acts_multipos(
            model, tokenizer, prompts, [0], positions="last", max_prompts=2,
        )
        assert acts[0].shape[0] == 2  # only 2 prompts processed

    def test_custom_positions(self):
        model = FakeModel(n_layers=1, d_model=8, seq_len=6)
        tokenizer = self._make_fake_tokenizer(seq_len=6)

        acts = collect_resid_acts_multipos(
            model, tokenizer, ["test"], [0], positions=[-1, -2],
        )
        assert acts[0].shape == (1, 2, 8)  # 1 prompt, 2 positions

    def test_unknown_positions_raises(self):
        model = FakeModel(n_layers=1, d_model=8, seq_len=4)
        tokenizer = self._make_fake_tokenizer(seq_len=4)

        with pytest.raises(ValueError, match="Unknown positions mode"):
            collect_resid_acts_multipos(
                model, tokenizer, ["test"], [0], positions="invalid",
            )


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------

class TestPackageExports:
    def test_all_exported(self):
        import refusal_lens
        expected = [
            "RefusalDirectionResult", "RefusalDirectionSet",
            "compute_refusal_directions", "find_best_layer",
            "gather_residual_activations", "get_model_layers",
            "collect_resid_acts_multipos",
            "save_directions", "load_directions",
        ]
        for name in expected:
            assert hasattr(refusal_lens, name), f"{name} not exported"

    def test_in_all(self):
        import refusal_lens
        for name in ["RefusalDirectionResult", "RefusalDirectionSet",
                      "compute_refusal_directions", "find_best_layer",
                      "save_directions", "load_directions"]:
            assert name in refusal_lens.__all__, f"{name} not in __all__"
