"""Tests for refusal_lens.config — centralized configuration."""
from __future__ import annotations

from pathlib import Path

from refusal_lens import config


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

class TestPaths:
    def test_repo_root_is_directory(self):
        assert config.REPO_ROOT.is_dir()

    def test_dataset_dir_exists(self):
        assert config.DATASET_DIR.exists()

    def test_splits_dir_exists(self):
        assert config.DATASET_SPLITS_DIR.exists()

    def test_processed_dir_exists(self):
        assert config.DATASET_PROCESSED_DIR.exists()

    def test_results_dir_under_repo(self):
        assert config.RESULTS_DIR.parent.parent == config.REPO_ROOT

    def test_subdirs_under_results(self):
        assert config.COMPUTED_DIRECTIONS_DIR.parent == config.RESULTS_DIR
        assert config.CIRCUITS_DIR.parent == config.RESULTS_DIR
        assert config.FEATURES_DIR.parent == config.RESULTS_DIR
        assert config.VALIDATION_DIR.parent == config.RESULTS_DIR

    def test_all_paths_are_path_objects(self):
        for name in [
            "REPO_ROOT", "DATASET_DIR", "DATASET_SPLITS_DIR",
            "DATASET_PROCESSED_DIR", "RESULTS_DIR", "COMPUTED_DIRECTIONS_DIR",
            "CIRCUITS_DIR", "FEATURES_DIR", "VALIDATION_DIR",
        ]:
            assert isinstance(getattr(config, name), Path), f"{name} is not a Path"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TestModel:
    def test_model_name(self):
        assert config.MODEL_NAME == "google/gemma-3-4b-it"

    def test_device(self):
        assert config.DEVICE == "cuda"

    def test_dtype(self):
        assert config.DTYPE == "bfloat16"


# ---------------------------------------------------------------------------
# SAE / Transcoder
# ---------------------------------------------------------------------------

class TestSAE:
    def test_scope_repo(self):
        assert config.DEFAULT_SCOPE_REPO == "google/gemma-scope-2-4b-it"

    def test_widths(self):
        assert config.WIDTH_16K == "16k"
        assert config.WIDTH_262K == "262k"

    def test_l0(self):
        assert config.L0 == "small"

    def test_neuronpedia_layers(self):
        assert config.NEURONPEDIA_LAYERS == [0, 16, 17, 18, 19]

    def test_analysis_layers(self):
        assert config.ANALYSIS_LAYERS == [6, 10, 13, 17, 20, 22]

    def test_wide_layers(self):
        assert config.WIDE_LAYERS == [16, 18]


# ---------------------------------------------------------------------------
# Refusal direction
# ---------------------------------------------------------------------------

class TestRefusalDirection:
    def test_refusal_layers_range(self):
        assert config.REFUSAL_LAYERS == list(range(0, 34))

    def test_refusal_position(self):
        assert config.REFUSAL_POSITION == "last_k"

    def test_measurement_layer(self):
        assert config.MEASUREMENT_LAYER == 32
        assert config.MEASUREMENT_LAYER in config.REFUSAL_LAYERS


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TestDataset:
    def test_n_train(self):
        assert config.N_TRAIN == 128

    def test_n_eval(self):
        assert config.N_EVAL == 64

    def test_seed(self):
        assert config.SEED == 42


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

class TestGeneration:
    def test_max_new_tokens(self):
        assert config.MAX_NEW_TOKENS == 512

    def test_n_base_prompts(self):
        assert config.N_BASE_PROMPTS == 15


# ---------------------------------------------------------------------------
# Consistency with existing modules
# ---------------------------------------------------------------------------

class TestConsistencyWithModules:
    def test_sae_constants_match(self):
        from refusal_lens.sae import (
            DEFAULT_SCOPE_REPO,
            NEURONPEDIA_LAYERS,
            ANALYSIS_LAYERS,
            WIDTH_16K,
            WIDTH_262K,
            WIDE_LAYERS,
        )
        assert config.DEFAULT_SCOPE_REPO == DEFAULT_SCOPE_REPO
        assert config.NEURONPEDIA_LAYERS == NEURONPEDIA_LAYERS
        assert config.ANALYSIS_LAYERS == ANALYSIS_LAYERS
        assert config.WIDTH_16K == WIDTH_16K
        assert config.WIDTH_262K == WIDTH_262K
        assert config.WIDE_LAYERS == WIDE_LAYERS

    def test_data_loader_constants_match(self):
        from refusal_lens.data_loader import (
            DEFAULT_N_TRAIN,
            DEFAULT_N_EVAL,
            DEFAULT_SEED,
            _DEFAULT_DATASET_DIR,
        )
        assert config.N_TRAIN == DEFAULT_N_TRAIN
        assert config.N_EVAL == DEFAULT_N_EVAL
        assert config.SEED == DEFAULT_SEED
        assert config.DATASET_DIR == _DEFAULT_DATASET_DIR


# ---------------------------------------------------------------------------
# Package export
# ---------------------------------------------------------------------------

class TestPackageExport:
    def test_config_exported(self):
        import refusal_lens
        assert hasattr(refusal_lens, "config")

    def test_config_in_all(self):
        import refusal_lens
        assert "config" in refusal_lens.__all__

    def test_config_is_module(self):
        import types
        import refusal_lens
        assert isinstance(refusal_lens.config, types.ModuleType)
