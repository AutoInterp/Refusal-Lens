"""
Centralized configuration for Refusal-Lens experiments.

All project-wide constants, paths, and hyperparameters live here.
Individual modules may import from this module rather than defining their own.
"""
from __future__ import annotations

from pathlib import Path

# paths (relative to repo root)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = REPO_ROOT / "dataset" / "refusal_direction_dataset"
DATASET_SPLITS_DIR = DATASET_DIR / "splits"
DATASET_PROCESSED_DIR = DATASET_DIR / "processed"
RESULTS_DIR = REPO_ROOT / "data" / "results"
COMPUTED_DIRECTIONS_DIR = RESULTS_DIR / "computed_directions"
CIRCUITS_DIR = RESULTS_DIR / "circuits"
FEATURES_DIR = RESULTS_DIR / "features"
VALIDATION_DIR = RESULTS_DIR / "validation"

# model
MODEL_NAME = "google/gemma-3-4b-it"
DEVICE = "cuda"
DTYPE = "bfloat16"

# sae / transcoder (gemma scope 2)
DEFAULT_SCOPE_REPO = "google/gemma-scope-2-4b-it"
WIDTH_16K = "16k"
WIDTH_262K = "262k"
L0 = "small"

NEURONPEDIA_LAYERS: list[int] = [0, 16, 17, 18, 19]
ANALYSIS_LAYERS: list[int] = [6, 10, 13, 17, 20, 22]
WIDE_LAYERS: list[int] = [16, 18]

# refusal direction comp
REFUSAL_LAYERS: list[int] = list(range(0, 34))  # Gemma-3-4B has 34 layers (0-33)
REFUSAL_POSITION = "last_k"
BEST_REFUSAL_POSITION: int = -2  # Tejas finding: position -2 (model token) gives best separation
MEASUREMENT_LAYER = 32  # Tejas finding: layer 32 gives best separation for Gemma-3-4B-IT
DIRECTION_DTYPE = "float64"  # float64 accumulation prevents precision loss (was bfloat16)

# dataset splits
N_TRAIN = 128
N_EVAL = 64
SEED = 42

# generation
MAX_NEW_TOKENS = 512
N_BASE_PROMPTS = 15
