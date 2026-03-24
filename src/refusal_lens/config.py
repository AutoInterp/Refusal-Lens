"""Centralized configuration for Refusal-Lens.

This module provides a single source of truth for all hyperparameters,
model settings, and path configurations.
"""

from __future__ import annotations

import os
from typing import ClassVar


class Config:
    """Centralized configuration for Refusal-Lens."""

    # ============ MODEL CONFIGURATION ============
    MODEL_NAME: str = "google/gemma-3-4b-it"
    DEVICE: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    TOKENIZER_PAD_TOKEN: str | None = None

    # ============ REFUSAL DIRECTION (Step 1) ============
    REFUSAL_LAYERS: ClassVar[list[int]] = list(range(8, 24))  # Gemma-3-4B has 24 layers
    REFUSAL_POSITION: str = "last_prompt_token"
    MIN_PROMPTS_PER_CLASS: int = 50
    BATCH_SIZE: int = 1

    # ============ ATTRIBUTION (Step 2) ============
    MEASUREMENT_LAYER: int = 20  # Update after Step 1 identifies best layer
    FEATURE_SET: str = "sae"
    ATTRIBUTION_METHOD: str = "logit"  # or "activation"

    # ============ JAILBREAK TESTING (Step 4) ============
    N_BASE_PROMPTS: int = 15
    N_ROLE_VARIATIONS: int = 20
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9

    # ============ DETECTION THRESHOLDS ============
    REFUSAL_STRENGTH_THRESHOLD: float = 0.5
    DETECTION_CONFIDENCE_THRESHOLD: float = 0.7

    # ============ PATHS ============
    DATASET_DIR: str = "dataset/refusal_direction_dataset"
    DATASET_SPLITS_DIR: str = f"{DATASET_DIR}/splits"
    DATASET_HARMFUL_TRAIN: str = f"{DATASET_SPLITS_DIR}/harmful_train.json"
    DATASET_HARMLESS_TRAIN: str = f"{DATASET_SPLITS_DIR}/harmless_train.json"
    DATASET_HARMFUL_VAL: str = f"{DATASET_SPLITS_DIR}/harmful_val.json"
    DATASET_HARMLESS_VAL: str = f"{DATASET_SPLITS_DIR}/harmless_val.json"
    DATASET_HARMFUL_TEST: str = f"{DATASET_SPLITS_DIR}/harmful_test.json"
    DATASET_HARMLESS_TEST: str = f"{DATASET_SPLITS_DIR}/harmless_test.json"

    OUTPUT_DIR: str = "data/results"
    COMPUTED_DIRECTIONS_DIR: str = f"{OUTPUT_DIR}/computed_directions"
    CIRCUITS_DIR: str = f"{OUTPUT_DIR}/circuits"
    JAILBREAK_TESTS_DIR: str = f"{OUTPUT_DIR}/jailbreak_tests"
    NEURONPEDIA_CACHE_DIR: str = f"{OUTPUT_DIR}/neuronpedia_cache"

    # ============ NEURONPEDIA CONFIG ============
    NEURONPEDIA_API_URL: str = "https://www.neuronpedia.org/api"
    NEURONPEDIA_FEATURES_URL: str = "https://www.neuronpedia.org/api/feature"

    # ============ EXPERIMENT CONFIG ============
    RANDOM_SEED: int = 42
    LOG_LEVEL: str = "INFO"
    ENABLE_PROGRESS_BARS: bool = True
    SAVE_INTERMEDIATE_RESULTS: bool = True

    # ============ INFERENCE CONFIG ============
    INFERENCE_MAX_RETRIES: int = 3
    INFERENCE_TIMEOUT: int = 60

    @classmethod
    def to_dict(cls) -> dict[str, object]:
        """Return configuration as dictionary."""
        return {k: v for k, v in vars(cls).items() if k.isupper()}

    @classmethod
    def set_device(cls, device: str) -> None:
        """Set the device for computations."""
        cls.DEVICE = device

    @classmethod
    def set_model_name(cls, model_name: str) -> None:
        """Set the model name."""
        cls.MODEL_NAME = model_name

    @classmethod
    def create_output_dirs(cls) -> None:
        """Create all output directories if they don't exist."""
        dirs = [
            cls.OUTPUT_DIR,
            cls.COMPUTED_DIRECTIONS_DIR,
            cls.CIRCUITS_DIR,
            cls.JAILBREAK_TESTS_DIR,
            cls.NEURONPEDIA_CACHE_DIR,
        ]
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)


# Default instance for easy import
config = Config()


__all__ = [
    "Config",
    "config",
]
