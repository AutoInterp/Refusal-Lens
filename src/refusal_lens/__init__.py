"""
Refusal-Lens: A package for analyzing refusal behavior in neural network models.

This package provides tools for:
- Classifying model responses into refusal categories
- Generating standardized prompt templates for testing
- Detecting and analyzing refusal behavior
- Running experiments with progress tracking
"""

from __future__ import annotations

from importlib.metadata import version

# Experiment Runner
from .experiment_runner import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
)

# Model loading
from .model_loader import (
    ModelConfig,
    generate_text,
    get_device_info,
    load_model,
    tokenize_prompt,
    display_topk,
)

# Prompt Templates
from .prompt_template import (
    PromptCategory,
    PromptTemplate,
    PromptTemplateLibrary,
)

# Refusal Classification
from .refusal_classifier import (
    ClassificationResult,
    RefusalCategory,
    RefusalClassifier,
)

# Refusal Detection
from .refusal_detector import (
    DetectionResult,
    DetectionStatus,
    RefusalDetector,
)

# SAE / Transcoder
from .sae import (
    JumpReLUSAE,
    load_sae,
    load_sae_flexible,
    load_sae_set,
    analyze_width_mismatch,
    DEFAULT_SCOPE_REPO,
    NEURONPEDIA_LAYERS,
    ANALYSIS_LAYERS,
    WIDTH_16K,
    WIDTH_262K,
    WIDE_LAYERS,
)

# Data loading
from .data_loader import (
    ContrastiveDataset,
    ContrastiveSplit,
    Instruction,
    create_contrastive_dataset,
    list_available_processed,
    list_available_splits,
    load_processed,
    load_split,
)

__all__ = (
    "ClassificationResult",
    "DetectionResult",
    # Refusal Detection
    "DetectionStatus",
    # Experiment Runner
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentRunner",
    # Prompt Templates
    "PromptCategory",
    "PromptTemplate",
    "PromptTemplateLibrary",
    # Refusal Classification
    "RefusalCategory",
    "RefusalClassifier",
    "RefusalDetector",
    # Version
    "__version__",
    # Model loading
    "ModelConfig",
    "generate_text",
    "get_device_info",
    "load_model",
    "tokenize_prompt",
    "display_topk",
    # SAE / Transcoder
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
    # Data loading
    "ContrastiveDataset",
    "ContrastiveSplit",
    "Instruction",
    "create_contrastive_dataset",
    "list_available_processed",
    "list_available_splits",
    "load_processed",
    "load_split",
)

try:
    __version__ = version("refusal_lens")
except Exception:
    __version__ = "0.0.0"
