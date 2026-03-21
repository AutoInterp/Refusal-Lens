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
)

try:
    __version__ = version("refusal_lens")
except Exception:
    __version__ = "0.0.0"
