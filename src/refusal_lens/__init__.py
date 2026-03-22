"""
Refusal-Lens: A package for analyzing refusal behavior in neural network models.

This package provides tools for:
- Computing refusal directions (Step 1)
- Analyzing attribution circuits (Step 2)
- Classifying model responses into refusal categories
- Generating standardized prompt templates for testing
- Detecting and analyzing refusal behavior
- Running experiments with progress tracking
"""

from __future__ import annotations

from importlib.metadata import version

# Circuit Analysis (Steps 1 & 2)
from .circuits import (
    AttributionGraph,
    AttributionResult,
    RefusalDirectionComputer,
    RefusalDirectionResult,
    analyze_circuit,
    attribute_to_direction,
    find_best_layer,
    load_directions,
    save_attributions,
    save_directions,
)

# Configuration
from .config import Config, config

# Experiment Runner
from .experiment_runner import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
)

# Jailbreaks (Step 4)
from .jailbreaks import (
    JailbreakTestResult,
    SystematicVariationGenerator,
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

# Utils (shared utilities)
from .utils import (
    get_device,
    load_dataset_split,
    load_json,
    load_prompts_from_json,
    save_json,
    set_seed,
)

__all__ = (
    "AttributionGraph",
    "AttributionResult",
    # Refusal Classification
    "ClassificationResult",
    # Config
    "Config",
    # Refusal Detection
    "DetectionResult",
    "DetectionStatus",
    # Experiment Runner
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentRunner",
    "JailbreakTestResult",
    # Prompt Templates
    "PromptCategory",
    "PromptTemplate",
    "PromptTemplateLibrary",
    "RefusalCategory",
    "RefusalClassifier",
    "RefusalDetector",
    # Circuits
    "RefusalDirectionComputer",
    "RefusalDirectionResult",
    # Jailbreaks
    "SystematicVariationGenerator",
    # Version
    "__version__",
    "analyze_circuit",
    "attribute_to_direction",
    "config",
    "find_best_layer",
    "get_device",
    "load_dataset_split",
    "load_directions",
    # Utils
    "load_json",
    "load_prompts_from_json",
    "save_attributions",
    "save_directions",
    "save_json",
    "set_seed",
)

try:
    __version__ = version("refusal_lens")
except Exception:
    __version__ = "0.0.0"
