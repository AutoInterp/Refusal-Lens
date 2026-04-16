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

# Supernode Analysis (Step 3)
from .supernode_analyzer import (
    NeuronPattern,
    SupernodeAnalyzer,
    SupernodeData,
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
    "ClassificationResult",
    "Config",
    "DetectionResult",
    "DetectionStatus",
    "JailbreakTestResult",
    "NeuronPattern",
    "PromptCategory",
    "PromptTemplate",
    "PromptTemplateLibrary",
    "RefusalCategory",
    "RefusalClassifier",
    "RefusalDetector",
    "RefusalDirectionComputer",
    "RefusalDirectionResult",
    "SupernodeAnalyzer",
    "SupernodeData",
    "SystematicVariationGenerator",
    "__version__",
    "analyze_circuit",
    "attribute_to_direction",
    "config",
    "find_best_layer",
    "get_device",
    "load_dataset_split",
    "load_directions",
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
