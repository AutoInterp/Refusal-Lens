"""Circuit analysis modules for Refusal-Lens.

This module provides tools for:
- Computing refusal directions (Step 1 of research plan)
- Analyzing attribution circuits (Step 2 of research plan)
"""

from __future__ import annotations

from .attribution import (
    AttributionGraph,
    AttributionResult,
    analyze_circuit,
    attribute_to_direction,
    save_attributions,
)
from .refusal_direction import (
    RefusalDirectionComputer,
    RefusalDirectionResult,
    find_best_layer,
    load_directions,
    save_directions,
)

__all__ = [
    # Attribution (Step 2)
    "AttributionGraph",
    "AttributionResult",
    # Refusal Direction (Step 1)
    "RefusalDirectionComputer",
    "RefusalDirectionResult",
    "analyze_circuit",
    "attribute_to_direction",
    "find_best_layer",
    "load_directions",
    "save_attributions",
    "save_directions",
]
