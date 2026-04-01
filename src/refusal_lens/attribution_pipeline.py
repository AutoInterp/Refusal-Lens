"""
End-to-end attribution pipeline for refusal direction analysis.

Orchestrates the full workflow: compute attribution graphs targeting the
refusal projection R, compare graphs contrastively across conditions,
aggregate features across multiple prompts, and export for visualization.

Builds on:
- ``clt.py``: ``attribute_to_refusal``, ``extract_top_features``, ``prune_refusal_graph``
- ``attribution.py``: ``attribute_to_direction``, ``validate_measurement_point``
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config

logger = logging.getLogger(__name__)

HAS_CIRCUIT_TRACER = False
try:
    import torch
    from circuit_tracer import Graph

    HAS_CIRCUIT_TRACER = True
except ImportError:
    pass


def _require_circuit_tracer() -> None:
    if not HAS_CIRCUIT_TRACER:
        raise ImportError(
            "circuit-tracer is required. "
            "Install via: pip install git+https://github.com/safety-research/circuit-tracer.git"
        )


@dataclass
class RefusalGraph:
    """
    An attribution graph with refusal-direction metadata.

    Wraps a circuit-tracer ``Graph`` with the prompt, measurement point,
    and extracted top features so results are self-describing.
    """
    pass

