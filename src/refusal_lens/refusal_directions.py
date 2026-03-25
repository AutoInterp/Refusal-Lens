"""
Refusal direction computation (Arditi et al. 2024)

Compute \hat{r} = E[x|harmful] - E[x|harmless] via difference-in-means or PCA.
This direction is the attribution target for circuit-tracer integration.

Requires optional deps: torch, transformers.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F

    HAD_TORCH = True
except ImportError:
    HAS_TORCH = False

def _require_torch() -> None:
    if not HAS_TORCH:
        msg = (
            "torch is required for refusal direction computation. "
            "Install with: pip install refusal-lens[steering]"
        )
        raise ImportError(msg)
    
