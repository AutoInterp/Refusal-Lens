"""
JumpRelu Sparse Autoencoder / Transcoder (Gemma Scope 2).

Provides the SAE architecture, loading utilities, and width-mismatch diagnostic.
Requires optional deps: torch, safetensors, huggingface_hub.
"""

import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    HAS_TORCH = True
except:
    HAS_TORCH = False

def _require_torch() -> None:
    if not HAS_TORCH:
        msg = (
            "torch, safetensors, and huggingface_hub are required for SAE Loading."
            "Install with: pip install refusal-lens[steering]"
        )
        raise ImportError(msg)

# some constants I used for the RP experiments
DEFAULT_SCOPE_REPO = "google/gemma-scope-2-4b-it"

NEURONPEDIA_LAYERS: list[int] = [0, 16, 17, 18, 19]
ANALYSIS_LAYERS: list[int] = [6, 10, 13, 17, 20, 22]

WIDTH_16K = "16k"
WIDTH_262K = "262k"
L0 = "small"

WIDE_LAYERS: list[int] = [16, 18]

