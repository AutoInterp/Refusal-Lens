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

class JumpReLUSAE(nn.module):
    """
    JumpRelu SAE / Transcoder from Gemma Scope 2.

    Arch:
        encode: x @ w_enc + b_enc -> JumpReLU(threshold) -> sparse features
        decode: features @ w_dec + b_dec -> reconstruction
        forward: encode -> decode (+ optional affine skip connection)
    """

    def __init__(self, d_in: int, d_sae: int, *, affine_skip_connection: bool = False) -> None:
        _require_torch()
        super().__init__()
        self.w_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.w_dec = nn.Parameter(torch.zeros(d_sae, d_in))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        if affine_skip_connection:
            self._affine_skip = nn.Parameter(torch.zeros(d_in, d_in))
        else:
            self._affine_skip = None
        
    # forward methods
    def encode(self, input_acts: torch.Tensor) -> torch.Tensor:
        """Encode input activations to sparse feature activations."""
        pre_acts = input_acts @ self.w_enc + self.b_enc
        mask = pre_acts > self.threshold
        return mask * F.relu(pre_acts)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to input space."""
        return acts @ self.w_dec + self.b_dec
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode -> decode (+ affine skip if present)."""
        acts = self.encode(x)
        recon = self.decode(acts)
        if self._affine_skip is not None:
            return recon + x @ self._affine_skip
        return recon
    
    # some properties
    @property
    def d_sae(self) -> int:
        """Dictionary size (number of SAE features)."""
        return self.w_enc.shape[1]
    
    @property
    def d_in(self) -> int:
        """Input dimensionality (model hidden size)."""
        return self.w_enc.shape[0]
    
    @property
    def affine_skip_connection(self) -> torch.nn.Parameter | None:
        """The affine skip-connection parameter, or None"""
        return self._affine_skip
