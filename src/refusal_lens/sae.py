"""
JumpRelu Sparse Autoencoder / Transcoder (Gemma Scope 2).

Provides the SAE architecture, loading utilities, and width-mismatch diagnostic.
Requires optional deps: torch, safetensors, huggingface_hub.
"""
from __future__ import annotations

import logging
from typing import Any

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

# utilities functions for loading
def load_sae(
        category: str,
        layer: int,
        width: str = WIDTH_16K,
        l0: str = L0,
        *,
        affine: bool = False,
        repo: str = DEFAULT_SCOPE_REPO,
        device: str | None = None,
) -> JumpReLUSAE:
    """
    Download and load a single SAE/Transcoder from Huggingface.

    Args:
        category: SAE category ('resid_post', 'transcoder', etc).
        layer: Model Layer index.
        width: Dictionary width string (e.g., '16k', '262k').
        l0: Sparsity level string (e.g., 'small').
        affine: Whether checkpoint has an affine skip connection.
        repo: Huggingface Repo ID.
        device: Device to plave the SAE on. Defaults to 'cuda' if available.
    
    Returns:
        A ``JumpReLUSAE`` in eval mode on the requested device.
    """
    _require_torch()
    affine_str = "_affine" if affine else ""
    filename = (
        f"{category}/layer_{layer}_width_{width}_l0_{l0}"
        f"{affine_str}/params.safetensors"
    )
    logger.info("Downloading: %s", filename)

    path = hf_hub_download(repo_id=repo, filename=filename)
    params = load_file(path)

    d_model, d_sae = params["w_enc"].shape
    sae = JumpReLUSAE(d_model, d_sae, affine_skip_connection=affine)
    sae.load_state_dict(params)

    if device is None:
        device = "cuda" if torch.cuda_is_available() else "cpu"
    return sae.to(device).eval()

def load_sae_flexible(
        category: str,
        layer: int,
        width: str = WIDTH_16K,
        l0: str = L0,
        *,
        affine: bool = False,
        repo: str = DEFAULT_SCOPE_REPO,
        device: str | None = None,
) -> JumpReLUSAE:
    """
    Try Multiple category/affine combinations to load an SAE.

    HuggingFace Gemma Scope repos sometimes use ``transcoder_all``
    instead of ``transcoder``, or have checkpoints with and without
    affine skip connections. This wrapper tries all combinations.

    Args:
        Same as :func:`load_sae`.
    
    Returns:
        The first successfully loaded ``JumpReLUSAE``.
    
    Raises:
        RuntimeError: If no combination succeeds.
    """
    _require_torch()
    categories = [category, f"{category}_all"]
    affine_options = [affine] if not affine else [True, False]

    for cat in categories:
        for aff in affine_options:
            try:
                return load_sae(
                    cat, layer, width, l0,
                    affine=aff, repo=repo, device=device,
                )
            except Exception:
                continue
    
    msg = (
        f"Could not load {category} layer {layer} "
        f"width {width} l0 {l0}"
    )
    raise RuntimeError(msg)

def load_sae_set(
        category: str,
        layers: list[int],
        width: str = WIDTH_16K,
        l0: str = L0,
        *,
        affine: bool = False,
        repo: str = DEFAULT_SCOPE_REPO,
        device: str | None = None,
) -> dict[int, JumpReLUSAE]:
    """
    Load SAEs/Transcoders for multiple layers, skipping failures.

    Args:
        category: SAE category (e.g., 'resid_post', 'transcoder').
        layers: List of layer indices to load.
        width: Dictionary width string.
        l0: Sparsity level string.
        affine: Whether to request affine skip connection.
        repo: HuggingFace repo ID.
        device: Device for the loaded SAEs.
    
    Returns:
        Dict mapping layer index -> loaded JumpReLUSAE.
        Layers that fail to load are omitted (with a warning logged).
    """
    _require_torch()
    loaded: dict[int, JumpReLUSAE] = {}
    for layer in layers:
        try:
            loaded[layer] = load_sae_flexible(
                category, layer, width, l0,
                affine=affine, repo=repo, device=device,
            )
            logger.info("Loaded %s layer %d (%s)", category, layer, width)
        except Exception:
            logger.warning(
                "Failed to load %s layer %d (%s)", category, layer, width
            )
    return loaded

def analyze_width_mismatch(
      feat_idx_16k: int,
      tc_16k: JumpReLUSAE,
      tc_wide: JumpReLUSAE,
      k: int = 10,
  ) -> dict[str, Any]:
      """Measure decoder-direction overlap between a 16k and wide transcoder.

      For a given 16k feature, finds the top-k most similar 262k features
      by cosine similarity of their decoder directions. High overlap with
      multiple 262k features indicates superposition risk.

      Args:
          feat_idx_16k: Feature index in the 16k transcoder.
          tc_16k: The 16k-width transcoder.
          tc_wide: The wide (e.g. 262k) transcoder.
          k: Number of top matches to return.

      Returns:
          Dict with keys:
              - ``top_indices``: list of matched 262k feature indices
              - ``top_cosines``: list of cosine similarities
              - ``n_high_overlap``: count of matches with |cos| > 0.3
      """
      _require_torch()
      # decoder direction for the 16k feature
      dec_16k = tc_16k.w_dec[feat_idx_16k].float()
      dec_16k = dec_16k / dec_16k.norm()

      # All decoder directions from the wide transcoder
      dec_wide = tc_wide.w_dec.float()
      norms = dec_wide.norm(dim=1, keepdim=True).clamp(min=1e-8)
      dec_wide_normed = dec_wide / norms

      cos_sims = dec_wide_normed @ dec_16k

      topk = torch.topk(cos_sims.abs(), k=k)

      return {
          "top_indices": topk.indices.tolist(),
          "top_cosines": [round(v, 4) for v in topk.values.tolist()],
          "n_high_overlap": int((topk.values > 0.3).sum().item()),
      }