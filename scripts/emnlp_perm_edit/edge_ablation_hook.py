"""Edge-ablation hook factory for Phase 0 sub-experiment 0b-simple.

Subtracts a scalar `delta` along the r_hat_unit direction from the residual at
every position. After the hook, the r_hat-projection of the residual at every
(batch, seq) is reduced by `delta`. Used to simulate "ablating an edge-type
bucket's contribution to direct_dot at L15 pos=-2" by directly modifying
the residual stream's r_hat component by the predicted amount.

Math:
    h_new = h - (delta / ||r_hat||^2) * r_hat
    h_new . r_hat = h . r_hat - delta * (r_hat . r_hat) / ||r_hat||^2 = h . r_hat - delta
"""
from __future__ import annotations

import torch


def make_scalar_rhat_subtraction_hook(r_hat: torch.Tensor, delta: float):
    """Return a forward_hook that subtracts `delta` from h . r_hat at every position.

    Args:
        r_hat: 1-D direction tensor (need NOT be unit-norm).
        delta: scalar amount to subtract from the r_hat-projection.

    The hook handles both tuple outputs (Gemma layer modules return tuples)
    and plain-tensor outputs (sublayer norms). Casts r_hat to the output dtype
    at hook time so we don't force fp32 math inside a bf16 model.
    """
    r_hat = r_hat.float()
    r_hat_norm_sq = (r_hat @ r_hat).item()

    def hook_fn(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        r_cast = r_hat.to(dtype=h.dtype, device=h.device)
        # Subtract (delta / ||r_hat||^2) * r_hat from every position
        coeff = delta / r_hat_norm_sq
        h_new = h - coeff * r_cast
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new
    return hook_fn
