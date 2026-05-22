"""Edge-ablation hook factory for Phase 0 sub-experiment 0b-simple.

Subtracts a scalar `delta` along the r_hat_unit direction from the residual.
After the hook, the r_hat-projection of the residual at the targeted positions
is reduced by `delta`. Used to simulate "ablating an edge-type bucket's
contribution to direct_dot at L15 pos=-2" by directly modifying the residual
stream's r_hat component by the predicted amount.

Math (at each targeted position):
    h_new = h - (delta / ||r_hat||^2) * r_hat
    h_new . r_hat = h . r_hat - delta * (r_hat . r_hat) / ||r_hat||^2 = h . r_hat - delta
"""
from __future__ import annotations

import torch


def make_scalar_rhat_subtraction_hook(
    r_hat: torch.Tensor,
    delta: float,
    position_mode: str = "all",
    target_position: int = -2,
):
    """Return a forward_hook that subtracts `delta` from h . r_hat.

    Args:
        r_hat: 1-D direction tensor (need NOT be unit-norm).
        delta: scalar amount to subtract from the r_hat-projection.
        position_mode:
            "all" (default): subtract at every (batch, seq) position on every
                forward pass — matches Arditi-style intervention and the
                original Phase 0 0b/0d/0e behavior.
            "last_prompt_only": subtract only at the prompt encoding pass's
                seq position `target_position`. Generation-step forward passes
                (seq_len == 1) are skipped.
        target_position: which seq position to edit in `last_prompt_only` mode.
            Default -2 for Gemma (the position whose representation predicts
            the first generated token under Gemma's chat template). Set -1 for
            models like Qwen where the direction was constructed at pos=-1.

    The hook handles both tuple outputs (Gemma layer modules return tuples)
    and plain-tensor outputs. Casts r_hat to the output dtype at hook time
    so we don't force fp32 math inside a bf16 model.
    """
    if position_mode not in ("all", "last_prompt_only"):
        raise ValueError(f"position_mode must be 'all' or 'last_prompt_only', got {position_mode!r}")

    r_hat = r_hat.float()
    r_hat_norm_sq = (r_hat @ r_hat).item()

    def hook_fn(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        r_cast = r_hat.to(dtype=h.dtype, device=h.device)
        coeff = delta / r_hat_norm_sq

        if position_mode == "all":
            h_new = h - coeff * r_cast
        else:  # last_prompt_only
            seq_len = h.shape[1]
            if seq_len > 1:
                # Prompt encoding pass: edit only `target_position`
                h_new = h.clone()
                h_new[:, target_position, :] = h[:, target_position, :] - coeff * r_cast
            else:
                # Generation step (single new token, seq_len == 1): no edit
                h_new = h

        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new
    return hook_fn
