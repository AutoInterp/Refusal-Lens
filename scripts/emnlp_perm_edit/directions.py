"""Direction construction for per-class JB orthogonalization.

For each JB class C, computes:
    r_jb_C       = mean(h_jb_C[L15, pos=-2]) - mean(h_bare[L15, pos=-2])    [Ball/Wang convention]
    r_jb_C^perp  = r_jb_C - (r_jb_C · r_hat / r_hat · r_hat) · r_hat        [class-specific orthogonal component]
    u_C          = r_jb_C^perp / ||r_jb_C^perp||                            [unit direction]

u_C is the load-bearing quantity for the runtime hook and weight edit.
It is by construction orthogonal to the canonical refusal direction r_hat.
"""
from __future__ import annotations

import torch


def project_out(v: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Return v with the component along `direction` removed.

    `direction` need NOT be unit-norm; the result has no component along it.
    Works on 1-D tensors. For batched inputs, vectorize at the call site.
    """
    coeff = (v @ direction) / (direction @ direction)
    return v - coeff * direction


def compute_u_C(r_hat: torch.Tensor, r_jb_C: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return the unit-norm class-specific orthogonal direction.

    u_C = (r_jb_C - proj_r_hat(r_jb_C)) / ||r_jb_C - proj_r_hat(r_jb_C)||

    Raises:
        ValueError: if r_jb_C is collinear with r_hat (orthogonal component has
                    norm below `eps`), the unit direction is undefined.
    """
    r_jb_perp = project_out(r_jb_C, r_hat)
    norm = r_jb_perp.norm()
    if norm < eps:
        raise ValueError(
            f"orthogonal component of r_jb_C against r_hat has norm {norm:.2e} < eps={eps:.2e}; "
            f"r_jb_C is (nearly) parallel to r_hat. No class-specific axis to orthogonalize against."
        )
    return r_jb_perp / norm
