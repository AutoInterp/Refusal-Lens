"""Apply Georg's mean-subtraction control to the pairwise u_C cosines.

Background: 00_compute_directions.py computed raw cosines between per-class
u_C vectors (orthogonal-to-r_hat components of per-class JB displacements)
and found them surprisingly high (+0.67 to +0.89 for 4 of 10 pairs).

Georg's question: are these inflated by all-ones-direction bias / anisotropy
of the residual stream? If we subtract the scalar mean of each vector before
computing cosine (Pearson-style), does the cosine drop?

Method (same as Phase 0 sub-experiment 0c, applied to u_C-u_C' instead of
r_jb_C-r_hat):
  raw_cos(u, v)     = u·v / (||u||·||v||)
  pearson_cos(u, v) = (u - mean_scalar(u)·1) · (v - mean_scalar(v)·1)
                      / (||u_centered||·||v_centered||)
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
CLASSES = ["fiction", "roleplay", "analytical", "completion", "cognitive_reframe"]


def pearson_cosine(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Cosine of mean-subtracted vectors (i.e., Pearson correlation coefficient).

    Returns 0.0 if either centered vector has zero norm.
    """
    v1c = v1.float() - v1.float().mean()
    v2c = v2.float() - v2.float().mean()
    n1, n2 = v1c.norm(), v2c.norm()
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return ((v1c @ v2c) / (n1 * n2)).item()


def main():
    in_path = REPO / "data/results/emnlp_perm_edit/phase1_runtime_hook/directions.pt"
    print(f"[uc-robustness] loading {in_path}")
    u_C_by_class = torch.load(in_path, weights_only=False)

    # Scalar mean per u_C
    print(f"\n[uc-robustness] scalar mean (proj onto all-ones direction):")
    for cls in CLASSES:
        u = u_C_by_class[cls].float()
        scalar_mean = u.mean().item()
        # Fraction of ||u|| explained by the all-ones component:
        # all-ones unit vec = 1/sqrt(d). Projection onto it: scalar_mean * sqrt(d)
        d = u.shape[0]
        all_ones_proj_magnitude = scalar_mean * (d ** 0.5)
        all_ones_frac = abs(all_ones_proj_magnitude) / u.norm().item()
        print(f"  {cls:22s}  mean={scalar_mean:+.4e}  |proj_onto_ones|/||u||={all_ones_frac:.4f}")

    print(f"\n[uc-robustness] pairwise: raw cos vs Pearson cos (mean-subtracted)")
    print(f"  {'pair':30s}  {'raw_cos':>10s}  {'pearson_cos':>12s}  {'delta':>10s}")
    results = {}
    for i, c1 in enumerate(CLASSES):
        for c2 in CLASSES[i+1:]:
            v1 = u_C_by_class[c1]
            v2 = u_C_by_class[c2]
            raw = torch.nn.functional.cosine_similarity(
                v1.unsqueeze(0), v2.unsqueeze(0)).item()
            pearson = pearson_cosine(v1, v2)
            delta = raw - pearson
            key = f"{c1}__{c2}"
            results[key] = {"raw_cos": raw, "pearson_cos": pearson, "delta": delta}
            print(f"  {c1[:14]:14s} x {c2[:14]:14s}  {raw:+10.4f}  {pearson:+12.4f}  {delta:+10.4f}")

    out_path = REPO / "data/results/emnlp_perm_edit/phase1_runtime_hook/uc_pairwise_robustness.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[uc-robustness] saved {out_path}")

    # Quick verdict
    max_delta = max(abs(r["delta"]) for r in results.values())
    print(f"\n[uc-robustness] max |raw - pearson| across 10 pairs = {max_delta:.4f}")
    if max_delta < 0.05:
        verdict = "PASS — all-ones-direction bias is NOT inflating raw cosines; the high u_C cosines are real geometric overlap."
    elif max_delta < 0.20:
        verdict = "PARTIAL — some all-ones-direction bias present but raw cosines are still mostly real geometric overlap."
    else:
        verdict = "FAIL — substantial all-ones-direction bias inflating raw cosines; the u_C 'overlap' may be an artifact, not real shared mechanism."
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()
