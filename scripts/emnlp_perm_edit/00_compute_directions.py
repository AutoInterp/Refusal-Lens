"""Compute per-class u_C directions for Phase 1 runtime-hook experiments.

For each of 5 JB classes, computes:
    r_jb_C       = mean(h_jb_C[L15, pos=-2]) - mean(h_bare[L15, pos=-2])
    u_C          = compute_u_C(r_hat, r_jb_C)    # unit, orthogonal to r_hat

Records pre-intervention diagnostics:
    ||u_C||                                       (should be 1.0)
    cos(r_hat, u_C)                               (should be 0.0 by construction)
    ||r_jb_C^perp|| / ||r_hat||                   (expected 0.24-0.38 per REPORT §5.5.2)
    pairwise cos(u_C, u_C')                       (load-bearing dissociation diagnostic)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from directions import compute_u_C  # noqa: E402


LAYER = 15
POS_IDX = 2                                     # position -2 in the [-5, -3, -2] saved tensor
CLASSES = ["fiction", "roleplay", "analytical", "completion", "cognitive_reframe"]


def parse_args():
    p = argparse.ArgumentParser(description="Compute per-class u_C directions for Phase 1")
    repo = Path(__file__).resolve().parents[2]
    p.add_argument("--run-dir", type=Path,
                   default=repo / "data/results/pipeline_runs/run_20260430_023247",
                   help="Run directory containing 01_direction/ and 02b_stats/")
    p.add_argument("--out-dir", type=Path,
                   default=repo / "data/results/emnlp_perm_edit/phase1_runtime_hook",
                   help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[directions] loading r_hat[L={LAYER}] from {args.run_dir}/01_direction/")
    r_dict = torch.load(args.run_dir / "01_direction/unnormalized_r.pt", weights_only=False)
    r_hat = r_dict[LAYER].float().cpu()
    r_hat_norm = r_hat.norm().item()
    print(f"  ||r_hat|| = {r_hat_norm:.2f}")

    print(f"[directions] loading residuals from {args.run_dir}/02b_stats/")
    residuals = torch.load(args.run_dir / "02b_stats/residuals_L15_per_cond.pt", weights_only=False)
    h_bare = residuals["bare"][:, POS_IDX, :].float().mean(dim=0)
    print(f"  ||mean(h_bare)|| = {h_bare.norm().item():.2f}")

    u_C_by_class: dict[str, torch.Tensor] = {}
    diagnostics: dict = {
        "metadata": {
            "measurement_layer": LAYER,
            "measurement_position": -2,
            "r_hat_norm": r_hat_norm,
            "convention": "r_jb_C = mean(h_jb_C) - mean(h_bare); u_C = orthogonal-to-r_hat unit direction",
            "construction_dataset": "run_20260430_023247 controlled 50-prompt set",
        },
        "per_class": {},
        "pairwise_cosines": {},
    }

    for cls in CLASSES:
        h_jb = residuals[f"jb_{cls}"][:, POS_IDX, :].float().mean(dim=0)
        r_jb_C = h_jb - h_bare
        u_C = compute_u_C(r_hat, r_jb_C)

        r_jb_perp_norm = (r_jb_C - (r_jb_C @ r_hat) / (r_hat @ r_hat) * r_hat).norm().item()
        cos_r_hat_u_C = torch.nn.functional.cosine_similarity(
            u_C.unsqueeze(0), r_hat.unsqueeze(0)).item()

        diagnostics["per_class"][cls] = {
            "r_jb_C_norm": r_jb_C.norm().item(),
            "r_jb_C_norm_over_r_hat": r_jb_C.norm().item() / r_hat_norm,
            "r_jb_C_perp_norm": r_jb_perp_norm,
            "r_jb_C_perp_norm_over_r_hat": r_jb_perp_norm / r_hat_norm,
            "u_C_norm": u_C.norm().item(),
            "cos_r_hat_u_C": cos_r_hat_u_C,
        }
        u_C_by_class[cls] = u_C
        print(f"  {cls:22s}  ||r_jb_perp||/||r_hat||={r_jb_perp_norm/r_hat_norm:.3f}  "
              f"cos(r_hat, u_C)={cos_r_hat_u_C:+.2e}")

    print("\n[directions] pairwise cos(u_C, u_C'):")
    for i, c1 in enumerate(CLASSES):
        for c2 in CLASSES[i+1:]:
            cos = torch.nn.functional.cosine_similarity(
                u_C_by_class[c1].unsqueeze(0), u_C_by_class[c2].unsqueeze(0)).item()
            key = f"{c1}__{c2}"
            diagnostics["pairwise_cosines"][key] = cos
            print(f"  cos(u_{c1[:8]:8s}, u_{c2[:8]:8s}) = {cos:+.4f}")

    torch.save(u_C_by_class, args.out_dir / "directions.pt")
    (args.out_dir / "direction_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(f"\n[directions] saved directions.pt ({len(CLASSES)} tensors) "
          f"and direction_diagnostics.json to {args.out_dir}/")


if __name__ == "__main__":
    main()
