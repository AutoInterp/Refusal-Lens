"""Phase 0 — Sub-experiment 0c: direction-alignment robustness audit.

Tests whether the +0.72 to +0.94 cosine between r_jb_C (toward jailbreak) and
-r_hat (toward harmless) reported in REPORT § 5.5.2 is a robust geometric fact
or partly inflated by high-dimensional residual stream anisotropy / all-ones-
direction bias / class-mean averaging artifacts.

Three diagnostics per class:
- 0c.1 per-prompt cosine: compute the JB displacement for each individual
  prompt and report per-class mean +/- std of per-prompt cosines.
- 0c.2 random-direction baseline: sample N random unit vectors and compare
  cos(r_jb_C, r_hat) against the 95th percentile of cos(r_jb_C, random_dir).
- 0c.3 Pearson-style mean-subtraction: compute cosine after subtracting each
  direction's scalar mean (all-ones-direction bias control).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]

LAYER = 15
POS_IDX = 2  # index of pos=-2 in the saved [-5, -3, -2] tensor
CLASSES = ["fiction", "roleplay", "analytical", "completion", "cognitive_reframe"]


def compute_per_prompt_cosines(h_jb: torch.Tensor, h_bare: torch.Tensor,
                               r_hat: torch.Tensor) -> dict:
    """For each prompt p, compute cos(h_jb[p] - h_bare[p], r_hat).

    Returns {mean_cos, std_cos, gt_half_rate, per_prompt: [...]}.
    """
    delta = h_jb.float() - h_bare.float()  # (n_prompts, d_model)
    per_prompt_cos = torch.nn.functional.cosine_similarity(
        delta, r_hat.float().unsqueeze(0).expand_as(delta), dim=-1
    )  # (n_prompts,)
    mean_cos = per_prompt_cos.mean().item()
    std_cos = per_prompt_cos.std().item() if per_prompt_cos.numel() > 1 else 0.0
    gt_half = (per_prompt_cos.abs() > 0.5).float().mean().item()
    return {
        "mean_cos": mean_cos,
        "std_cos": std_cos,
        "gt_half_rate": gt_half,
        "per_prompt_cosines": per_prompt_cos.tolist(),
    }


def random_baseline_cosine_stats(r_jb_C: torch.Tensor, r_hat: torch.Tensor,
                                 n_random: int = 1000, seed: int = 42) -> dict:
    """Sample n_random unit vectors and compare cos(r_jb_C, r_hat) to their cosines."""
    g = torch.Generator().manual_seed(seed)
    d = r_jb_C.shape[0]
    rand = torch.randn(n_random, d, generator=g)
    rand = rand / rand.norm(dim=1, keepdim=True)
    rand_cos = torch.nn.functional.cosine_similarity(
        r_jb_C.float().unsqueeze(0), rand, dim=-1
    )  # (n_random,)
    real_cos = torch.nn.functional.cosine_similarity(
        r_jb_C.float().unsqueeze(0), r_hat.float().unsqueeze(0), dim=-1
    ).item()
    p95 = rand_cos.abs().quantile(0.95).item()
    # Rank: how many random cosines have |cos| larger than real cos
    rank = int((rand_cos.abs() > abs(real_cos)).sum().item())
    return {
        "n_random": n_random,
        "p95_abs_random_cos": p95,
        "real_cos_with_r_hat": real_cos,
        "rank_of_real_in_random": rank,
        "real_passes_p95": abs(real_cos) > p95,
    }


def pearson_cosine(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Cosine of mean-subtracted vectors (i.e., Pearson correlation coefficient).

    Returns 0.0 if either centered vector has zero norm (e.g., constant vectors).
    """
    v1c = v1.float() - v1.float().mean()
    v2c = v2.float() - v2.float().mean()
    n1 = v1c.norm()
    n2 = v2c.norm()
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return ((v1c @ v2c) / (n1 * n2)).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path,
                   default=REPO / "data/results/pipeline_runs/run_20260430_023247")
    p.add_argument("--out-dir", type=Path,
                   default=REPO / "data/results/emnlp_perm_edit/phase0_controllability")
    p.add_argument("--n-random", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"[0c] loading r_hat[L{LAYER}]")
    r_dict = torch.load(args.run_dir / "01_direction/unnormalized_r.pt", weights_only=False)
    r_hat = r_dict[LAYER].float()
    print(f"  ||r_hat|| = {r_hat.norm().item():.2f}")

    print(f"[0c] loading residuals from {args.run_dir}/02b_stats/")
    R = torch.load(args.run_dir / "02b_stats/residuals_L15_per_cond.pt", weights_only=False)
    h_bare = R["bare"][:, POS_IDX, :].float()  # (50, 2560)

    out = {
        "metadata": {
            "n_random": args.n_random, "seed": args.seed,
            "r_hat_norm": r_hat.norm().item(),
            "n_prompts": h_bare.shape[0],
            "convention": "r_jb_C = mean(h_jb_C) - mean(h_bare); Ball/Wang (points toward JB)",
        },
        "per_class": {},
    }

    print(f"\n[0c] per-class diagnostics:")
    print(f"  {'class':22s}  {'class_cos':>10s}  {'pp_mean':>10s}  {'pp_std':>10s}  "
          f"{'p95_rand':>10s}  {'pearson':>10s}  {'rank/N':>10s}")

    for cls in CLASSES:
        h_jb = R[f"jb_{cls}"][:, POS_IDX, :].float()
        r_jb_C = h_jb.mean(0) - h_bare.mean(0)
        class_mean_cos = torch.nn.functional.cosine_similarity(
            r_jb_C.unsqueeze(0), r_hat.unsqueeze(0)).item()

        per_prompt = compute_per_prompt_cosines(h_jb, h_bare, r_hat)
        rand_stats = random_baseline_cosine_stats(
            r_jb_C, r_hat, n_random=args.n_random, seed=args.seed)
        pcos = pearson_cosine(r_jb_C, r_hat)

        out["per_class"][cls] = {
            "class_mean_cos_with_r_hat": class_mean_cos,
            "class_mean_cos_with_neg_r_hat": -class_mean_cos,
            "per_prompt": per_prompt,
            "random_baseline": rand_stats,
            "pearson_cos": pcos,
            "delta_raw_minus_pearson": abs(class_mean_cos - pcos),
        }

        print(f"  {cls:22s}  {class_mean_cos:+10.4f}  {per_prompt['mean_cos']:+10.4f}  "
              f"{per_prompt['std_cos']:10.4f}  {rand_stats['p95_abs_random_cos']:10.4f}  "
              f"{pcos:+10.4f}  {rand_stats['rank_of_real_in_random']:4d}/{args.n_random:4d}")

    # Phase 0 H0-5 verdict per class
    print(f"\n[0c] H0-5 per-class verdicts:")
    n_pass_all = 0
    for cls in CLASSES:
        blob = out["per_class"][cls]
        c1_pass = abs(blob["per_prompt"]["mean_cos"] - blob["class_mean_cos_with_r_hat"]) < 0.10
        c2_pass = blob["random_baseline"]["real_passes_p95"]
        c3_pass = blob["delta_raw_minus_pearson"] < 0.10
        full_pass = c1_pass and c2_pass and c3_pass
        if full_pass:
            n_pass_all += 1
        print(f"  {cls:22s}  per_prompt={'PASS' if c1_pass else 'FAIL'}  "
              f"random={'PASS' if c2_pass else 'FAIL'}  "
              f"pearson={'PASS' if c3_pass else 'FAIL'}  "
              f"all={'PASS' if full_pass else 'FAIL'}")
    out["h0_5_summary"] = {
        "n_classes_passing_all": n_pass_all,
        "n_classes_total": len(CLASSES),
        "overall_pass": n_pass_all >= 4,
    }
    print(f"\nH0-5 overall: {n_pass_all}/{len(CLASSES)} classes pass all 3 controls "
          f"-> {'PASS' if out['h0_5_summary']['overall_pass'] else 'FAIL'}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "direction_robustness.json").write_text(json.dumps(out, indent=2))
    print(f"\n[0c] wrote direction_robustness.json")


if __name__ == "__main__":
    main()
