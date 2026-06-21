"""Deterministically (re)build the Gemma refusal-direction variants
(full / outlier / complement) and write the unit-normalized vectors into each
gemma_var_<v> run-dir so Stage 02 attributes toward them.

Recipe (matches docs/REFUSAL_DIRECTION_INVESTIGATION_2026-06-16.md):
  r_full      = unnormalized diff-in-means direction at L15 (dict[layer]->tensor)
  outlier_dim = argmax(|r_full|)                      (== 443 for Gemma L15)
  full        = r_full
  outlier     = zeros; outlier[outlier_dim] = r_full[outlier_dim]
  complement  = r_full.clone(); complement[outlier_dim] = 0
each unit-normalized and written to
  gemma_var_<v>/01_direction/{directions/layer_15.pt, positions_L15/pos_-2.pt}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
DEFAULT_FULL = REPO / "data/results/pipeline_runs/run_20260430_023247/01_direction/unnormalized_r.pt"
SPLIT_STATS = REPO / "data/results/emnlp_perm_edit/phase0_controllability/gemma_outlier_split_stats.json"
RUNS_BASE = REPO / "data/results/pipeline_runs"
LAYER = 15
POS = -2
EXPECT_OUTLIER_DIM = 443


def load_full_direction(path: Path, layer: int = LAYER) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        if layer in obj:
            return obj[layer]
        if "direction" in obj:
            return obj["direction"]
        raise ValueError(f"layer {layer} not in direction dict (keys {list(obj)[:5]})")
    if isinstance(obj, torch.Tensor):
        return obj
    raise ValueError(f"unrecognized direction file format: {type(obj)}")


def build_variant_directions(r_full: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return {full, outlier, complement} as UNIT-normalized 1-D tensors."""
    r = r_full.detach().float().flatten()
    outlier_dim = int(r.abs().argmax())
    full = r.clone()
    outlier = torch.zeros_like(r)
    outlier[outlier_dim] = r[outlier_dim]
    complement = r.clone()
    complement[outlier_dim] = 0.0
    return {name: v / v.norm() for name, v in
            (("full", full), ("outlier", outlier), ("complement", complement))}


def write_variant_dirs(variants: dict[str, torch.Tensor], runs_base: Path,
                       layer: int = LAYER, pos: int = POS) -> None:
    for name, unit in variants.items():
        rd = runs_base / f"gemma_var_{name}" / "01_direction"
        (rd / "directions").mkdir(parents=True, exist_ok=True)
        (rd / f"positions_L{layer}").mkdir(parents=True, exist_ok=True)
        existing = rd / "directions" / f"layer_{layer:02d}.pt"
        if existing.exists():
            old = torch.load(existing, map_location="cpu", weights_only=False)
            old = old["direction"] if isinstance(old, dict) and "direction" in old else old
            cos = float(torch.nn.functional.cosine_similarity(
                unit, old.float().flatten(), dim=0))
            assert cos > 0.999, (
                f"{name}: rebuilt direction disagrees with committed file (cos={cos:.4f}); "
                f"refusing to overwrite. Investigate the canonical source.")
        torch.save(unit, rd / "directions" / f"layer_{layer:02d}.pt")
        torch.save(unit, rd / f"positions_L{layer}" / f"pos_{pos:+d}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-direction", type=Path, default=DEFAULT_FULL)
    ap.add_argument("--runs-base", type=Path, default=RUNS_BASE)
    ap.add_argument("--check-only", action="store_true",
                    help="Run self-checks but do not write the run-dir files")
    args = ap.parse_args()

    r_full = load_full_direction(args.full_direction)
    variants = build_variant_directions(r_full)

    outlier_dim = int(r_full.float().flatten().abs().argmax())
    assert outlier_dim == EXPECT_OUTLIER_DIM, f"outlier_dim {outlier_dim} != {EXPECT_OUTLIER_DIM}"
    assert float(variants["complement"][EXPECT_OUTLIER_DIM]) == 0.0
    assert torch.nonzero(variants["outlier"]).flatten().tolist() == [EXPECT_OUTLIER_DIM]
    for v in variants.values():
        assert abs(float(v.norm()) - 1.0) < 1e-5
    if SPLIT_STATS.exists():
        norms = json.loads(SPLIT_STATS.read_text())["norms"]
        ratio = norms["outlier"] / norms["full"]
        assert abs(ratio - 0.8998) < 0.02, f"norm ratio {ratio} (expected ~0.90)"
    print(f"self-check OK: outlier_dim={outlier_dim}, unit-normalized, norms match split-stats")

    if args.check_only:
        return
    write_variant_dirs(variants, args.runs_base)
    print(f"wrote directions into gemma_var_{{full,outlier,complement}}/01_direction/ "
          f"(layer_{LAYER:02d}.pt + positions_L{LAYER}/pos_{POS:+d}.pt)")


if __name__ == "__main__":
    main()
