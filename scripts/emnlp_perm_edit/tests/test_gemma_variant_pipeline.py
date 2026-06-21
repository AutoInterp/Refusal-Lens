"""CPU unit tests for the Gemma-variant pipeline (no GPU, no HF)."""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))                 # scripts/emnlp_perm_edit
sys.path.insert(0, str(HERE.parents[2] / "pipeline"))    # scripts/pipeline

from ensure_gemma_variant_directions import build_variant_directions  # noqa: E402


def _fake_r_full(d=2560, spike_dim=443, spike=-2790.53):
    torch.manual_seed(0)
    r = torch.randn(d) * 3.0
    r[spike_dim] = spike
    return r


def test_build_variant_directions():
    r = _fake_r_full()
    v = build_variant_directions(r)
    # all unit-normalized
    for name in ("full", "outlier", "complement"):
        assert abs(float(v[name].norm()) - 1.0) < 1e-5, name
    # outlier is nonzero only at 443
    nz = torch.nonzero(v["outlier"]).flatten().tolist()
    assert nz == [443], nz
    # complement zeros dim 443
    assert float(v["complement"][443]) == 0.0
    # the outlier is the dominant direction (on real Gemma data ‖outlier‖/‖full‖ ≈ 0.90;
    # the synthetic fixture's exact ratio depends on the noise scale, so assert dominance
    # rather than pinning to 0.90 — this is the real invariant build_variant_directions relies on).
    assert r[443].abs().item() / r.norm().item() > 0.85
    print("PASS test_build_variant_directions")


if __name__ == "__main__":
    test_build_variant_directions()
    print("ALL PASS")
