"""Tests for direction-math library."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from directions import compute_u_C, project_out  # noqa: E402


def test_project_out_result_orthogonal_to_direction():
    torch.manual_seed(0)
    d = torch.randn(2560)
    v = torch.randn(2560)
    result = project_out(v, d)
    assert torch.allclose(result @ d, torch.tensor(0.0), atol=1e-4)


def test_project_out_preserves_orthogonal_input():
    d = torch.tensor([1.0, 0.0, 0.0])
    v = torch.tensor([0.0, 1.0, 1.0])
    result = project_out(v, d)
    assert torch.allclose(result, v, atol=1e-6)


def test_project_out_zeroes_parallel_input():
    d = torch.tensor([1.0, 0.0, 0.0])
    v = torch.tensor([3.0, 0.0, 0.0])
    result = project_out(v, d)
    assert torch.allclose(result, torch.zeros_like(v), atol=1e-6)


def test_compute_u_C_orthogonal_to_r_hat():
    torch.manual_seed(0)
    r_hat = torch.randn(2560)
    r_jb = torch.randn(2560)
    u_C = compute_u_C(r_hat, r_jb)
    cos = torch.nn.functional.cosine_similarity(u_C.unsqueeze(0), r_hat.unsqueeze(0))
    assert cos.abs().item() < 1e-5


def test_compute_u_C_unit_norm():
    torch.manual_seed(0)
    r_hat = torch.randn(2560)
    r_jb = torch.randn(2560)
    u_C = compute_u_C(r_hat, r_jb)
    assert torch.allclose(u_C.norm(), torch.tensor(1.0), atol=1e-5)


def test_compute_u_C_raises_on_collinear_inputs():
    """If r_jb is exactly parallel to r_hat, r_jb_perp is zero — should raise."""
    r_hat = torch.tensor([1.0, 0.0, 0.0])
    r_jb = torch.tensor([2.5, 0.0, 0.0])
    with pytest.raises(ValueError, match="orthogonal component"):
        compute_u_C(r_hat, r_jb)
