"""Tests for edge-ablation r_hat-projection hook factory."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge_ablation_hook import make_scalar_rhat_subtraction_hook  # noqa: E402


def test_hook_subtracts_predicted_amount_from_rhat_projection():
    """After hook, h_new . r_hat = (h . r_hat) - delta."""
    torch.manual_seed(0)
    r_hat = torch.randn(2560)
    delta = 1500.0
    h_in = torch.randn(2, 5, 2560)
    proj_in = (h_in.float() * r_hat).sum(-1)

    hook_fn = make_scalar_rhat_subtraction_hook(r_hat, delta)
    h_out = hook_fn(None, None, h_in.clone())
    proj_out = (h_out.float() * r_hat).sum(-1)

    diff = proj_in - proj_out
    assert torch.allclose(diff, torch.full_like(diff, delta), atol=1e-2), \
        f"expected diff ~= {delta}, got mean diff {diff.mean().item():.3f}"


def test_hook_handles_tuple_output():
    """For Gemma layer-output hooks the output is a tuple; first element is hidden states."""
    torch.manual_seed(0)
    r_hat = torch.randn(2560)
    h_in = torch.randn(2, 5, 2560)
    extra = torch.zeros(1)
    output_tuple = (h_in.clone(), extra)

    hook_fn = make_scalar_rhat_subtraction_hook(r_hat, 1000.0)
    result = hook_fn(None, None, output_tuple)
    assert isinstance(result, tuple)
    proj_in = (h_in.float() * r_hat).sum(-1)
    proj_out = (result[0].float() * r_hat).sum(-1)
    diff = proj_in - proj_out
    assert torch.allclose(diff, torch.full_like(diff, 1000.0), atol=1e-2)


def test_hook_zero_delta_is_identity():
    """delta=0 should not modify the residual."""
    r_hat = torch.randn(2560)
    h_in = torch.randn(2, 5, 2560)
    hook_fn = make_scalar_rhat_subtraction_hook(r_hat, 0.0)
    h_out = hook_fn(None, None, h_in.clone())
    assert torch.allclose(h_out, h_in, atol=1e-4)


def test_hook_negative_delta_increases_rhat_projection():
    """Negative delta should add r_hat-magnitude (push toward refusal)."""
    r_hat = torch.randn(2560)
    h_in = torch.randn(2, 5, 2560)
    proj_in = (h_in.float() * r_hat).sum(-1)
    hook_fn = make_scalar_rhat_subtraction_hook(r_hat, -500.0)
    h_out = hook_fn(None, None, h_in.clone())
    proj_out = (h_out.float() * r_hat).sum(-1)
    diff = proj_in - proj_out
    assert torch.allclose(diff, torch.full_like(diff, -500.0), atol=1e-2)
