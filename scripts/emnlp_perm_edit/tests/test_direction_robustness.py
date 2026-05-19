"""Tests for direction-robustness audit helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# 00_ prefixed scripts are not importable directly; use importlib
import importlib.util
spec = importlib.util.spec_from_file_location(
    "direction_robustness",
    Path(__file__).resolve().parents[1] / "00_direction_robustness.py",
)
direction_robustness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(direction_robustness)

compute_per_prompt_cosines = direction_robustness.compute_per_prompt_cosines
random_baseline_cosine_stats = direction_robustness.random_baseline_cosine_stats
pearson_cosine = direction_robustness.pearson_cosine


def test_per_prompt_cosines_match_class_mean_when_all_prompts_identical():
    """Sanity: if every prompt's displacement is identical, per-prompt mean equals class-mean cosine."""
    torch.manual_seed(0)
    r_hat = torch.randn(2560)
    common_delta = torch.randn(2560)
    h_jb = common_delta.unsqueeze(0).expand(50, 2560).clone()
    h_bare = torch.zeros(50, 2560)
    per_prompt = compute_per_prompt_cosines(h_jb, h_bare, r_hat)
    class_mean_cos = torch.nn.functional.cosine_similarity(
        common_delta.unsqueeze(0), r_hat.unsqueeze(0)
    ).item()
    assert abs(per_prompt["mean_cos"] - class_mean_cos) < 1e-5
    assert per_prompt["std_cos"] < 1e-5


def test_per_prompt_cosines_have_nonzero_std_when_prompts_differ():
    """When per-prompt displacements vary, std is nonzero."""
    torch.manual_seed(0)
    r_hat = torch.randn(2560)
    h_jb = torch.randn(50, 2560)
    h_bare = torch.randn(50, 2560)
    per_prompt = compute_per_prompt_cosines(h_jb, h_bare, r_hat)
    assert per_prompt["std_cos"] > 0.01


def test_random_baseline_returns_expected_stats():
    """Random-baseline returns 95th percentile of absolute cosines and the test direction's cosine."""
    torch.manual_seed(0)
    r_hat = torch.randn(2560)
    r_jb = torch.randn(2560)
    stats = random_baseline_cosine_stats(r_jb, r_hat, n_random=500, seed=42)
    assert "p95_abs_random_cos" in stats
    assert "real_cos_with_r_hat" in stats
    assert "rank_of_real_in_random" in stats
    # rank is in [0, n_random]
    assert 0 <= stats["rank_of_real_in_random"] <= 500


def test_pearson_cosine_zeros_out_all_ones_bias():
    """If both vectors are pure all-ones, raw cosine is 1.0 but Pearson cosine is 0/0 -> 0."""
    r_hat = torch.ones(100)
    r_jb = torch.ones(100) * 3.0  # parallel to all-ones, but scaled
    raw = torch.nn.functional.cosine_similarity(
        r_jb.unsqueeze(0), r_hat.unsqueeze(0)
    ).item()
    pearson = pearson_cosine(r_jb, r_hat)
    assert raw == pytest.approx(1.0)
    # After mean-subtraction both vectors are zero; pearson returns 0 by convention
    assert pearson == pytest.approx(0.0, abs=1e-5)


def test_pearson_cosine_orthogonal_random_vectors_differs_from_raw():
    """For random vectors, Pearson and raw cosines should be similar but not identical."""
    torch.manual_seed(0)
    r_hat = torch.randn(2560)
    r_jb = torch.randn(2560)
    raw = torch.nn.functional.cosine_similarity(
        r_jb.unsqueeze(0), r_hat.unsqueeze(0)
    ).item()
    pearson = pearson_cosine(r_jb, r_hat)
    # Both should be close to 0 for random pairs; small numerical difference is expected
    assert abs(raw - pearson) < 0.1
