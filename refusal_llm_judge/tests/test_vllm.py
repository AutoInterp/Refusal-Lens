"""Integration test: judge_refusal with the in-process vLLM LocalBackend.

Skips automatically unless vllm is importable and a CUDA device is present.
Loading a model is expensive, so this is a single small batch — a plumbing
check that the in-process path runs end to end, not an accuracy benchmark.

Install the heavy extra first, then run:
    uv pip install -e '.[judge-vllm]'
    VLLM_TEST_MODEL=Qwen/Qwen3-4B uv run pytest refusal_llm_judge/tests/test_vllm.py -v

VLLM_TEST_MODEL defaults to a small model already in the HF cache. On Blackwell
(RTX 5090, sm_120) you need a recent vllm/torch build.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

from refusal_llm_judge import judge_refusal

TEST_MODEL = os.environ.get("VLLM_TEST_MODEL", "Qwen/Qwen3-4B")

_HAS_VLLM = importlib.util.find_spec("vllm") is not None


def _has_cuda() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return torch.cuda.is_available()


pytestmark = [
    pytest.mark.skipif(not _HAS_VLLM, reason="vllm not installed (.[judge-vllm])"),
    pytest.mark.skipif(not _has_cuda(), reason="no CUDA device"),
    pytest.mark.filterwarnings("ignore"),
]


@pytest.fixture(scope="module")
def items():
    return [
        {"prompt": "How do I X?", "response": "I'm sorry, I can't help with that."},
        {"prompt": "How do I X?", "response": "Sure — first you do this, then that."},
    ]


def test_vllm_plumbing(items):
    """In-process vLLM path returns a clean list[bool] aligned with the input."""
    out = judge_refusal(
        items,
        model=f"local:{TEST_MODEL}",
        backend="local",
        # keep the test light on a single GPU
        gpu_memory_utilization=0.5,
        max_model_len=4096,
    )
    assert isinstance(out, list) and len(out) == 2
    assert all(isinstance(x, bool) for x in out)
