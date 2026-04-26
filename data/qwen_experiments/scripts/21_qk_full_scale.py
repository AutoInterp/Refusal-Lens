"""
21: Q/K attention attribution at scale (Qwen3-4B port).
=======================================================
PORT FROM: data/tejas_experiments/scripts/21_qk_full_scale.py

Critical for Qwen: the Gemma analysis revealed attention carries 99.6% of
the refusal signal. The Q/K decomposition is what lets you attribute that
signal to specific heads. Numbers will differ for Qwen.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from CONFIG import RESULTS_V2_DIR  # noqa: E402, F401

raise NotImplementedError(
    "Port from data/tejas_experiments/scripts/21_qk_full_scale.py."
)
