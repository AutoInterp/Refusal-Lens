"""
22: Deep Q/K analysis + ablation (Qwen3-4B port).
==================================================
PORT FROM: data/tejas_experiments/scripts/22_qk_deep_rigorous.py

Cross-validated head identification + ablation. Note that Qwen3-4B uses
grouped-query attention (GQA) with a different num_attention_heads /
num_key_value_heads ratio than Gemma — be careful when iterating heads.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from CONFIG import RESULTS_V2_DIR  # noqa: E402, F401

raise NotImplementedError(
    "Port from data/tejas_experiments/scripts/22_qk_deep_rigorous.py."
)
