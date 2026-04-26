"""
16: Arditi causal intervention (Qwen3-4B port).
================================================
PORT FROM: data/tejas_experiments/scripts/16_causal_arditi.py

The Arditi method: add r to the residual stream at the causal layer at
EVERY generation step (not just prefill). On Gemma this flipped 95/95
jailbroken prompts at L15.

For Qwen you need to:
  1. Sweep candidate causal layers (don't assume L15)
  2. Find the layer where adding r flips JB → refuse without breaking benign
  3. Save the chosen layer back to CONFIG.QWEN_CAUSAL_LAYER
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from CONFIG import RESULTS_V2_DIR  # noqa: E402, F401

raise NotImplementedError(
    "Port from data/tejas_experiments/scripts/16_causal_arditi.py."
)
