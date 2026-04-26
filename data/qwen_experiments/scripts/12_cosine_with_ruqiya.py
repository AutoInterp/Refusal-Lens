"""
12: Cosine similarity with Ruqiya's direction (Qwen3-4B port).
==============================================================
PORT FROM: data/tejas_experiments/scripts/12_cosine_with_ruqiya.py

What changes vs Gemma:
  - Use MODEL_NAME / paths from CONFIG.py
  - Ruqiya's direction was for Gemma; for Qwen you need a Qwen reference
    direction (or re-purpose to compare across configurations of your own
    Qwen direction, e.g. different positions/layers).
  - Replace `range(34)` with `range(model.config.num_hidden_layers)` (= 36).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from CONFIG import RESULTS_V2_DIR  # noqa: E402, F401

raise NotImplementedError(
    "Port from data/tejas_experiments/scripts/12_cosine_with_ruqiya.py. "
    "See INDEX.md for the porting checklist."
)
