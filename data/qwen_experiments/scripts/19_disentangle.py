"""
19: 2x2 disentanglement of intervention components (Qwen3-4B port).
===================================================================
PORT FROM: data/tejas_experiments/scripts/19_disentangle.py

Gemma finding: "every step" is the critical factor (not "all positions").
Pos=-2 every-step matched all-positions every-step (10/10 control, 16/16 JB).

For Qwen, validate whether the same disentanglement holds — the analogue of
"pos=-2" is whatever QWEN_BEST_POSITION turns out to be.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from CONFIG import RESULTS_V2_DIR  # noqa: E402, F401

raise NotImplementedError(
    "Port from data/tejas_experiments/scripts/19_disentangle.py."
)
