"""
15: Direct ablation along refusal direction (Qwen3-4B port).
============================================================
PORT FROM: data/tejas_experiments/scripts/15_causal_intervention.py

On Gemma this version was the "failed" baseline before Arditi: directly
removing the projection along r at L15 didn't reliably flip jailbreaks.
Re-run on Qwen to confirm the same negative result before moving to 16/17.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from CONFIG import RESULTS_V2_DIR  # noqa: E402, F401

raise NotImplementedError(
    "Port from data/tejas_experiments/scripts/15_causal_intervention.py."
)
