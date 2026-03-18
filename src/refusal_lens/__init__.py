"""
ٍRefusal-Lens: A great package for Refusal-Lens
"""

from __future__ import annotations

from importlib.metadata import version

__all__ = ("__version__",)

try:
    __version__ = version("refusal_lens")
except Exception:
    __version__ = "0.0.0"
