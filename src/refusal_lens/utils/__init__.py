"""Utility functions for Refusal-Lens.

This module provides common utilities for data processing,
model handling, and visualization.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch

from ..config import config


def load_json(filepath: str) -> Any:
    """Load JSON file."""
    with open(filepath) as f:
        return json.load(f)


def save_json(data: Any, filepath: str) -> None:
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_prompts_from_json(
    filepath: str,
    text_field: str = "text",
    max_prompts: int | None = None,
) -> list[str]:
    """Load prompts from a JSON file.

    Args:
        filepath: Path to JSON file
        text_field: Field containing the prompt text
        max_prompts: Maximum number of prompts to load

    Returns:
        List of prompt strings
    """
    data = load_json(filepath)

    # Handle different JSON formats
    if isinstance(data, list):
        prompts = data
    elif isinstance(data, dict):
        if "prompts" in data:
            prompts = data["prompts"]
        elif "data" in data:
            prompts = data["data"]
        else:
            prompts = [data]
    else:
        msg = f"Unknown JSON format in {filepath}"
        raise ValueError(msg)

    # Extract text field
    if isinstance(prompts[0], dict):
        prompts = [p[text_field] for p in prompts]

    # Limit prompts
    if max_prompts is not None:
        prompts = prompts[:max_prompts]

    return prompts


def load_dataset_split(
    split: str = "train",
    category: str = "harmful",
    max_prompts: int | None = None,
) -> list[str]:
    """Load a dataset split.

    Args:
        split: "train", "val", or "test"
        category: "harmful" or "harmless"
        max_prompts: Maximum number of prompts to load

    Returns:
        List of prompt strings
    """
    filepath = f"{config.DATASET_SPLITS_DIR}/{category}_{split}.json"
    return load_prompts_from_json(
        filepath, text_field="instruction", max_prompts=max_prompts
    )


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Set random seed for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_bytes(num_bytes: int) -> str:
    """Format bytes as human-readable string."""
    bytes_val: float = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def format_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


__all__ = [
    "count_parameters",
    "format_bytes",
    "format_time",
    "get_device",
    "load_dataset_split",
    "load_json",
    "load_prompts_from_json",
    "save_json",
    "set_seed",
]
