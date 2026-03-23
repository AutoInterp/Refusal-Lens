"""
Dataset loading for refusal-direction experiments.

Reads from a local JSON splits in ``dataset/refusal_direction_dataset/splits/``
and produces deterministic contrastive pairs from difference-in-means analysis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "dataset" / "refusal_direction_dataset"

DEFAULT_N_TRAIN = 128
DEFAULT_N_EVAL = 64
DEFAULT_SEED = 42

@dataclass
class Instruction:
    """A single instruction from the dataset."""

    text: str
    category: str | None = None
    source: str | None = None

@dataclass
class ContrastiveSplit:
    """A paired set of harmful/harmless instructions from one split."""

    harmful: list[str]
    harmless: list[str]

    def __len__(self) -> int:
        return len(self.harmful)
    
    def __post_init__(self) -> None:
        if len(self.harmful) != len(self.harmless):
            msg = (
                f"Harmful ({len(self.harmful)}) and harmless ({len(self.harmless)}) "
                "lists must have equal length for contrastive analysis."
            )
            raise ValueError(msg)

@dataclass
class ContrastiveDataset:
    """Train and eval splits of contrastive harmful/harmless pairs."""

    train: ContrastiveSplit
    eval: ContrastiveSplit
    seed: int = DEFAULT_SEED
    metadata: dict[str, Any] = field(default_factory=dict)

def load_split(
    split_name: str,
    *,
    dataset_dir: Path | None = None,
) -> list[Instruction]:
    """
    Load a single JSON split file.

    Args:
        split_name: Name of the split file without extension (e.g., 'harmful_train', 'harmless_test').
        dataset_dir: Path of the dataset root directory. Defaults to ``dataset/refusal_direction_dataset/``.
    
    Returns:
        List of :class:`Instruction` objects.
    
    Raises:
        FileNotFoundError: If the split file does not exist.
    """
    if dataset_dir is None:
        dataset_dir = _DEFAULT_DATASET_DIR
    path = dataset_dir / "splits" / f"{split_name}.json"

    if not path.exists():
        msg = f"Split file not found: {path}"
        raise FileNotFoundError(msg)
    
    with open(path) as f:
        raw: list[dict[str, Any]] = json.load(f)
    
    return [
        Instruction(
            text=item["instruction"],
            category=item.get("category"),
            source=split_name,
        ) for item in raw
    ]

def load_processed(
    name: str,
    *,
    dataset_dir: Path | None = None,
) -> list[Instruction]:
    """
    Load a processed dataset file (e.g., 'advbench', 'alpaca').

    Args:
        name: Name of the processed file without extension.
        dataset_dir: Path to the dataset root directory.
    
    Returns:
        List of :class:`Instruction` objects.
    """
    if dataset_dir is None:
        dataset_dir = _DEFAULT_DATASET_DIR
    path = dataset_dir / "processed" / f"{name}.json"

    if not path.exists():
        msg = f"Processed file not found: {path}"
        raise FileNotFoundError(msg)

    with open(path) as f:
        raw: list[dict[str, Any]] = json.load(f)
    
    return [
        Instruction(
            text=item["instruction"],
            category=item.get("category"),
            source=name,
        ) for item in raw
    ]

def list_available_splits(
    *,
    dataset_dir: Path | None = None,
) -> list[str]:
    """
    List available split names in the splits directory.

    Returns:
        Sorted list of processed file names (without .json extension).
    """
    if dataset_dir is None:
        dataset_dir = _DEFAULT_DATASET_DIR
    splits_dir = dataset_dir / "splits"
    if not splits_dir.exists():
        return []
    return sorted(p.stem for p in splits_dir.glob("*.json"))

def list_available_processed(
    *,
    dataset_dir: Path | None = None,
) -> list[str]:
    """
    List available processed dataset names.

    Returns:
        Sorted list of processed file names (without .json extension).
    """
    if dataset_dir is None:
        dataset_dir = _DEFAULT_DATASET_DIR
    processed_dir = dataset_dir / "processed"
    if not processed_dir.exists():
        return []
    return sorted(p.stem for p in processed_dir.glob("*.json"))

def create_contrastive_dataset(
    *,
    n_train: int = DEFAULT_N_TRAIN,
    n_eval: int = DEFAULT_N_EVAL,
    seed: int = DEFAULT_SEED,
    dataset_dir: Path | None = None,
) -> ContrastiveDataset:
    """
    Create contrastive train/eval splits from local dataset files.

    Args:
        n_train: Number of contrastive pairs for training.
        n_eval: Number of contrastive pairs for evaluation.
        seed: Random seed for deterministic shuffling.
        dataset_dir: Path to the dataset root directory.
    
    Returns:
        A :class:`ContrastiveDataset` with train and eval splits.
    
    Raises:
        ValueError: If the dataset does not contain enough samples.
    """
    harmful_all = load_split("harmful_train", dataset_dir=dataset_dir)
    harmless_all = load_split("harmless_train", dataset_dir=dataset_dir)

    total_needed = n_train + n_eval

    if len(harmful_all) < total_needed:
        msg = (
            f"Need {total_needed} harmful samples (train={n_train} + eval={n_eval}), "
            f"but only {len(harmful_all)} available"
        )
        raise ValueError(msg)
    if len(harmless_all) < total_needed:
        msg = (
            f"Need {total_needed} harmless samples (train={n_train} + eval={n_eval}), "
            f"but only {len(harmless_all)} available"
        )
        raise ValueError(msg)
    
    rng = np.random.RandomState(seed)
    harmful_idx = rng.permutation(len(harmful_all))
    harmless_idx = rng.permutation(len(harmless_all))

    harmful_train = [harmful_all[i].text for i in harmful_idx[:n_train]]
    harmful_eval = [harmful_all[i].text for i in harmful_idx[n_train:total_needed]]
    harmless_train = [harmless_all[i].text for i in harmless_idx[:n_train]]
    harmless_eval = [harmless_all[i].text for i in harmless_idx[n_train:total_needed]]

    logger.info(
        "Created contrastive dataset: train=%d, eval=%d (seed=%d)",
        n_train, n_eval, seed,
    )

    return ContrastiveDataset(
        train=ContrastiveSplit(harmful=harmful_train, harmless=harmless_train),
        eval=ContrastiveSplit(harmful=harmful_eval, harmless=harmless_eval),
        seed=seed,
        metadata={
            "n_harmful_total": len(harmful_all),
            "n_harmless_total": len(harmless_all),
            "n_train": n_train,
            "n_eval": n_eval,
        },
    )