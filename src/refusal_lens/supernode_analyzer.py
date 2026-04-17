"""
Supernode Steering and Neuronpedia Analysis Module
Analyzes supernodes and neural patterns from Neuronpedia data.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NeuronPattern:
    """Data class representing a neuron pattern."""

    neuron_id: int
    activation_strength: float
    features: list[str]
    metadata: dict[str, Any]


@dataclass
class SupernodeData:
    """Data class representing a supernode."""

    supernode_id: int
    neurons: list[NeuronPattern]
    steering_vector: np.ndarray | None = None
    activation_distribution: dict[str, float] | None = None


class SupernodeAnalyzer:
    """
    Analyzes supernodes and steering vectors from Neuronpedia data.

    This class handles loading, processing, and analyzing supernode configurations
    and their neural activations.
    """

    def __init__(self, data_dir: Path | None = None):
        """
        Initialize the SupernodeAnalyzer.

        Args:
            data_dir: Directory containing neuronpedia data files.
        """
        self.data_dir = Path(data_dir) if data_dir else Path("./data")
        self.supernodes: dict[int, SupernodeData] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_supernode_data(self, filepath: Path) -> dict[int, SupernodeData]:
        """
        Load supernode data from a JSON file.

        Args:
            filepath: Path to the supernode data file.

        Returns:
            Dictionary mapping supernode IDs to SupernodeData objects.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            msg = f"Data file not found: {filepath}"
            raise FileNotFoundError(msg)

        try:
            with open(filepath) as f:
                raw_data = json.load(f)

            self.supernodes = self._parse_supernode_data(raw_data)
            self.logger.info(
                "Loaded %d supernodes from %s", len(self.supernodes), filepath
            )
            return self.supernodes

        except json.JSONDecodeError as e:
            self.logger.error("Failed to parse JSON: %s", e)
            raise

    def _parse_supernode_data(
        self, raw_data: dict[str, Any]
    ) -> dict[int, SupernodeData]:
        """
        Parse raw data into SupernodeData objects.

        Args:
            raw_data: Raw dictionary from JSON file.

        Returns:
            Dictionary of SupernodeData objects.
        """
        supernodes = {}

        for supernode_id, data in raw_data.items():
            try:
                neurons = [
                    NeuronPattern(
                        neuron_id=neuron["id"],
                        activation_strength=neuron.get("strength", 0.0),
                        features=neuron.get("features", []),
                        metadata=neuron.get("metadata", {}),
                    )
                    for neuron in data.get("neurons", [])
                ]

                steering_vector = None
                if "steering_vector" in data:
                    steering_vector = np.array(data["steering_vector"])

                supernodes[int(supernode_id)] = SupernodeData(
                    supernode_id=int(supernode_id),
                    neurons=neurons,
                    steering_vector=steering_vector,
                    activation_distribution=data.get("activation_distribution"),
                )
            except (KeyError, ValueError) as e:
                self.logger.warning("Failed to parse supernode %s: %s", supernode_id, e)
                continue

        return supernodes

    def get_supernode(self, supernode_id: int) -> SupernodeData | None:
        """
        Retrieve a specific supernode by ID.

        Args:
            supernode_id: The ID of the supernode.

        Returns:
            SupernodeData object or None if not found.
        """
        return self.supernodes.get(supernode_id)

    def get_top_neurons(
        self, supernode_id: int, top_k: int = 10
    ) -> list[NeuronPattern]:
        """
        Get top-k neurons by activation strength for a supernode.

        Args:
            supernode_id: The ID of the supernode.
            top_k: Number of top neurons to return.

        Returns:
            List of top NeuronPattern objects.
        """
        supernode = self.get_supernode(supernode_id)
        if not supernode:
            return []

        sorted_neurons = sorted(
            supernode.neurons, key=lambda n: n.activation_strength, reverse=True
        )
        return sorted_neurons[:top_k]

    def analyze_steering_vector(self, supernode_id: int) -> dict[str, Any] | None:
        """
        Analyze the steering vector of a supernode.

        Args:
            supernode_id: The ID of the supernode.

        Returns:
            Dictionary with steering vector statistics or None.
        """
        supernode = self.get_supernode(supernode_id)
        if not supernode or supernode.steering_vector is None:
            return None

        vector = supernode.steering_vector
        return {
            "magnitude": float(np.linalg.norm(vector)),
            "mean": float(np.mean(vector)),
            "std": float(np.std(vector)),
            "min": float(np.min(vector)),
            "max": float(np.max(vector)),
            "dimension": len(vector),
        }

    def get_feature_distribution(self, supernode_id: int) -> dict[str, int]:
        """
        Get the distribution of features in a supernode.

        Args:
            supernode_id: The ID of the supernode.

        Returns:
            Dictionary mapping feature names to their frequency.
        """
        supernode = self.get_supernode(supernode_id)
        if not supernode:
            return {}

        feature_counts: dict[str, int] = {}
        for neuron in supernode.neurons:
            for feature in neuron.features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1

        return dict(sorted(feature_counts.items(), key=lambda x: x[1], reverse=True))

    def save_analysis(self, output_path: Path, analysis: dict[str, Any]) -> None:
        """
        Save analysis results to a JSON file.

        Args:
            output_path: Path where to save the results.
            analysis: Dictionary containing analysis results.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(analysis, f, indent=2)

        self.logger.info("Analysis saved to %s", output_path)

    def get_summary_statistics(self) -> dict[str, Any]:
        """
        Get summary statistics for all loaded supernodes.

        Returns:
            Dictionary with aggregated statistics.
        """
        if not self.supernodes:
            return {}

        total_neurons = sum(len(sn.neurons) for sn in self.supernodes.values())
        avg_neurons = total_neurons / len(self.supernodes)

        return {
            "total_supernodes": len(self.supernodes),
            "total_neurons": total_neurons,
            "average_neurons_per_supernode": avg_neurons,
            "supernode_ids": list(self.supernodes.keys()),
        }
