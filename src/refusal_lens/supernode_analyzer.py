"""
Supernode Steering and Neuronpedia Analysis Module
Analyzes supernodes and neural patterns from Neuronpedia data.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

# constants for testing
NEURONPEDIA_GRAPH_URL: str = (
    "https://www.neuronpedia.org/gemma-3-4b-it/graph?"
    "slug=iamanfbiagentstu-1773058125916&pruningThreshold=0.8"
    "&densityThreshold=0.99"
    "&pinnedIds=35_19058_60%2C35_236777_60%2C19_5480_56%2C22_879_56%2C21_7608_56"
    "%2C22_11662_59%2C20_6779_59%2C17_7356_59%2C18_10166_59%2C0_3393_58"
    "%2C19_4137_59%2C18_3677_59%2C17_6638_59%2C18_84158_59%2C16_37041_59"
    "%2C16_9947_59"
    "&supernodes=%5B%5B%22Role+Playing+Features%22%2C%2218_84158_59%22%2C"
    "%2216_37041_59%22%2C%2219_4137_59%22%2C%2218_3677_59%22%2C"
    "%2217_6638_59%22%5D%2C%5B%22Refuse%22%2C%2216_9947_59%22%2C"
    "%220_3393_58%22%2C%2218_10166_59%22%2C%2217_7356_59%22%5D%5D"
)

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

class Feature(NamedTuple):
    """
    A single SAE/transcoder feature identified by its coordinates.

    Attributes:
        layer: the transformer layer index.
        feature_idx: the feature index within the SAE/transcoder dictionary.
        token_pos: the token position where this feature was identified.
    """
    layer: int
    feature_idx: int
    token_pos: int

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

from urllib.parse import urlparse, parse_qs, unquote

def parse_neuronpedia_url(url: str) -> dict[str, list[Feature]]:
    """
    Parse a Neuronpedia graph URL into a named groups of features.

    The URL encodes supernodes as a JSON array in the ``supernodes`` query
    parameter. Each supernode is ``[name, "layer_idx_pos", ...]```.

    Args:
        url: a full Neuronpedia graph URL with a ``supernodes`` query param.
    
    Returns:
        Dictionary mapping supernode names to lists of Feature objects
    
    Raises:
        ValueError: If the URL has no ``supernodes`` parameter or the JSON inside it cannot be decoded
    """
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    raw = params.get("supernodes", [None])[0]
    if raw is None:
        raise ValueError("URL has no 'supernodes' query parameter")
    
    supernodes_data = json.loads(unquote(raw))
    result: dict[str, list[Feature]] = {}
    for sn in supernodes_data:
        name = sn[0]
        features: list[Feature] = []
        for node_id in sn[1:]:
            parts = node_id.split("_")
            features.append(Feature(
                layer=int(parts[0]),
                feature_idx=int(parts[1]),
                token_pos=int(parts[2]),
            ))
        result[name] = features
    return result

def get_transcoder_for_feature(
    feature: Feature,
    transcoders_16k: dict[int, Any],
    transcoders_wide: dict[int, Any],
    wide_threshold: int = 16384,
) -> tuple[Any, str | None]:
    """
    Look up the correct transcoder for a given Feature.

    The notebook uses 262k-width transcoders for features with
    ``feature_idx >= 16384`` on layers that have wide transcoders loaded,
    falling back to 16k-width transcoders otherwise.

    Args:
        feature: The Feature to look up.
        transcoders_16k: Mapping of layer index → 16k-width transcoder.
        transcoders_wide: Mapping of layer index → 262k-width transcoder.
        wide_threshold: Feature index threshold for wide transcoders (default 16384).

    Returns:
        ``(transcoder, width_label)`` where *width_label* is ``"262k"``,
        ``"16k"``, or ``None`` if no matching transcoder was found.
    """
    layer = feature.layer
    idx = feature.feature_idx

    if idx >= wide_threshold and layer in transcoders_wide:
        tc = transcoders_wide[layer]
        if hasattr(tc, "d_sae") and idx < tc.d_sae:
            return tc, "262k"

    if layer in transcoders_16k:
        tc = transcoders_16k[layer]
        if hasattr(tc, "d_sae") and idx < tc.d_sae:
            return tc, "16k"

    return None, None

ROLEPLAY_FEATURES: list[Feature] = [
    Feature(layer=18, feature_idx=84158, token_pos=59),
    Feature(layer=16, feature_idx=37041, token_pos=59),
    Feature(layer=19, feature_idx=4137, token_pos=59),
    Feature(layer=18, feature_idx=3677, token_pos=59),
    Feature(layer=17, feature_idx=6638, token_pos=59),
]

REFUSAL_FEATURES: list[Feature] = [
    Feature(layer=16, feature_idx=9947, token_pos=59),
    Feature(layer=0, feature_idx=3393, token_pos=58),
    Feature(layer=18, feature_idx=10166, token_pos=59),
    Feature(layer=17, feature_idx=7356, token_pos=59),
]

def supernodes_to_features(supernodes: dict[int, SupernodeData]) -> list[Feature]:
    """
    Convert loaded SupernodeData neurons to Feature objects.

    Maps each NeuronPattern in each supernode to a Feature using
    ``neuron_id`` as ``feature_idx`` and the supernode_id as a proxy
    for layer.  This is a best-effort bridge — the mapping is only
    meaningful when NeuronPattern metadata contains layer information.

    Args:
        supernodes: Dictionary of SupernodeData objects (from load_supernode_data).

    Returns:
        Flat list of Feature objects.
    """
    features: list[Feature] = []
    for sn in supernodes.values():
        for neuron in sn.neurons:
            layer = neuron.metadata.get("layer", sn.supernode_id)
            features.append(Feature(
                layer=int(layer),
                feature_idx=neuron.neuron_id,
                token_pos=neuron.metadata.get("token_pos", 0),
            ))
    return features