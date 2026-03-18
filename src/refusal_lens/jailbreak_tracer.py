"""
Jailbreak Behavior Tracing Module
Traces and analyzes jailbreak attempts and behaviors in neural network models.
"""

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class AttackType(Enum):
    """Enumeration of jailbreak attack types."""

    PROMPT_INJECTION = "prompt_injection"
    ROLE_PLAY = "role_play"
    CONTEXT_CONFUSION = "context_confusion"
    TOKEN_SMUGGLING = "token_smuggling"
    INDIRECT_REFERENCE = "indirect_reference"
    UNKNOWN = "unknown"


@dataclass
class JailbreakAttempt:
    """Data class representing a jailbreak attempt."""

    attempt_id: str
    timestamp: str
    attack_type: AttackType
    input_text: str
    model_response: str
    success: bool
    confidence_score: float
    triggered_layers: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BehaviorTrace:
    """Data class representing a behavior trace through the model."""

    trace_id: str
    attempt_id: str
    layer: int
    activation_values: list[float]
    neuron_indices: list[int]
    timestamp: str


class JailbreakTracer:
    """
    Traces and analyzes jailbreak behavior through neural network layers.

    This class handles loading, processing, and analyzing jailbreak attempts
    and their propagation through the model.
    """

    def __init__(self, data_dir: Path | None = None):
        """
        Initialize the JailbreakTracer.

        Args:
            data_dir: Directory containing jailbreak trace data files.
        """
        self.data_dir = Path(data_dir) if data_dir else Path("./data")
        self.attempts: dict[str, JailbreakAttempt] = {}
        self.traces: dict[str, list[BehaviorTrace]] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_attempts(self, filepath: Path) -> dict[str, JailbreakAttempt]:
        """
        Load jailbreak attempts from a JSON file.

        Args:
            filepath: Path to the jailbreak attempts file.

        Returns:
            Dictionary mapping attempt IDs to JailbreakAttempt objects.

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

            self.attempts = self._parse_attempts(raw_data)
            self.logger.info(
                "Loaded %d jailbreak attempts from %s", len(self.attempts), filepath
            )
            return self.attempts

        except json.JSONDecodeError as e:
            self.logger.error("Failed to parse JSON: %s", e)
            raise

    def _parse_attempts(self, raw_data: dict[str, Any]) -> dict[str, JailbreakAttempt]:
        """
        Parse raw data into JailbreakAttempt objects.

        Args:
            raw_data: Raw dictionary from JSON file.

        Returns:
            Dictionary of JailbreakAttempt objects.
        """
        attempts = {}

        for attempt_id, data in raw_data.items():
            try:
                attack_type = AttackType(data.get("attack_type", "unknown"))

                attempt = JailbreakAttempt(
                    attempt_id=attempt_id,
                    timestamp=data.get("timestamp", datetime.now().isoformat()),
                    attack_type=attack_type,
                    input_text=data.get("input_text", ""),
                    model_response=data.get("model_response", ""),
                    success=data.get("success", False),
                    confidence_score=float(data.get("confidence_score", 0.0)),
                    triggered_layers=data.get("triggered_layers", []),
                    metadata=data.get("metadata", {}),
                )
                attempts[attempt_id] = attempt

            except (KeyError, ValueError) as e:
                self.logger.warning("Failed to parse attempt %s: %s", attempt_id, e)
                continue

        return attempts

    def load_traces(self, filepath: Path) -> dict[str, list[BehaviorTrace]]:
        """
        Load behavior traces from a JSON file.

        Args:
            filepath: Path to the traces file.

        Returns:
            Dictionary mapping attempt IDs to lists of BehaviorTrace objects.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            msg = f"Data file not found: {filepath}"
            raise FileNotFoundError(msg)

        try:
            with open(filepath) as f:
                raw_data = json.load(f)

            self.traces = self._parse_traces(raw_data)
            self.logger.info("Loaded traces for %d attempts", len(self.traces))
            return self.traces

        except json.JSONDecodeError as e:
            self.logger.error("Failed to parse JSON: %s", e)
            raise

    def _parse_traces(self, raw_data: dict[str, Any]) -> dict[str, list[BehaviorTrace]]:
        """
        Parse raw trace data into BehaviorTrace objects.

        Args:
            raw_data: Raw dictionary from JSON file.

        Returns:
            Dictionary of trace lists.
        """
        traces = {}

        for attempt_id, trace_list in raw_data.items():
            attempt_traces = []
            for trace_data in trace_list:
                try:
                    trace = BehaviorTrace(
                        trace_id=trace_data.get("trace_id", ""),
                        attempt_id=attempt_id,
                        layer=int(trace_data.get("layer", 0)),
                        activation_values=trace_data.get("activation_values", []),
                        neuron_indices=trace_data.get("neuron_indices", []),
                        timestamp=trace_data.get(
                            "timestamp", datetime.now().isoformat()
                        ),
                    )
                    attempt_traces.append(trace)
                except (KeyError, ValueError) as e:
                    self.logger.warning(
                        "Failed to parse trace in %s: %s", attempt_id, e
                    )
                    continue

            if attempt_traces:
                traces[attempt_id] = attempt_traces

        return traces

    def get_attempt(self, attempt_id: str) -> JailbreakAttempt | None:
        """
        Retrieve a specific jailbreak attempt.

        Args:
            attempt_id: The ID of the attempt.

        Returns:
            JailbreakAttempt object or None if not found.
        """
        return self.attempts.get(attempt_id)

    def get_successful_attempts(self) -> list[JailbreakAttempt]:
        """
        Get all successful jailbreak attempts.

        Returns:
            List of successful JailbreakAttempt objects.
        """
        return [a for a in self.attempts.values() if a.success]

    def get_attempts_by_type(self, attack_type: AttackType) -> list[JailbreakAttempt]:
        """
        Get all attempts of a specific attack type.

        Args:
            attack_type: The type of attack to filter by.

        Returns:
            List of JailbreakAttempt objects matching the type.
        """
        return [a for a in self.attempts.values() if a.attack_type == attack_type]

    def get_trace(self, attempt_id: str) -> list[BehaviorTrace] | None:
        """
        Retrieve the behavior trace for an attempt.

        Args:
            attempt_id: The ID of the attempt.

        Returns:
            List of BehaviorTrace objects or None if not found.
        """
        return self.traces.get(attempt_id)

    def get_activated_layers(self, attempt_id: str) -> set[int]:
        """
        Get all layers that were activated during an attempt.

        Args:
            attempt_id: The ID of the attempt.

        Returns:
            Set of layer indices that were activated.
        """
        trace = self.get_trace(attempt_id)
        if not trace:
            return set()
        return {t.layer for t in trace}

    def get_layer_statistics(
        self, attempt_id: str, layer: int
    ) -> dict[str, float] | None:
        """
        Get statistics for a specific layer during an attempt.

        Args:
            attempt_id: The ID of the attempt.
            layer: The layer number.

        Returns:
            Dictionary with layer statistics or None.
        """
        trace = self.get_trace(attempt_id)
        if not trace:
            return None

        layer_traces = [t for t in trace if t.layer == layer]
        if not layer_traces:
            return None

        # Combine all activation values from this layer
        all_activations = []
        for t in layer_traces:
            all_activations.extend(t.activation_values)

        if not all_activations:
            return None

        activations = np.array(all_activations)

        return {
            "mean_activation": float(np.mean(activations)),
            "std_activation": float(np.std(activations)),
            "max_activation": float(np.max(activations)),
            "min_activation": float(np.min(activations)),
            "activated_neurons": len(
                {n for t in layer_traces for n in t.neuron_indices}
            ),
        }

    def get_attack_success_rate(self, attack_type: AttackType | None = None) -> float:
        """
        Calculate the success rate of jailbreak attempts.

        Args:
            attack_type: Optional filter by attack type.

        Returns:
            Success rate as a float between 0 and 1.
        """
        if attack_type:
            attempts = self.get_attempts_by_type(attack_type)
        else:
            attempts = list(self.attempts.values())

        if not attempts:
            return 0.0

        successful = sum(1 for a in attempts if a.success)
        return successful / len(attempts)

    def get_high_confidence_attempts(
        self, threshold: float = 0.8
    ) -> list[JailbreakAttempt]:
        """
        Get jailbreak attempts above a confidence threshold.

        Args:
            threshold: Confidence score threshold (0-1).

        Returns:
            List of high-confidence JailbreakAttempt objects.
        """
        return [a for a in self.attempts.values() if a.confidence_score >= threshold]

    def analyze_attack_pattern(self, attack_type: AttackType) -> dict[str, Any]:
        """
        Analyze patterns for a specific attack type.

        Args:
            attack_type: The attack type to analyze.

        Returns:
            Dictionary with pattern analysis.
        """
        attempts = self.get_attempts_by_type(attack_type)

        if not attempts:
            return {}

        successful = [a for a in attempts if a.success]

        return {
            "total_attempts": len(attempts),
            "successful_attempts": len(successful),
            "success_rate": len(successful) / len(attempts),
            "average_confidence": sum(a.confidence_score for a in attempts)
            / len(attempts),
            "most_triggered_layer": self._get_most_common_layer(attempts),
        }

    def _get_most_common_layer(self, attempts: list[JailbreakAttempt]) -> int | None:
        """Get the most commonly triggered layer across attempts."""
        all_layers = []
        for attempt in attempts:
            all_layers.extend(attempt.triggered_layers)

        if not all_layers:
            return None

        return Counter(all_layers).most_common(1)[0][0]

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
        Get summary statistics for all loaded data.

        Returns:
            Dictionary with aggregated statistics.
        """
        if not self.attempts:
            return {}

        attack_types = {}
        for attack_type in AttackType:
            attempts = self.get_attempts_by_type(attack_type)
            if attempts:
                attack_types[attack_type.value] = {
                    "count": len(attempts),
                    "success_rate": len([a for a in attempts if a.success])
                    / len(attempts),
                }

        return {
            "total_attempts": len(self.attempts),
            "successful_attempts": len(self.get_successful_attempts()),
            "overall_success_rate": self.get_attack_success_rate(),
            "attack_types": attack_types,
            "traces_available": len(self.traces),
        }
