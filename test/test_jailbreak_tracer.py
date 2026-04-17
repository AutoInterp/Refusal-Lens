"""
Tests for JailbreakTracer module.

This module contains comprehensive tests for the JailbreakTracer class,
including tests for loading data, querying attempts and traces, and
performing analysis.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from refusal_lens.jailbreak_tracer import (
    AttackType,
    BehaviorTrace,
    JailbreakAttempt,
    JailbreakTracer,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def test_data_dir():
    """Return the path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture()
def test_attempts_file(test_data_dir):
    """Return the path to test attempts file."""
    return test_data_dir / "test_attempts.json"


@pytest.fixture()
def test_traces_file(test_data_dir):
    """Return the path to test traces file."""
    return test_data_dir / "test_traces.json"


@pytest.fixture()
def tracer(test_attempts_file, test_traces_file):
    """Create a JailbreakTracer with loaded test data."""
    tracer = JailbreakTracer()
    tracer.load_attempts(test_attempts_file)
    tracer.load_traces(test_traces_file)
    return tracer


@pytest.fixture()
def empty_tracer():
    """Create an empty JailbreakTracer."""
    return JailbreakTracer()


# ============================================================================
# Tests for AttackType enum
# ============================================================================


class TestAttackType:
    """Tests for the AttackType enum."""

    def test_attack_type_values(self):
        """Test that all expected attack types exist."""
        expected_types = [
            "prompt_injection",
            "role_play",
            "context_confusion",
            "token_smuggling",
            "indirect_reference",
            "unknown",
        ]
        actual_types = [at.value for at in AttackType]

        for expected in expected_types:
            assert expected in actual_types

    def test_attack_type_from_string(self):
        """Test creating AttackType from string."""
        assert AttackType("prompt_injection") == AttackType.PROMPT_INJECTION
        assert AttackType("role_play") == AttackType.ROLE_PLAY
        assert AttackType("unknown") == AttackType.UNKNOWN


# ============================================================================
# Tests for JailbreakAttempt dataclass
# ============================================================================


class TestJailbreakAttempt:
    """Tests for the JailbreakAttempt dataclass."""

    def test_create_attempt(self):
        """Test creating a JailbreakAttempt."""
        attempt = JailbreakAttempt(
            attempt_id="test_001",
            timestamp="2024-01-15T10:00:00",
            attack_type=AttackType.PROMPT_INJECTION,
            input_text="Test input",
            model_response="Test response",
            success=True,
            confidence_score=0.9,
            triggered_layers=[1, 2, 3],
            metadata={"key": "value"},
        )

        assert attempt.attempt_id == "test_001"
        assert attempt.attack_type == AttackType.PROMPT_INJECTION
        assert attempt.success is True
        assert attempt.confidence_score == 0.9
        assert attempt.triggered_layers == [1, 2, 3]

    def test_default_values(self):
        """Test default values for optional fields."""
        attempt = JailbreakAttempt(
            attempt_id="test_002",
            timestamp="2024-01-15T10:00:00",
            attack_type=AttackType.ROLE_PLAY,
            input_text="Test input",
            model_response="Test response",
            success=False,
            confidence_score=0.5,
        )

        assert attempt.triggered_layers == []
        assert attempt.metadata == {}


# ============================================================================
# Tests for BehaviorTrace dataclass
# ============================================================================


class TestBehaviorTrace:
    """Tests for the BehaviorTrace dataclass."""

    def test_create_trace(self):
        """Test creating a BehaviorTrace."""
        trace = BehaviorTrace(
            trace_id="trace_001",
            attempt_id="attempt_001",
            layer=5,
            activation_values=[0.1, 0.2, 0.3],
            neuron_indices=[10, 20, 30],
            timestamp="2024-01-15T10:00:00",
        )

        assert trace.trace_id == "trace_001"
        assert trace.layer == 5
        assert trace.activation_values == [0.1, 0.2, 0.3]
        assert trace.neuron_indices == [10, 20, 30]


# ============================================================================
# Tests for JailbreakTracer initialization
# ============================================================================


class TestJailbreakTracerInit:
    """Tests for JailbreakTracer initialization."""

    def test_default_initialization(self):
        """Test initialization with default data directory."""
        tracer = JailbreakTracer()
        assert tracer.data_dir == Path("./data")
        assert tracer.attempts == {}
        assert tracer.traces == {}

    def test_custom_data_directory(self):
        """Test initialization with custom data directory."""
        tracer = JailbreakTracer(data_dir="/custom/path")
        assert tracer.data_dir == Path("/custom/path")


# ============================================================================
# Tests for loading attempts
# ============================================================================


class TestLoadAttempts:
    """Tests for loading jailbreak attempts."""

    def test_load_attempts_success(self, test_attempts_file):
        """Test successful loading of attempts."""
        tracer = JailbreakTracer()
        attempts = tracer.load_attempts(test_attempts_file)

        assert len(attempts) == 5
        assert "attempt_001" in attempts
        assert "attempt_002" in attempts

    def test_load_attempts_file_not_found(self):
        """Test loading non-existent file raises FileNotFoundError."""
        tracer = JailbreakTracer()

        with pytest.raises(FileNotFoundError):
            tracer.load_attempts(Path("/nonexistent/file.json"))

    def test_load_attempts_parses_correctly(self, test_attempts_file):
        """Test that attempts are parsed with correct values."""
        tracer = JailbreakTracer()
        attempts = tracer.load_attempts(test_attempts_file)

        attempt = attempts["attempt_001"]
        assert attempt.attack_type == AttackType.PROMPT_INJECTION
        assert attempt.success is False
        assert attempt.confidence_score == 0.95
        assert attempt.triggered_layers == [3, 5, 7]

    def test_load_attempts_with_role_play(self, test_attempts_file):
        """Test loading role_play attack type."""
        tracer = JailbreakTracer()
        attempts = tracer.load_attempts(test_attempts_file)

        attempt = attempts["attempt_002"]
        assert attempt.attack_type == AttackType.ROLE_PLAY
        assert attempt.success is True
        assert attempt.confidence_score == 0.85


# ============================================================================
# Tests for loading traces
# ============================================================================


class TestLoadTraces:
    """Tests for loading behavior traces."""

    def test_load_traces_success(self, test_traces_file):
        """Test successful loading of traces."""
        tracer = JailbreakTracer()
        traces = tracer.load_traces(test_traces_file)

        assert len(traces) == 3
        assert "attempt_001" in traces
        assert "attempt_002" in traces

    def test_load_traces_file_not_found(self):
        """Test loading non-existent file raises FileNotFoundError."""
        tracer = JailbreakTracer()

        with pytest.raises(FileNotFoundError):
            tracer.load_traces(Path("/nonexistent/file.json"))

    def test_load_traces_parses_correctly(self, test_traces_file):
        """Test that traces are parsed with correct values."""
        tracer = JailbreakTracer()
        traces = tracer.load_traces(test_traces_file)

        attempt_traces = traces["attempt_001"]
        assert len(attempt_traces) == 3

        first_trace = attempt_traces[0]
        assert first_trace.layer == 3
        assert first_trace.activation_values == [0.5, 0.8, 0.3, 0.9, 0.2]
        assert first_trace.neuron_indices == [10, 25, 40, 55, 70]


# ============================================================================
# Tests for retrieving attempts
# ============================================================================


class TestGetAttempt:
    """Tests for retrieving jailbreak attempts."""

    def test_get_existing_attempt(self, tracer):
        """Test retrieving an existing attempt."""
        attempt = tracer.get_attempt("attempt_001")

        assert attempt is not None
        assert attempt.attempt_id == "attempt_001"

    def test_get_nonexistent_attempt(self, tracer):
        """Test retrieving a non-existent attempt returns None."""
        attempt = tracer.get_attempt("nonexistent_id")

        assert attempt is None


class TestGetSuccessfulAttempts:
    """Tests for getting successful jailbreak attempts."""

    def test_get_successful_attempts(self, tracer):
        """Test retrieving successful attempts."""
        successful = tracer.get_successful_attempts()

        assert len(successful) == 2
        assert all(a.success for a in successful)

    def test_successful_attempt_ids(self, tracer):
        """Test that correct attempts are marked as successful."""
        successful = tracer.get_successful_attempts()
        successful_ids = {a.attempt_id for a in successful}

        assert successful_ids == {"attempt_002", "attempt_004"}


class TestGetAttemptsByType:
    """Tests for filtering attempts by attack type."""

    def test_get_attempts_by_type(self, tracer):
        """Test filtering attempts by attack type."""
        prompt_injection = tracer.get_attempts_by_type(AttackType.PROMPT_INJECTION)

        assert len(prompt_injection) == 2
        assert all(
            a.attack_type == AttackType.PROMPT_INJECTION for a in prompt_injection
        )

    def test_get_attempts_by_type_role_play(self, tracer):
        """Test filtering role_play attacks."""
        role_play = tracer.get_attempts_by_type(AttackType.ROLE_PLAY)

        assert len(role_play) == 1
        assert role_play[0].attempt_id == "attempt_002"

    def test_get_attempts_by_type_none_found(self, tracer):
        """Test filtering by non-existent type returns empty list."""
        indirect = tracer.get_attempts_by_type(AttackType.INDIRECT_REFERENCE)

        assert len(indirect) == 0


# ============================================================================
# Tests for retrieving traces
# ============================================================================


class TestGetTrace:
    """Tests for retrieving behavior traces."""

    def test_get_existing_trace(self, tracer):
        """Test retrieving an existing trace."""
        trace = tracer.get_trace("attempt_001")

        assert trace is not None
        assert len(trace) == 3

    def test_get_nonexistent_trace(self, tracer):
        """Test retrieving a non-existent trace returns None."""
        trace = tracer.get_trace("nonexistent_id")

        assert trace is None


class TestGetActivatedLayers:
    """Tests for getting activated layers."""

    def test_get_activated_layers(self, tracer):
        """Test getting activated layers for an attempt."""
        layers = tracer.get_activated_layers("attempt_001")

        assert layers == {3, 5, 7}

    def test_get_activated_layers_no_trace(self, tracer):
        """Test getting activated layers when no trace exists."""
        layers = tracer.get_activated_layers("attempt_003")

        assert layers == set()


# ============================================================================
# Tests for layer statistics
# ============================================================================


class TestGetLayerStatistics:
    """Tests for getting layer statistics."""

    def test_get_layer_statistics(self, tracer):
        """Test getting statistics for a specific layer."""
        stats = tracer.get_layer_statistics("attempt_001", 3)

        assert stats is not None
        assert "mean_activation" in stats
        assert "std_activation" in stats
        assert "max_activation" in stats
        assert "min_activation" in stats
        assert "activated_neurons" in stats
        assert stats["max_activation"] == 0.9
        assert stats["min_activation"] == 0.2

    def test_get_layer_statistics_no_trace(self, tracer):
        """Test getting statistics when no trace exists."""
        stats = tracer.get_layer_statistics("attempt_003", 3)

        assert stats is None

    def test_get_layer_statistics_no_layer(self, tracer):
        """Test getting statistics for non-existent layer."""
        stats = tracer.get_layer_statistics("attempt_001", 99)

        assert stats is None


# ============================================================================
# Tests for attack analysis
# ============================================================================


class TestGetAttackSuccessRate:
    """Tests for calculating attack success rates."""

    def test_overall_success_rate(self, tracer):
        """Test calculating overall success rate."""
        rate = tracer.get_attack_success_rate()

        assert rate == 2 / 5  # 2 successful out of 5 attempts

    def test_success_rate_by_type(self, tracer):
        """Test calculating success rate for specific attack type."""
        rate = tracer.get_attack_success_rate(AttackType.PROMPT_INJECTION)

        # 2 prompt_injection attempts: attempt_001 (failed), attempt_004 (success)
        assert rate == 0.5

    def test_success_rate_no_attempts(self, empty_tracer):
        """Test success rate with no attempts returns 0.0."""
        rate = empty_tracer.get_attack_success_rate()

        assert rate == 0.0

    def test_success_rate_type_no_attempts(self, tracer):
        """Test success rate for type with no attempts."""
        rate = tracer.get_attack_success_rate(AttackType.INDIRECT_REFERENCE)

        assert rate == 0.0


class TestGetHighConfidenceAttempts:
    """Tests for getting high confidence attempts."""

    def test_get_high_confidence_default_threshold(self, tracer):
        """Test getting high confidence attempts with default threshold."""
        high_conf = tracer.get_high_confidence_attempts()

        assert len(high_conf) == 4
        assert all(a.confidence_score >= 0.8 for a in high_conf)

    def test_get_high_confidence_custom_threshold(self, tracer):
        """Test getting high confidence attempts with custom threshold."""
        high_conf = tracer.get_high_confidence_attempts(threshold=0.9)

        assert len(high_conf) == 2
        assert all(a.confidence_score >= 0.9 for a in high_conf)

    def test_get_high_confidence_no_matches(self, tracer):
        """Test getting high confidence with very high threshold."""
        high_conf = tracer.get_high_confidence_attempts(threshold=1.0)

        assert len(high_conf) == 0


class TestAnalyzeAttackPattern:
    """Tests for analyzing attack patterns."""

    def test_analyze_attack_pattern_prompt_injection(self, tracer):
        """Test analyzing prompt_injection pattern."""
        analysis = tracer.analyze_attack_pattern(AttackType.PROMPT_INJECTION)

        assert analysis["total_attempts"] == 2
        assert analysis["successful_attempts"] == 1
        assert analysis["success_rate"] == 0.5
        assert analysis["average_confidence"] == (0.95 + 0.88) / 2

    def test_analyze_attack_pattern_role_play(self, tracer):
        """Test analyzing role_play pattern."""
        analysis = tracer.analyze_attack_pattern(AttackType.ROLE_PLAY)

        assert analysis["total_attempts"] == 1
        assert analysis["successful_attempts"] == 1
        assert analysis["success_rate"] == 1.0

    def test_analyze_attack_pattern_no_attempts(self, tracer):
        """Test analyzing pattern with no attempts returns empty dict."""
        analysis = tracer.analyze_attack_pattern(AttackType.INDIRECT_REFERENCE)

        assert analysis == {}


# ============================================================================
# Tests for summary statistics
# ============================================================================


class TestGetSummaryStatistics:
    """Tests for getting summary statistics."""

    def test_get_summary_statistics(self, tracer):
        """Test getting summary statistics."""
        stats = tracer.get_summary_statistics()

        assert stats["total_attempts"] == 5
        assert stats["successful_attempts"] == 2
        assert stats["overall_success_rate"] == 0.4
        assert stats["traces_available"] == 3
        assert "attack_types" in stats

    def test_summary_attack_types(self, tracer):
        """Test attack types in summary."""
        stats = tracer.get_summary_statistics()

        assert "prompt_injection" in stats["attack_types"]
        assert "role_play" in stats["attack_types"]

        # Check counts and success rates
        assert stats["attack_types"]["prompt_injection"]["count"] == 2
        assert stats["attack_types"]["prompt_injection"]["success_rate"] == 0.5

    def test_summary_statistics_empty(self, empty_tracer):
        """Test summary statistics with empty tracer."""
        stats = empty_tracer.get_summary_statistics()

        assert stats == {}


# ============================================================================
# Tests for saving analysis
# ============================================================================


class TestSaveAnalysis:
    """Tests for saving analysis results."""

    def test_save_analysis(self, empty_tracer):
        """Test saving analysis to file."""
        analysis = {
            "test_key": "test_value",
            "test_number": 42,
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            output_path = Path(f.name)

        try:
            empty_tracer.save_analysis(output_path, analysis)

            assert output_path.exists()

            with open(output_path) as f:
                saved_data = json.load(f)

            assert saved_data["test_key"] == "test_value"
            assert saved_data["test_number"] == 42
        finally:
            output_path.unlink()

    def test_save_analysis_creates_directory(self, empty_tracer):
        """Test that save_analysis creates parent directories."""
        analysis = {"key": "value"}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "analysis.json"

            empty_tracer.save_analysis(output_path, analysis)

            assert output_path.exists()
            assert output_path.parent.exists()


# ============================================================================
# Tests for edge cases and error handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_load_attempts_with_invalid_json(self):
        """Test loading file with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            f.write("{ invalid json }")
            temp_path = Path(f.name)

        try:
            tracer = JailbreakTracer()
            with pytest.raises(json.JSONDecodeError):
                tracer.load_attempts(temp_path)
        finally:
            temp_path.unlink()

    def test_parse_attempts_handles_missing_fields(self):
        """Test that parsing handles missing optional fields gracefully."""
        raw_data = {
            "attempt_001": {
                "attack_type": "prompt_injection",
                "input_text": "test",
            }
        }

        tracer = JailbreakTracer()
        attempts = tracer._parse_attempts(raw_data)

        assert "attempt_001" in attempts

    def test_parse_attempts_with_invalid_attack_type(self):
        """Test that parsing with invalid attack type skips the attempt."""
        raw_data = {
            "attempt_001": {
                "attack_type": "unknown_type",
                "input_text": "test",
                "model_response": "response",
                "success": False,
                "confidence_score": 0.5,
            }
        }

        tracer = JailbreakTracer()
        attempts = tracer._parse_attempts(raw_data)

        # Invalid attack types are skipped (not added to the dictionary)
        assert "attempt_001" not in attempts

    def test_parse_traces_with_missing_fields(self):
        """Test that parsing traces handles missing optional fields."""
        raw_data = {
            "attempt_001": [
                {
                    "trace_id": "trace_001",
                    "layer": 1,
                }
            ]
        }

        tracer = JailbreakTracer()
        traces = tracer._parse_traces(raw_data)

        assert "attempt_001" in traces
        assert traces["attempt_001"][0].layer == 1


# ============================================================================
# Tests for internal methods
# ============================================================================


class TestInternalMethods:
    """Tests for internal helper methods."""

    def test_get_most_common_layer(self, tracer):
        """Test getting the most common layer."""
        attempts = tracer.get_attempts_by_type(AttackType.PROMPT_INJECTION)
        most_common = tracer._get_most_common_layer(attempts)

        # Layer 3 appears in both prompt_injection attempts
        assert most_common == 3

    def test_get_most_common_layer_no_layers(self, empty_tracer):
        """Test getting most common layer with no triggered layers."""
        most_common = empty_tracer._get_most_common_layer([])

        assert most_common is None
