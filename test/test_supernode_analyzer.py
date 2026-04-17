"""
Tests for SupernodeAnalyzer module.

This module contains comprehensive tests for the SupernodeAnalyzer class,
including tests for loading data, querying supernodes, and performing analysis.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from refusal_lens.supernode_analyzer import (
    NeuronPattern,
    SupernodeAnalyzer,
    SupernodeData,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def test_data_dir():
    """Return the path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture()
def test_supernodes_file(test_data_dir):
    """Return the path to test supernodes file."""
    return test_data_dir / "test_supernodes.json"


@pytest.fixture()
def analyzer(test_supernodes_file):
    """Create a SupernodeAnalyzer with loaded test data."""
    analyzer = SupernodeAnalyzer()
    analyzer.load_supernode_data(test_supernodes_file)
    return analyzer


@pytest.fixture()
def empty_analyzer():
    """Create an empty SupernodeAnalyzer."""
    return SupernodeAnalyzer()


# ============================================================================
# Tests for NeuronPattern dataclass
# ============================================================================


class TestNeuronPattern:
    """Tests for the NeuronPattern dataclass."""

    def test_create_neuron_pattern(self):
        """Test creating a NeuronPattern."""
        pattern = NeuronPattern(
            neuron_id=101,
            activation_strength=0.95,
            features=["harmful", "dangerous"],
            metadata={"layer": 5},
        )

        assert pattern.neuron_id == 101
        assert pattern.activation_strength == 0.95
        assert pattern.features == ["harmful", "dangerous"]
        assert pattern.metadata == {"layer": 5}

    def test_default_values(self):
        """Test creating neuron pattern with required fields."""
        pattern = NeuronPattern(
            neuron_id=201,
            activation_strength=0.5,
            features=[],
            metadata={},
        )

        assert pattern.neuron_id == 201
        assert pattern.activation_strength == 0.5


# ============================================================================
# Tests for SupernodeData dataclass
# ============================================================================


class TestSupernodeData:
    """Tests for the SupernodeData dataclass."""

    def test_create_supernode_data(self):
        """Test creating a SupernodeData."""
        import numpy as np

        neurons = [
            NeuronPattern(
                neuron_id=101, activation_strength=0.9, features=["test"], metadata={}
            )
        ]
        steering_vector = np.array([0.5, 0.6, 0.7])

        supernode = SupernodeData(
            supernode_id=1,
            neurons=neurons,
            steering_vector=steering_vector,
            activation_distribution={"mean": 0.6, "std": 0.1},
        )

        assert supernode.supernode_id == 1
        assert len(supernode.neurons) == 1
        assert supernode.steering_vector is not None
        assert supernode.activation_distribution == {"mean": 0.6, "std": 0.1}


# ============================================================================
# Tests for SupernodeAnalyzer initialization
# ============================================================================


class TestSupernodeAnalyzerInit:
    """Tests for SupernodeAnalyzer initialization."""

    def test_default_initialization(self):
        """Test initialization with default data directory."""
        analyzer = SupernodeAnalyzer()
        assert analyzer.data_dir == Path("./data")
        assert analyzer.supernodes == {}

    def test_custom_data_directory(self):
        """Test initialization with custom data directory."""
        analyzer = SupernodeAnalyzer(data_dir="/custom/path")
        assert analyzer.data_dir == Path("/custom/path")


# ============================================================================
# Tests for loading supernode data
# ============================================================================


class TestLoadSupernodeData:
    """Tests for loading supernode data."""

    def test_load_supernode_data_success(self, test_supernodes_file):
        """Test successful loading of supernode data."""
        analyzer = SupernodeAnalyzer()
        supernodes = analyzer.load_supernode_data(test_supernodes_file)

        assert len(supernodes) == 4
        assert 1 in supernodes
        assert 2 in supernodes
        assert 3 in supernodes
        assert 4 in supernodes

    def test_load_supernode_data_file_not_found(self):
        """Test loading non-existent file raises FileNotFoundError."""
        analyzer = SupernodeAnalyzer()

        with pytest.raises(FileNotFoundError):
            analyzer.load_supernode_data(Path("/nonexistent/file.json"))

    def test_load_supernode_parses_correctly(self, test_supernodes_file):
        """Test that supernodes are parsed with correct values."""
        analyzer = SupernodeAnalyzer()
        supernodes = analyzer.load_supernode_data(test_supernodes_file)

        supernode = supernodes[1]
        assert supernode.supernode_id == 1
        assert len(supernode.neurons) == 8
        assert supernode.steering_vector is not None
        assert supernode.activation_distribution == {"mean": 0.65, "std": 0.22}

    def test_load_supernode_parses_neurons(self, test_supernodes_file):
        """Test that neurons are parsed correctly."""
        analyzer = SupernodeAnalyzer()
        supernodes = analyzer.load_supernode_data(test_supernodes_file)

        neurons = supernodes[1].neurons
        assert neurons[0].neuron_id == 101
        assert neurons[0].activation_strength == 0.95
        assert neurons[0].features == ["harmful", "dangerous"]

    def test_load_supernode_without_steering_vector(self, test_supernodes_file):
        """Test loading supernode without steering vector."""
        analyzer = SupernodeAnalyzer()
        supernodes = analyzer.load_supernode_data(test_supernodes_file)

        supernode = supernodes[3]
        assert supernode.steering_vector is None
        assert supernode.activation_distribution is not None


# ============================================================================
# Tests for retrieving supernodes
# ============================================================================


class TestGetSupernode:
    """Tests for retrieving supernodes."""

    def test_get_existing_supernode(self, analyzer):
        """Test retrieving an existing supernode."""
        supernode = analyzer.get_supernode(1)

        assert supernode is not None
        assert supernode.supernode_id == 1

    def test_get_nonexistent_supernode(self, analyzer):
        """Test retrieving a non-existent supernode returns None."""
        supernode = analyzer.get_supernode(999)

        assert supernode is None


# ============================================================================
# Tests for getting top neurons
# ============================================================================


class TestGetTopNeurons:
    """Tests for getting top neurons by activation strength."""

    def test_get_top_neurons_default(self, analyzer):
        """Test getting top 10 neurons by default."""
        top_neurons = analyzer.get_top_neurons(1)

        assert len(top_neurons) == 8  # All 8 neurons
        # Verify they are sorted by activation strength descending
        for i in range(len(top_neurons) - 1):
            assert (
                top_neurons[i].activation_strength
                >= top_neurons[i + 1].activation_strength
            )

    def test_get_top_neurons_custom_k(self, analyzer):
        """Test getting top k neurons."""
        top_neurons = analyzer.get_top_neurons(1, top_k=3)

        assert len(top_neurons) == 3
        assert top_neurons[0].activation_strength == 0.95

    def test_get_top_neurons_nonexistent_supernode(self, analyzer):
        """Test getting top neurons for non-existent supernode."""
        top_neurons = analyzer.get_top_neurons(999)

        assert top_neurons == []

    def test_get_top_neurons_supernode_4(self, analyzer):
        """Test getting top neurons for supernode with 11 neurons."""
        top_neurons = analyzer.get_top_neurons(4, top_k=5)

        assert len(top_neurons) == 5
        # First neuron should have highest strength (0.99)
        assert top_neurons[0].activation_strength == 0.99


# ============================================================================
# Tests for analyzing steering vectors
# ============================================================================


class TestAnalyzeSteeringVector:
    """Tests for analyzing steering vectors."""

    def test_analyze_steering_vector(self, analyzer):
        """Test analyzing steering vector."""
        stats = analyzer.analyze_steering_vector(1)

        assert stats is not None
        assert "magnitude" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "dimension" in stats
        assert stats["dimension"] == 9

    def test_analyze_steering_vector_no_steering_vector(self, analyzer):
        """Test analyzing when no steering vector exists."""
        stats = analyzer.analyze_steering_vector(3)

        assert stats is None

    def test_analyze_steering_vector_nonexistent(self, analyzer):
        """Test analyzing non-existent supernode."""
        stats = analyzer.analyze_steering_vector(999)

        assert stats is None

    def test_steering_vector_statistics(self, analyzer):
        """Test steering vector statistics values."""
        stats = analyzer.analyze_steering_vector(1)

        # Check that statistics are calculated correctly
        assert stats["magnitude"] > 0
        assert stats["max"] == 0.9
        assert stats["min"] == 0.2


# ============================================================================
# Tests for feature distribution
# ============================================================================


class TestGetFeatureDistribution:
    """Tests for getting feature distribution."""

    def test_get_feature_distribution(self, analyzer):
        """Test getting feature distribution."""
        distribution = analyzer.get_feature_distribution(1)

        assert "harmful" in distribution
        assert "dangerous" in distribution
        assert distribution["harmful"] == 3  # appears in neurons 101, 103, 106

    def test_get_feature_distribution_sorted(self, analyzer):
        """Test that features are sorted by frequency."""
        distribution = analyzer.get_feature_distribution(1)

        # Should be sorted by count descending
        counts = list(distribution.values())
        assert counts == sorted(counts, reverse=True)

    def test_get_feature_distribution_nonexistent_supernode(self, analyzer):
        """Test getting distribution for non-existent supernode."""
        distribution = analyzer.get_feature_distribution(999)

        assert distribution == {}


# ============================================================================
# Tests for saving analysis
# ============================================================================


class TestSaveAnalysis:
    """Tests for saving analysis results."""

    def test_save_analysis(self, empty_analyzer):
        """Test saving analysis to file."""
        analysis = {
            "test_key": "test_value",
            "test_number": 42,
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            output_path = Path(f.name)

        try:
            empty_analyzer.save_analysis(output_path, analysis)

            assert output_path.exists()

            with open(output_path) as f:
                saved_data = json.load(f)

            assert saved_data["test_key"] == "test_value"
            assert saved_data["test_number"] == 42
        finally:
            output_path.unlink()

    def test_save_analysis_creates_directory(self, empty_analyzer):
        """Test that save_analysis creates parent directories."""
        analysis = {"key": "value"}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "analysis.json"

            empty_analyzer.save_analysis(output_path, analysis)

            assert output_path.exists()
            assert output_path.parent.exists()


# ============================================================================
# Tests for summary statistics
# ============================================================================


class TestGetSummaryStatistics:
    """Tests for getting summary statistics."""

    def test_get_summary_statistics(self, analyzer):
        """Test getting summary statistics."""
        stats = analyzer.get_summary_statistics()

        assert stats["total_supernodes"] == 4
        assert stats["total_neurons"] == 28  # 8 + 5 + 4 + 11
        assert stats["average_neurons_per_supernode"] == 7.0
        assert stats["supernode_ids"] == [1, 2, 3, 4]

    def test_summary_statistics_empty(self, empty_analyzer):
        """Test summary statistics with empty analyzer."""
        stats = empty_analyzer.get_summary_statistics()

        assert stats == {}


# ============================================================================
# Tests for edge cases and error handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_load_supernodes_with_invalid_json(self):
        """Test loading file with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            f.write("{ invalid json }")
            temp_path = Path(f.name)

        try:
            analyzer = SupernodeAnalyzer()
            with pytest.raises(json.JSONDecodeError):
                analyzer.load_supernode_data(temp_path)
        finally:
            temp_path.unlink()

    def test_parse_supernode_handles_missing_fields(self):
        """Test that parsing handles missing optional fields gracefully."""
        raw_data = {"1": {"neurons": [{"id": 101}]}}

        analyzer = SupernodeAnalyzer()
        supernodes = analyzer._parse_supernode_data(raw_data)

        assert 1 in supernodes
        assert supernodes[1].neurons[0].activation_strength == 0.0
        assert supernodes[1].neurons[0].features == []

    def test_parse_supernode_handles_invalid_neurons(self):
        """Test that parsing handles invalid neuron data gracefully."""
        raw_data = {"1": {"neurons": [{"invalid": "data"}]}}

        analyzer = SupernodeAnalyzer()
        supernodes = analyzer._parse_supernode_data(raw_data)

        # Invalid neurons cause the entire supernode to be skipped
        assert 1 not in supernodes

    def test_parse_supernode_invalid_supernode_id(self):
        """Test that parsing handles invalid supernode ID gracefully."""
        raw_data = {"invalid_id": {"neurons": []}}

        analyzer = SupernodeAnalyzer()
        supernodes = analyzer._parse_supernode_data(raw_data)

        # Should not raise, but may skip the invalid entry
        assert len(supernodes) == 0


# ============================================================================
# Tests for internal methods
# ============================================================================


class TestInternalMethods:
    """Tests for internal helper methods."""

    def test_parse_supernode_data(self, test_supernodes_file):
        """Test internal parsing method."""
        with open(test_supernodes_file) as f:
            raw_data = json.load(f)

        analyzer = SupernodeAnalyzer()
        supernodes = analyzer._parse_supernode_data(raw_data)

        assert len(supernodes) == 4
        assert 1 in supernodes
        assert 2 in supernodes
        assert 3 in supernodes
        assert 4 in supernodes
