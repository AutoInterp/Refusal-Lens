"""Tests for refusal_lens.data_loader — real data only, no mocks."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from refusal_lens.data_loader import (
    ContrastiveDataset,
    ContrastiveSplit,
    Instruction,
    create_contrastive_dataset,
    list_available_processed,
    list_available_splits,
    load_processed,
    load_split,
    DEFAULT_N_EVAL,
    DEFAULT_N_TRAIN,
    DEFAULT_SEED,
    _DEFAULT_DATASET_DIR,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(tmp: Path, n_harmful: int = 10, n_harmless: int = 10) -> Path:
    """Create a minimal dataset directory for testing."""
    splits = tmp / "splits"
    processed = tmp / "processed"
    splits.mkdir(parents=True)
    processed.mkdir(parents=True)

    harmful = [{"instruction": f"harmful prompt {i}", "category": None} for i in range(n_harmful)]
    harmless = [{"instruction": f"harmless prompt {i}", "category": None} for i in range(n_harmless)]

    (splits / "harmful_train.json").write_text(json.dumps(harmful))
    (splits / "harmless_train.json").write_text(json.dumps(harmless))

    # A processed file with categories
    advbench = [
        {"instruction": f"advbench prompt {i}", "category": "test_cat"} for i in range(5)
    ]
    (processed / "advbench.json").write_text(json.dumps(advbench))

    return tmp


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_default_n_train(self):
        assert DEFAULT_N_TRAIN == 128

    def test_default_n_eval(self):
        assert DEFAULT_N_EVAL == 64

    def test_default_seed(self):
        assert DEFAULT_SEED == 42

    def test_default_dataset_dir_points_to_repo(self):
        assert _DEFAULT_DATASET_DIR.name == "refusal_direction_dataset"
        assert _DEFAULT_DATASET_DIR.parent.name == "dataset"


# ---------------------------------------------------------------------------
# Instruction dataclass
# ---------------------------------------------------------------------------

class TestInstruction:
    def test_required_text(self):
        inst = Instruction(text="hello")
        assert inst.text == "hello"
        assert inst.category is None
        assert inst.source is None

    def test_all_fields(self):
        inst = Instruction(text="test", category="harmful", source="advbench")
        assert inst.category == "harmful"
        assert inst.source == "advbench"


# ---------------------------------------------------------------------------
# ContrastiveSplit
# ---------------------------------------------------------------------------

class TestContrastiveSplit:
    def test_equal_length(self):
        split = ContrastiveSplit(harmful=["a", "b"], harmless=["c", "d"])
        assert len(split) == 2

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            ContrastiveSplit(harmful=["a"], harmless=["b", "c"])

    def test_empty_split(self):
        split = ContrastiveSplit(harmful=[], harmless=[])
        assert len(split) == 0


# ---------------------------------------------------------------------------
# ContrastiveDataset
# ---------------------------------------------------------------------------

class TestContrastiveDataset:
    def test_defaults(self):
        train = ContrastiveSplit(harmful=["a"], harmless=["b"])
        eval_ = ContrastiveSplit(harmful=["c"], harmless=["d"])
        ds = ContrastiveDataset(train=train, eval=eval_)
        assert ds.seed == DEFAULT_SEED
        assert ds.metadata == {}

    def test_custom_seed_and_metadata(self):
        train = ContrastiveSplit(harmful=["a"], harmless=["b"])
        eval_ = ContrastiveSplit(harmful=["c"], harmless=["d"])
        ds = ContrastiveDataset(train=train, eval=eval_, seed=99, metadata={"key": "val"})
        assert ds.seed == 99
        assert ds.metadata["key"] == "val"


# ---------------------------------------------------------------------------
# load_split — with temp data
# ---------------------------------------------------------------------------

class TestLoadSplit:
    def test_loads_instructions(self, tmp_path):
        ds_dir = _make_dataset(tmp_path)
        result = load_split("harmful_train", dataset_dir=ds_dir)
        assert len(result) == 10
        assert all(isinstance(r, Instruction) for r in result)

    def test_text_content(self, tmp_path):
        ds_dir = _make_dataset(tmp_path)
        result = load_split("harmful_train", dataset_dir=ds_dir)
        assert result[0].text == "harmful prompt 0"
        assert result[3].text == "harmful prompt 3"

    def test_source_is_split_name(self, tmp_path):
        ds_dir = _make_dataset(tmp_path)
        result = load_split("harmful_train", dataset_dir=ds_dir)
        assert all(r.source == "harmful_train" for r in result)

    def test_missing_file_raises(self, tmp_path):
        ds_dir = _make_dataset(tmp_path)
        with pytest.raises(FileNotFoundError, match="not found"):
            load_split("nonexistent", dataset_dir=ds_dir)

    def test_category_preserved(self, tmp_path):
        """Split files with category=null should yield None."""
        ds_dir = _make_dataset(tmp_path)
        result = load_split("harmful_train", dataset_dir=ds_dir)
        assert result[0].category is None


# ---------------------------------------------------------------------------
# load_split — with real repo data
# ---------------------------------------------------------------------------

class TestLoadSplitRealData:
    def test_harmful_train_exists(self):
        result = load_split("harmful_train")
        assert len(result) == 260

    def test_harmless_train_exists(self):
        result = load_split("harmless_train")
        assert len(result) == 18793

    def test_harmful_val_has_categories(self):
        result = load_split("harmful_val")
        assert len(result) == 39
        # harmful_val items have non-null categories
        categorized = [r for r in result if r.category is not None]
        assert len(categorized) > 0

    def test_all_six_splits_load(self):
        for name in ["harmful_train", "harmful_val", "harmful_test",
                      "harmless_train", "harmless_val", "harmless_test"]:
            result = load_split(name)
            assert len(result) > 0, f"{name} is empty"


# ---------------------------------------------------------------------------
# load_processed — with temp data
# ---------------------------------------------------------------------------

class TestLoadProcessed:
    def test_loads_processed_file(self, tmp_path):
        ds_dir = _make_dataset(tmp_path)
        result = load_processed("advbench", dataset_dir=ds_dir)
        assert len(result) == 5

    def test_category_preserved(self, tmp_path):
        ds_dir = _make_dataset(tmp_path)
        result = load_processed("advbench", dataset_dir=ds_dir)
        assert result[0].category == "test_cat"

    def test_source_is_name(self, tmp_path):
        ds_dir = _make_dataset(tmp_path)
        result = load_processed("advbench", dataset_dir=ds_dir)
        assert all(r.source == "advbench" for r in result)

    def test_missing_raises(self, tmp_path):
        ds_dir = _make_dataset(tmp_path)
        with pytest.raises(FileNotFoundError, match="not found"):
            load_processed("nonexistent", dataset_dir=ds_dir)


# ---------------------------------------------------------------------------
# load_processed — real repo data
# ---------------------------------------------------------------------------

class TestLoadProcessedRealData:
    def test_advbench(self):
        result = load_processed("advbench")
        assert len(result) == 520

    def test_alpaca(self):
        result = load_processed("alpaca")
        assert len(result) > 0
        assert all(r.source == "alpaca" for r in result)

    def test_all_processed_files_load(self):
        names = list_available_processed()
        assert len(names) == 8
        for name in names:
            result = load_processed(name)
            assert len(result) > 0, f"{name} is empty"


# ---------------------------------------------------------------------------
# list_available_splits
# ---------------------------------------------------------------------------

class TestListAvailableSplits:
    def test_with_temp_dir(self, tmp_path):
        ds_dir = _make_dataset(tmp_path)
        result = list_available_splits(dataset_dir=ds_dir)
        assert result == ["harmful_train", "harmless_train"]

    def test_empty_dir(self, tmp_path):
        result = list_available_splits(dataset_dir=tmp_path)
        assert result == []

    def test_real_repo(self):
        result = list_available_splits()
        assert len(result) == 6
        assert "harmful_train" in result
        assert "harmless_train" in result

    def test_sorted(self):
        result = list_available_splits()
        assert result == sorted(result)


# ---------------------------------------------------------------------------
# list_available_processed
# ---------------------------------------------------------------------------

class TestListAvailableProcessed:
    def test_with_temp_dir(self, tmp_path):
        ds_dir = _make_dataset(tmp_path)
        result = list_available_processed(dataset_dir=ds_dir)
        assert result == ["advbench"]

    def test_empty_dir(self, tmp_path):
        result = list_available_processed(dataset_dir=tmp_path)
        assert result == []

    def test_real_repo(self):
        result = list_available_processed()
        assert "advbench" in result
        assert "alpaca" in result
        assert len(result) == 8

    def test_sorted(self):
        result = list_available_processed()
        assert result == sorted(result)


# ---------------------------------------------------------------------------
# create_contrastive_dataset — with temp data
# ---------------------------------------------------------------------------

class TestCreateContrastiveDatasetTemp:
    def test_basic_creation(self, tmp_path):
        ds_dir = _make_dataset(tmp_path, n_harmful=20, n_harmless=20)
        ds = create_contrastive_dataset(
            n_train=5, n_eval=3, seed=42, dataset_dir=ds_dir,
        )
        assert len(ds.train) == 5
        assert len(ds.eval) == 3

    def test_train_eval_disjoint(self, tmp_path):
        ds_dir = _make_dataset(tmp_path, n_harmful=20, n_harmless=20)
        ds = create_contrastive_dataset(
            n_train=5, n_eval=3, seed=42, dataset_dir=ds_dir,
        )
        # No overlap between train and eval harmful prompts
        assert set(ds.train.harmful).isdisjoint(set(ds.eval.harmful))

    def test_metadata(self, tmp_path):
        ds_dir = _make_dataset(tmp_path, n_harmful=20, n_harmless=20)
        ds = create_contrastive_dataset(
            n_train=5, n_eval=3, seed=42, dataset_dir=ds_dir,
        )
        assert ds.metadata["n_train"] == 5
        assert ds.metadata["n_eval"] == 3
        assert ds.metadata["n_harmful_total"] == 20
        assert ds.metadata["n_harmless_total"] == 20

    def test_not_enough_harmful_raises(self, tmp_path):
        ds_dir = _make_dataset(tmp_path, n_harmful=3, n_harmless=20)
        with pytest.raises(ValueError, match="harmful samples"):
            create_contrastive_dataset(
                n_train=5, n_eval=3, dataset_dir=ds_dir,
            )

    def test_not_enough_harmless_raises(self, tmp_path):
        ds_dir = _make_dataset(tmp_path, n_harmful=20, n_harmless=3)
        with pytest.raises(ValueError, match="harmless samples"):
            create_contrastive_dataset(
                n_train=5, n_eval=3, dataset_dir=ds_dir,
            )

    def test_seed_stored(self, tmp_path):
        ds_dir = _make_dataset(tmp_path, n_harmful=20, n_harmless=20)
        ds = create_contrastive_dataset(
            n_train=5, n_eval=3, seed=99, dataset_dir=ds_dir,
        )
        assert ds.seed == 99


# ---------------------------------------------------------------------------
# create_contrastive_dataset — real repo data
# ---------------------------------------------------------------------------

class TestCreateContrastiveDatasetReal:
    def test_default_sizes(self):
        ds = create_contrastive_dataset()
        assert len(ds.train) == 128
        assert len(ds.eval) == 64

    def test_deterministic_same_seed(self):
        ds1 = create_contrastive_dataset(seed=42)
        ds2 = create_contrastive_dataset(seed=42)
        assert ds1.train.harmful == ds2.train.harmful
        assert ds1.train.harmless == ds2.train.harmless
        assert ds1.eval.harmful == ds2.eval.harmful
        assert ds1.eval.harmless == ds2.eval.harmless

    def test_different_seed_gives_different_data(self):
        ds1 = create_contrastive_dataset(seed=42)
        ds2 = create_contrastive_dataset(seed=99)
        assert ds1.train.harmful != ds2.train.harmful

    def test_train_eval_no_overlap(self):
        ds = create_contrastive_dataset()
        assert set(ds.train.harmful).isdisjoint(set(ds.eval.harmful))
        assert set(ds.train.harmless).isdisjoint(set(ds.eval.harmless))

    def test_all_strings(self):
        ds = create_contrastive_dataset()
        for prompt in ds.train.harmful + ds.train.harmless:
            assert isinstance(prompt, str)
            assert len(prompt) > 0

    def test_custom_sizes(self):
        ds = create_contrastive_dataset(n_train=32, n_eval=16)
        assert len(ds.train) == 32
        assert len(ds.eval) == 16

    def test_max_capacity(self):
        """260 harmful samples → max is 260 total (train + eval)."""
        ds = create_contrastive_dataset(n_train=200, n_eval=60)
        assert len(ds.train) == 200
        assert len(ds.eval) == 60

    def test_exceeds_harmful_capacity_raises(self):
        with pytest.raises(ValueError, match="harmful"):
            create_contrastive_dataset(n_train=200, n_eval=100)

    def test_metadata_reflects_real_counts(self):
        ds = create_contrastive_dataset()
        assert ds.metadata["n_harmful_total"] == 260
        assert ds.metadata["n_harmless_total"] == 18793


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------

class TestPackageExports:
    def test_all_exported(self):
        import refusal_lens
        expected = [
            "ContrastiveDataset", "ContrastiveSplit", "Instruction",
            "create_contrastive_dataset", "list_available_processed",
            "list_available_splits", "load_processed", "load_split",
        ]
        for name in expected:
            assert hasattr(refusal_lens, name), f"{name} not exported"

    def test_in_all(self):
        import refusal_lens
        for name in ["ContrastiveDataset", "ContrastiveSplit", "Instruction",
                      "create_contrastive_dataset", "list_available_processed",
                      "list_available_splits", "load_processed", "load_split"]:
            assert name in refusal_lens.__all__, f"{name} not in __all__"
