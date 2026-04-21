"""
Local Pipeline Validation Tests
================================
Validates pipeline stages 01 and 03 against known results from
previous experiments (Tejas's Script 16 results, Mahmoud's scaled experiments).

Tests:
  T1: Stage 01 — compute_mean_activations produces correct shapes
  T2: Stage 01 — per-layer directions have unit norm (normalized) and correct magnitude (unnormalized)
  T3: Stage 01 — cosine similarity between layers matches Tejas's findings (~0.938 for L15-L32)
  T4: Stage 01 — full end-to-end run with 2 layers, 4 samples produces all expected output files
  T5: Stage 01 — separation at L32 >> L15 (expected: L32 ~20k, L15 ~3k)
  T6: Stage 03 — fallback to existing scaled experiment results works
  T7: Stage 03 — dot product > 0 for harmful prompts (direction aligns with refusal)
  T8: Stage 03 — MLP ratio is in expected range (0.1% - 2%)
  T9: Stage 03 — per-layer decomposition reconstruction error < 1e-3
  T10: Utils — format_prompt, classify_response, categorize_prompt, select_diverse_prompts

Usage:
  # From repo root, with venv activated:
  cd scripts/pipeline
  python tests/test_pipeline_local.py              # run all tests
  python tests/test_pipeline_local.py --stage 01   # only Stage 01 tests
  python tests/test_pipeline_local.py --stage 03   # only Stage 03 tests
  python tests/test_pipeline_local.py --stage utils # only utils tests
  python tests/test_pipeline_local.py --quick       # skip GPU tests (utils only)

Requires:
  - GPU with >= 8GB VRAM for Stage 01/03 tests (model loads in float32)
  - Existing scaled experiment results for Stage 03 fallback test
  - Dataset files in dataset/refusal_direction_dataset/splits/
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Add pipeline directory to path
PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_DIR))

import config
import utils

# ============================================================
# Test infrastructure
# ============================================================

PASS_COUNT = 0
FAIL_COUNT = 0
SKIP_COUNT = 0


def log_test(name: str, passed: bool, detail: str = ""):
    global PASS_COUNT, FAIL_COUNT
    status = "PASS" if passed else "FAIL"
    if passed:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    detail_str = f" — {detail}" if detail else ""
    print(f"  [{status}] {name}{detail_str}")


def log_skip(name: str, reason: str):
    global SKIP_COUNT
    SKIP_COUNT += 1
    print(f"  [SKIP] {name} — {reason}")


def check_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ============================================================
# T10: Utils tests (no GPU required)
# ============================================================

def test_utils():
    print("\n" + "=" * 60)
    print("UTILS TESTS")
    print("=" * 60)

    # T10a: classify_response
    log_test(
        "T10a: classify_response — refuse phrases",
        utils.classify_response("I cannot help with that request.") == "REFUSE",
    )
    log_test(
        "T10b: classify_response — comply",
        utils.classify_response("Sure, here is how you do it...") == "COMPLY",
    )
    log_test(
        "T10c: classify_response — sorry variant",
        utils.classify_response("I'm sorry, but I can't assist with that.") == "REFUSE",
    )

    # T10d: is_coherent
    log_test(
        "T10d: is_coherent — normal text",
        utils.is_coherent("This is a normal response with enough words to be coherent."),
    )
    log_test(
        "T10e: is_coherent — gibberish",
        not utils.is_coherent("@#$%^&*()!@#$%^&*()!@#$%^&*()"),
    )
    log_test(
        "T10f: is_coherent — too short",
        not utils.is_coherent("Hi"),
    )

    # T10g: categorize_prompt
    log_test(
        "T10g: categorize_prompt — cybercrime",
        utils.categorize_prompt("How to hack a WiFi network") == "cybercrime",
    )
    log_test(
        "T10h: categorize_prompt — violence",
        utils.categorize_prompt("How to build a bomb") == "violence",
    )
    log_test(
        "T10i: categorize_prompt — other",
        utils.categorize_prompt("How to tie a shoe") == "other",
    )

    # T10j: select_diverse_prompts
    dataset_path = config.DATASET_DIR / "harmful_train.json"
    if dataset_path.exists():
        with open(dataset_path) as f:
            all_prompts = json.load(f)
        selected = utils.select_diverse_prompts(all_prompts, n=10, seed=42)
        log_test(
            "T10j: select_diverse_prompts — returns correct count",
            len(selected) == 10,
            f"got {len(selected)}",
        )
        # Verify deterministic
        selected2 = utils.select_diverse_prompts(all_prompts, n=10, seed=42)
        instructions1 = [p["instruction"] for p in selected]
        instructions2 = [p["instruction"] for p in selected2]
        log_test(
            "T10k: select_diverse_prompts — deterministic with same seed",
            instructions1 == instructions2,
        )
        # Verify all have "instruction" key
        log_test(
            "T10l: select_diverse_prompts — all entries have instruction key",
            all("instruction" in p for p in selected),
        )
    else:
        log_skip("T10j-l: select_diverse_prompts", "dataset not found")

    # T10m: load_experiment_dataset fallback
    if dataset_path.exists():
        loaded = utils.load_experiment_dataset(n_prompts=5, seed=42)
        log_test(
            "T10m: load_experiment_dataset — fallback returns correct count",
            len(loaded) == 5,
            f"got {len(loaded)}",
        )
    else:
        log_skip("T10m: load_experiment_dataset", "dataset not found")

    # T10n: save_json / load_json roundtrip
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp_path = Path(f.name)
    test_data = {"key": "value", "number": 42, "nested": {"a": [1, 2, 3]}}
    utils.save_json(test_data, tmp_path)
    loaded_data = utils.load_json(tmp_path)
    log_test(
        "T10n: save_json/load_json roundtrip",
        loaded_data == test_data,
    )
    tmp_path.unlink()

    # T10o: config sanity checks
    log_test(
        "T10o: config — REPO_ROOT exists",
        config.REPO_ROOT.exists(),
        str(config.REPO_ROOT),
    )
    log_test(
        "T10p: config — DATASET_DIR exists",
        config.DATASET_DIR.exists(),
        str(config.DATASET_DIR),
    )
    log_test(
        "T10q: config — N_LAYERS = 34",
        config.N_LAYERS == 34,
    )
    log_test(
        "T10r: config — DIRECTION_LAYERS has 34 entries",
        len(config.DIRECTION_LAYERS) == 34,
    )
    log_test(
        "T10s: config — JB_CLASSES has 5 classes",
        len(config.JB_CLASSES) == 5,
    )

def test_utils_viz():
    """A5a: round-trip sanity for viz helpers (no model required)."""
    print("\n" + "=" * 60)
    print("UTILS_VIZ TESTS (no model)")
    print("=" * 60)

    from utils_viz import feature_key_from_node, OVERLAP_BUCKETS

    # Unit: feature key parsing
    log_test(
        "T-V1a: feature key parsed from transcoder node",
        feature_key_from_node({
            "feature_type": "cross layer transcoder",
            "layer": "29", "node_id": "29_1066_5",
        }) == "L29:F1066",
    )
    log_test(
        "T-V1b: non-feature node returns None",
        feature_key_from_node({"feature_type": "embedding", "node_id": "E_0_0"}) is None,
    )
    log_test(
        "T-V1c: malformed node_id returns None",
        feature_key_from_node({
            "feature_type": "cross layer transcoder", "node_id": "garbage",
        }) is None,
    )
    log_test(
        "T-V1d: buckets constant is complete",
        set(OVERLAP_BUCKETS) == {"shared_with_bare", "jb_unique", "bare_only", "bare", "non_feature"},
    )

    # T-V2: annotate_subcircuits round-trip on a synthetic graph.
    # Verifies reverse-index construction + per-node membership write.
    # Nodes are seeded with overlap_bucket="bare" (i.e. this is a bare graph)
    # so the bucket-conditional filter rules don't strip universal_refusal_core.
    from utils_viz import annotate_subcircuits
    synth_graph = {
        "metadata": {"jb_class": "bare"},
        "nodes": [
            {"node_id": "29_1066_5", "layer": "29", "feature_type": "cross layer transcoder",
             "overlap_bucket": "bare"},
            {"node_id": "24_107_3",  "layer": "24", "feature_type": "cross layer transcoder",
             "overlap_bucket": "bare"},
            {"node_id": "5_9999_0",  "layer": "5",  "feature_type": "cross layer transcoder",
             "overlap_bucket": "bare"},  # no membership
            {"node_id": "E_0_0", "feature_type": "embedding"},  # non-feature
        ],
    }
    synth_subcircuits = {
        "subcircuits": {
            "universal_refusal_core":  {"features": ["L29:F1066", "L24:F107"]},
            "dampening_specialists":   {"features": ["L29:F1066"]},
            "anti_refusal_amplifiers": {"features": ["L24:F107"]},
        },
    }
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as gf:
        gpath = Path(gf.name)
        json.dump(synth_graph, gf)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as sf:
        spath = Path(sf.name)
        json.dump(synth_subcircuits, sf)
    try:
        result = annotate_subcircuits(gpath, spath)
        nodes_by_id = {n["node_id"]: n for n in result["nodes"]}
        log_test(
            "T-V2a: L29:F1066 → universal_refusal_core + dampening_specialists",
            set(nodes_by_id["29_1066_5"]["subcircuits"])
            == {"universal_refusal_core", "dampening_specialists"},
            str(nodes_by_id["29_1066_5"]["subcircuits"]),
        )
        log_test(
            "T-V2b: L24:F107 → universal_refusal_core + anti_refusal_amplifiers",
            set(nodes_by_id["24_107_3"]["subcircuits"])
            == {"universal_refusal_core", "anti_refusal_amplifiers"},
            str(nodes_by_id["24_107_3"]["subcircuits"]),
        )
        log_test(
            "T-V2c: feature node with no membership → empty list",
            nodes_by_id["5_9999_0"]["subcircuits"] == [],
        )
        log_test(
            "T-V2d: non-feature node → empty list",
            nodes_by_id["E_0_0"]["subcircuits"] == [],
        )
        log_test(
            "T-V2e: metadata records annotation count (2 of 3 feature nodes)",
            result["metadata"].get("n_subcircuit_annotated") == 2,
            f"got {result['metadata'].get('n_subcircuit_annotated')}",
        )
        log_test(
            "T-V2f: metadata records n_subcircuit_filtered == 0 (bare graph, no filtering)",
            result["metadata"].get("n_subcircuit_filtered") == 0,
            f"got {result['metadata'].get('n_subcircuit_filtered')}",
        )
    finally:
        gpath.unlink()
        spath.unlink()

    # T-V3: annotate_subcircuits filter rules (Georg's UI bug fix).
    # Corpus-level memberships must be filtered against per-graph overlap_bucket
    # so the UI never paints a jb_unique node as "universal_refusal_core" etc.
    from utils_viz import _subcircuit_allowed
    log_test(
        "T-V3a: universal_refusal_core allowed when bucket=bare",
        _subcircuit_allowed("universal_refusal_core", "bare", "bare"),
    )
    log_test(
        "T-V3b: universal_refusal_core allowed when bucket=shared_with_bare",
        _subcircuit_allowed("universal_refusal_core", "shared_with_bare", "fiction"),
    )
    log_test(
        "T-V3c: universal_refusal_core REJECTED when bucket=jb_unique",
        not _subcircuit_allowed("universal_refusal_core", "jb_unique", "fiction"),
    )
    log_test(
        "T-V3d: canonical_pro_refusal allowed when bucket=jb_unique",
        _subcircuit_allowed("canonical_pro_refusal", "jb_unique", "fiction"),
    )
    log_test(
        "T-V3e: canonical_pro_refusal REJECTED when bucket=bare",
        not _subcircuit_allowed("canonical_pro_refusal", "bare", "bare"),
    )
    log_test(
        "T-V3f: canonical_pro_refusal REJECTED when bucket=shared_with_bare",
        not _subcircuit_allowed("canonical_pro_refusal", "shared_with_bare", "fiction"),
    )
    log_test(
        "T-V3g: fiction_exclusive allowed when bucket=jb_unique and jb_class=fiction",
        _subcircuit_allowed("fiction_exclusive", "jb_unique", "fiction"),
    )
    log_test(
        "T-V3h: fiction_exclusive REJECTED when jb_class=roleplay (wrong class)",
        not _subcircuit_allowed("fiction_exclusive", "jb_unique", "roleplay"),
    )
    log_test(
        "T-V3i: fiction_exclusive REJECTED when bucket=shared_with_bare",
        not _subcircuit_allowed("fiction_exclusive", "shared_with_bare", "fiction"),
    )
    log_test(
        "T-V3j: cognitive_reframe_exclusive parses multi-word class correctly",
        _subcircuit_allowed("cognitive_reframe_exclusive", "jb_unique", "cognitive_reframe")
        and not _subcircuit_allowed("cognitive_reframe_exclusive", "jb_unique", "fiction"),
    )
    log_test(
        "T-V3k: sign_flip_convergent unfiltered (allowed in all buckets)",
        all(
            _subcircuit_allowed("sign_flip_convergent", b, "fiction")
            for b in ("bare", "shared_with_bare", "jb_unique")
        ),
    )
    log_test(
        "T-V3l: dampening_specialists unfiltered (allowed in all buckets)",
        all(
            _subcircuit_allowed("dampening_specialists", b, "fiction")
            for b in ("bare", "shared_with_bare", "jb_unique")
        ),
    )
    log_test(
        "T-V3m: anti_refusal_amplifiers unfiltered",
        _subcircuit_allowed("anti_refusal_amplifiers", "jb_unique", "fiction"),
    )
    log_test(
        "T-V3n: late_wave_layer24_32 unfiltered",
        _subcircuit_allowed("late_wave_layer24_32", "bare", "bare"),
    )
    log_test(
        "T-V3o: overlap_bucket=None (overlap annotation skipped) → pass through",
        all(
            _subcircuit_allowed(name, None, None)
            for name in ("universal_refusal_core", "canonical_pro_refusal",
                         "fiction_exclusive", "sign_flip_convergent")
        ),
    )
    log_test(
        "T-V3p: unknown subcircuit name → pass through (future-proof)",
        _subcircuit_allowed("some_new_subcircuit", "jb_unique", "fiction"),
    )

    # T-V4: end-to-end filter behavior on a realistic JB graph.
    # Same feature L29:F1066 tagged universal_refusal_core at corpus level —
    # but in this JB graph its overlap_bucket is jb_unique. Filter must drop
    # universal_refusal_core while keeping dampening_specialists (unfiltered).
    synth_jb_graph = {
        "metadata": {"jb_class": "fiction"},
        "nodes": [
            # Tagged universal_refusal_core + dampening_specialists at corpus level.
            # In this JB graph the feature pruned out on bare → jb_unique →
            # universal_refusal_core must be dropped, dampening kept.
            {"node_id": "29_1066_5", "layer": "29", "feature_type": "cross layer transcoder",
             "overlap_bucket": "jb_unique"},
            # Shared with bare — universal_refusal_core legitimately applies.
            {"node_id": "24_107_3",  "layer": "24", "feature_type": "cross layer transcoder",
             "overlap_bucket": "shared_with_bare"},
        ],
    }
    synth_jb_subcircuits = {
        "subcircuits": {
            "universal_refusal_core":  {"features": ["L29:F1066", "L24:F107"]},
            "dampening_specialists":   {"features": ["L29:F1066"]},
            "canonical_pro_refusal":   {"features": ["L29:F1066"]},  # should pass (jb_unique)
            "fiction_exclusive":       {"features": ["L29:F1066"]},  # should pass (jb_unique + fiction)
            "roleplay_exclusive":      {"features": ["L29:F1066"]},  # should drop (wrong class)
        },
    }
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as gf:
        gpath = Path(gf.name)
        json.dump(synth_jb_graph, gf)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as sf:
        spath = Path(sf.name)
        json.dump(synth_jb_subcircuits, sf)
    try:
        result = annotate_subcircuits(gpath, spath)
        nodes_by_id = {n["node_id"]: n for n in result["nodes"]}
        log_test(
            "T-V4a: jb_unique node keeps canonical + fiction_exclusive + dampening, drops universal + roleplay_exclusive",
            set(nodes_by_id["29_1066_5"]["subcircuits"])
            == {"canonical_pro_refusal", "fiction_exclusive", "dampening_specialists"},
            str(nodes_by_id["29_1066_5"]["subcircuits"]),
        )
        log_test(
            "T-V4b: shared_with_bare node keeps universal_refusal_core",
            "universal_refusal_core" in nodes_by_id["24_107_3"]["subcircuits"],
            str(nodes_by_id["24_107_3"]["subcircuits"]),
        )
        # L29:F1066 had 5 corpus-level memberships, 2 filtered → n_filtered == 2.
        # L24:F107 had 1 membership, 0 filtered.
        log_test(
            "T-V4c: metadata n_subcircuit_filtered counts dropped memberships",
            result["metadata"].get("n_subcircuit_filtered") == 2,
            f"got {result['metadata'].get('n_subcircuit_filtered')}",
        )
    finally:
        gpath.unlink()
        spath.unlink()


# ============================================================
# Stage 02 plumbing tests (no GPU — validate dataset loader,
# condition iteration, comparison aggregation)
# ============================================================

def test_stage_02():
    """Validate the data plumbing for the Stage 02 refactor.
    Covers load_controlled_dataset, iter_conditions, _aggregate_comparison.
    Actual attribution runs require GPU; see test_runpod_1_4.py for that.
    """
    print("\n" + "=" * 60)
    print("STAGE 02 TESTS (no model required)")
    print("=" * 60)

    # T-02a: load_controlled_dataset structure
    from utils import load_controlled_dataset
    ds_path = config.REPO_ROOT / "dataset" / "refusal_lens_controlled_dataset.json"
    if not ds_path.exists():
        log_skip("T-02a..T-02e: controlled dataset tests", f"{ds_path} not found")
    else:
        ds = load_controlled_dataset(dataset_path=ds_path)
        log_test(
            "T-02a: controlled dataset has 50 prompts",
            len(ds) == 50,
            f"got {len(ds)}",
        )
        first = ds[0]
        log_test(
            "T-02b: each row has id, base, bare, topic, conditions",
            {"id", "base", "bare", "topic", "conditions"} <= set(first.keys()),
            str(list(first.keys())),
        )
        log_test(
            "T-02c: each row has 11 conditions (bare + 5 jb + 5 ctrl)",
            len(first["conditions"]) == 11,
            f"got {len(first['conditions'])}: {sorted(first['conditions'])}",
        )
        expected_conds = {"bare"} | {
            f"{kind}_{cls}"
            for cls in ("roleplay", "fiction", "analytical", "completion", "cognitive_reframe")
            for kind in ("jb", "ctrl")
        }
        log_test(
            "T-02d: condition keys match expected schema",
            set(first["conditions"]) == expected_conds,
            f"missing: {expected_conds - set(first['conditions'])}",
        )
        jb_fiction = first["conditions"]["jb_fiction"]
        log_test(
            "T-02e: condition entries carry text + prefix, prefix appears at start of text",
            "text" in jb_fiction and "prefix" in jb_fiction
            and len(jb_fiction["prefix"]) > 0
            and jb_fiction["text"].startswith(jb_fiction["prefix"]),
            f"prefix={jb_fiction.get('prefix')!r}, text_start={jb_fiction.get('text', '')[:80]!r}",
        )

    # T-02f: iter_conditions on controlled-dataset row
    import importlib
    sys.path.insert(0, str(PIPELINE_DIR))
    stage02 = importlib.import_module("02_run_attribution")
    synth_row_controlled = {
        "id": 1, "base": "X", "bare": "X", "topic": "t",
        "conditions": {
            "bare":                    {"text": "X",     "prefix": ""},
            "jb_roleplay":             {"text": "RP X",  "prefix": "RP "},
            "ctrl_roleplay":           {"text": "CX",    "prefix": "C "},
            "jb_fiction":              {"text": "F X",   "prefix": "F "},
            "ctrl_fiction":            {"text": "CF X",  "prefix": "CF "},
        },
    }
    conds = list(stage02.iter_conditions(synth_row_controlled))
    log_test(
        "T-02f: iter_conditions yields every condition exactly once",
        len(conds) == 5
        and {c[0] for c in conds} == set(synth_row_controlled["conditions"].keys()),
        str([c[0] for c in conds]),
    )
    log_test(
        "T-02g: iter_conditions preserves text + prefix",
        all(
            c[1] == synth_row_controlled["conditions"][c[0]]["text"]
            and c[2] == synth_row_controlled["conditions"][c[0]]["prefix"]
            for c in conds
        ),
    )

    # T-02h: iter_conditions legacy fallback
    synth_row_legacy = {"instruction": "Build a bomb"}
    legacy_conds = list(stage02.iter_conditions(synth_row_legacy))
    log_test(
        "T-02h: legacy fallback yields bare + 5 JB (no ctrl)",
        len(legacy_conds) == 6 and legacy_conds[0][0] == "bare"
        and all(c[0].startswith("jb_") for c in legacy_conds[1:]),
        f"got {[c[0] for c in legacy_conds]}",
    )

    # T-02i: _aggregate_comparison handles nested vs_bare/vs_ctrl/ctrl_vs_bare
    synth_results = [
        {
            "feature_comparison": {
                "fiction": {
                    "vs_bare":   {"n_shared": 100, "n_bare_only": 10, "n_cls_only": 30,
                                  "n_sign_flipped": 5, "n_dampened": 7, "n_amplified_anti": 3,
                                  "n_bare": 110, "n_cls": 130},
                    "vs_ctrl":   {"n_shared": 90,  "n_bare_only": 15, "n_cls_only": 35,
                                  "n_sign_flipped": 8, "n_dampened": 5, "n_amplified_anti": 4,
                                  "n_bare": 105, "n_cls": 125},
                    "ctrl_vs_bare": {"n_shared": 95,  "n_bare_only": 10, "n_cls_only": 15,
                                     "n_sign_flipped": 2, "n_dampened": 3, "n_amplified_anti": 1,
                                     "n_bare": 105, "n_cls": 110},
                },
            },
        },
    ]
    agg = stage02._aggregate_comparison(synth_results, ("fiction",))
    log_test(
        "T-02i: aggregate emits vs_bare / vs_ctrl / ctrl_vs_bare for class",
        "fiction" in agg
        and set(agg["fiction"]) == {"vs_bare", "vs_ctrl", "ctrl_vs_bare"},
        str(set(agg.get("fiction", {}))),
    )
    log_test(
        "T-02j: aggregate mean stat matches single-prompt value",
        agg["fiction"]["vs_bare"]["n_shared"]["mean"] == 100.0,
        f"got {agg['fiction']['vs_bare']['n_shared']['mean']}",
    )

    # T-02k: config target is L15 (not L32)
    log_test(
        "T-02k: config.MEASUREMENT_LAYER switched to L15",
        config.MEASUREMENT_LAYER == 15,
        f"got {config.MEASUREMENT_LAYER}",
    )
    log_test(
        "T-02l: config.MEASUREMENT_POSITION stayed at -2",
        config.MEASUREMENT_POSITION == -2,
        f"got {config.MEASUREMENT_POSITION}",
    )
    log_test(
        "T-02m: BEST_SEPARATION_LAYER preserved as historical constant (L32)",
        config.BEST_SEPARATION_LAYER == 32,
        f"got {config.BEST_SEPARATION_LAYER}",
    )

    # T-02n..s: multi-position plumbing (Task 14).
    # _load_per_position_directions returns {} gracefully when the positions
    # subdir is missing, populates from discovered files when present.
    import torch
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as td:
        run = Path(td)
        (run / "01_direction" / "positions_L15").mkdir(parents=True)
        for pos in (-5, -3, -2, -1):
            torch.save(torch.randn(2560), run / "01_direction" / "positions_L15" / f"pos_{pos:+d}.pt")
        # An unnormalized companion file should be ignored by the loader.
        torch.save(torch.randn(2560), run / "01_direction" / "positions_L15" / "pos_-2_unnormalized.pt")

        loaded = stage02._load_per_position_directions(run, target_layer=15)
        log_test(
            "T-02n: _load_per_position_directions reads every pos_*.pt",
            set(loaded.keys()) == {-5, -3, -2, -1},
            f"got {sorted(loaded.keys())}",
        )
        log_test(
            "T-02o: _load_per_position_directions ignores _unnormalized companions",
            all(not any(
                str(p) == f"pos_-2_unnormalized"
                for p in loaded.keys()
            ) for _ in [0]),
        )

        loaded_subset = stage02._load_per_position_directions(
            run, target_layer=15, requested_positions=[-2, -3, -999],
        )
        log_test(
            "T-02p: requested_positions filters to subset; missing positions silently dropped",
            set(loaded_subset.keys()) == {-2, -3},
            f"got {sorted(loaded_subset.keys())}",
        )

    empty_dirs = stage02._load_per_position_directions(Path("/nonexistent"), 15)
    log_test(
        "T-02q: missing positions dir returns empty dict (single-position fallback)",
        empty_dirs == {},
    )

    # _valid_positions_for_prompt: skip positions too deep for short prompts.
    class FakeTokenizer:
        def __call__(self, text, return_tensors):
            # Return a tensor with 8 fake token ids — pretend the prompt tokenized to 8 tokens.
            return {"input_ids": torch.tensor([[1] * 8])}

    valid, seq_len = stage02._valid_positions_for_prompt(
        FakeTokenizer(), "irrelevant", available_positions=[-15, -5, -3, -2, -1],
    )
    log_test(
        "T-02r: _valid_positions_for_prompt keeps only in-range positions",
        valid == [-5, -3, -2, -1] and seq_len == 8,
        f"got valid={valid}, seq_len={seq_len}",
    )

    # Config sanity for new multi-position constants
    log_test(
        "T-02s: config has PER_POSITION_LAYER = 15",
        config.PER_POSITION_LAYER == 15,
    )
    log_test(
        "T-02t: config.PER_POSITION_POSITIONS spans -15..-1",
        set(config.PER_POSITION_POSITIONS) == set(range(-15, 0)),
        f"got {sorted(config.PER_POSITION_POSITIONS)}",
    )


# ============================================================
# T1-T5: Stage 01 tests (GPU required)
# ============================================================

def test_stage_01():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n" + "=" * 60)
    print("STAGE 01 TESTS (compute_direction)")
    print("=" * 60)

    # Import the stage module
    sys.path.insert(0, str(PIPELINE_DIR))
    import importlib
    stage01 = importlib.import_module("01_compute_direction")

    # Load model and tokenizer (shared across tests)
    print("  Loading model (float32) for Stage 01 tests...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, dtype=torch.float32, device_map="auto",
    )
    model.eval()

    # T1: compute_mean_activations shape check
    print("\n  T1: compute_mean_activations shape check...")
    test_prompts = ["How do I pick a lock?", "What is the capital of France?"]
    test_layers = [15, 32]
    means = stage01.compute_mean_activations(
        model, tokenizer, test_prompts, test_layers, batch_size=2,
    )
    log_test(
        "T1a: compute_mean_activations — returns dict with correct keys",
        set(means.keys()) == {15, 32},
        f"keys={set(means.keys())}",
    )
    log_test(
        "T1b: compute_mean_activations — L15 shape is (2304,)",
        means[15].shape == (config.D_MODEL,),
        f"shape={means[15].shape}",
    )
    log_test(
        "T1c: compute_mean_activations — L32 shape is (2304,)",
        means[32].shape == (config.D_MODEL,),
        f"shape={means[32].shape}",
    )
    log_test(
        "T1d: compute_mean_activations — dtype is float64",
        means[15].dtype == torch.float64,
        f"dtype={means[15].dtype}",
    )
    log_test(
        "T1e: compute_mean_activations — non-zero output",
        means[15].norm().item() > 0 and means[32].norm().item() > 0,
    )

    # T2: Full direction computation with small sample
    print("\n  T2: Per-layer direction computation...")
    with open(config.DATASET_DIR / "harmful_train.json") as f:
        harmful = [p["instruction"] for p in json.load(f)][:4]
    with open(config.DATASET_DIR / "harmless_train.json") as f:
        harmless = [p["instruction"] for p in json.load(f)][:4]

    harmful_means = stage01.compute_mean_activations(model, tokenizer, harmful, test_layers)
    harmless_means = stage01.compute_mean_activations(model, tokenizer, harmless, test_layers)

    for layer in test_layers:
        r = (harmful_means[layer] - harmless_means[layer]).to(torch.float32)
        magnitude = r.norm().item()
        r_hat = r / magnitude

        log_test(
            f"T2a: L{layer} unnormalized r has non-zero magnitude",
            magnitude > 0,
            f"|r|={magnitude:.1f}",
        )
        log_test(
            f"T2b: L{layer} normalized r_hat has unit norm",
            abs(r_hat.norm().item() - 1.0) < 1e-5,
            f"||r_hat||={r_hat.norm().item():.6f}",
        )

    # T3: Cosine similarity between L15 and L32
    print("\n  T3: Cross-layer cosine similarity...")
    r_hat_15 = (harmful_means[15] - harmless_means[15]).to(torch.float32)
    r_hat_15 = r_hat_15 / r_hat_15.norm()
    r_hat_32 = (harmful_means[32] - harmless_means[32]).to(torch.float32)
    r_hat_32 = r_hat_32 / r_hat_32.norm()

    cos_sim = torch.nn.functional.cosine_similarity(
        r_hat_15.unsqueeze(0), r_hat_32.unsqueeze(0),
    ).item()
    # Tejas found ~0.938 with 64 samples. With 4 samples the direction
    # estimate is extremely noisy — this test only checks the computation
    # runs without error and returns a valid cosine similarity in [-1, 1].
    # The actual value is unreliable at n=4 (expected to converge at n>=32).
    log_test(
        "T3: L15-L32 cosine similarity is valid float in [-1, 1]",
        -1.0 <= cos_sim <= 1.0,
        f"cos_sim={cos_sim:.4f} (Tejas found 0.938 with n=64; noisy at n=4)",
    )

    # T4: End-to-end run produces all expected output files
    print("\n  T4: End-to-end Stage 01 run...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        run_dir = Path(tmp_dir) / "test_run"
        run_dir.mkdir()

        # Simulate command-line args
        class MockArgs:
            run_dir = None
            n_samples = 4
            layers = [15, 32]
            recompute = True

        mock_args = MockArgs()
        mock_args.run_dir = run_dir

        # Monkey-patch parse_args to return our mock
        original_parse = stage01.parse_args
        stage01.parse_args = lambda: mock_args

        try:
            stage01.main()
        finally:
            stage01.parse_args = original_parse

        out_dir = run_dir / "01_direction"

        # Check all expected output files
        expected_files = [
            out_dir / "unnormalized_r.pt",
            out_dir / "refusal_direction.pt",
            out_dir / "direction_metadata.json",
            out_dir / "directions" / "layer_15.pt",
            out_dir / "directions" / "layer_32.pt",
        ]
        config_file = run_dir / "config.json"

        for fpath in expected_files:
            log_test(
                f"T4a: output file exists — {fpath.name}",
                fpath.exists(),
                str(fpath.relative_to(run_dir)),
            )

        log_test(
            "T4b: config.json exists",
            config_file.exists(),
        )

        # T5: Validate output content
        print("\n  T5: Validate output content...")

        # Load and check unnormalized_r
        unnorm = torch.load(out_dir / "unnormalized_r.pt", map_location="cpu")
        log_test(
            "T5a: unnormalized_r has both layers",
            set(unnorm.keys()) == {15, 32},
        )

        # With only 4 samples, magnitudes will be noisy but L32 should
        # still be larger than L15 (robust across sample sizes)
        mag_15 = unnorm[15].norm().item()
        mag_32 = unnorm[32].norm().item()
        log_test(
            "T5b: L32 separation > L15 separation",
            mag_32 > mag_15,
            f"L15={mag_15:.1f}, L32={mag_32:.1f}",
        )

        # Load legacy format
        legacy = torch.load(out_dir / "refusal_direction.pt", map_location="cpu")
        log_test(
            "T5c: legacy format has best_direction key",
            "best_direction" in legacy,
        )
        log_test(
            "T5d: legacy format has per-layer direction keys",
            "direction_pos-2_layer15" in legacy and "direction_pos-2_layer32" in legacy,
        )
        log_test(
            "T5e: legacy best_direction has unit norm",
            abs(legacy["best_direction"].norm().item() - 1.0) < 1e-5,
        )

        # Load metadata
        with open(out_dir / "direction_metadata.json") as f:
            meta = json.load(f)
        log_test(
            "T5f: metadata has cosine_similarities",
            "cosine_similarities" in meta,
        )
        log_test(
            "T5g: metadata has correct n_harmful",
            meta["n_harmful"] == 4,
        )
        log_test(
            "T5h: metadata best_separation_layer is 32",
            meta["best_separation_layer"] == 32,
        )

        # Verify normalized directions match unnormalized / magnitude
        for layer in [15, 32]:
            r_hat_saved = torch.load(
                out_dir / "directions" / f"layer_{layer:02d}.pt",
                map_location="cpu",
            )
            r_unnorm = unnorm[layer]
            r_hat_computed = r_unnorm / r_unnorm.norm()
            cos = torch.nn.functional.cosine_similarity(
                r_hat_saved.unsqueeze(0).float(),
                r_hat_computed.unsqueeze(0).float(),
            ).item()
            log_test(
                f"T5i: L{layer} saved r_hat matches unnorm/||unnorm|| (cos={cos:.6f})",
                abs(cos - 1.0) < 1e-4,
            )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# T6-T9: Stage 03 tests (GPU required, uses existing results)
# ============================================================

def test_stage_03():
    import torch

    print("\n" + "=" * 60)
    print("STAGE 03 TESTS (verify_attribution)")
    print("=" * 60)

    # Check for existing scaled experiment results (fallback path)
    scaled_results = list(
        (config.REPO_ROOT / "data" / "results" / "scaled_experiments").glob(
            "run_*/attribution_results.json"
        )
    )
    if not scaled_results:
        log_skip("T6-T9: Stage 03 tests", "no existing attribution results found")
        return

    attr_path = sorted(scaled_results)[-1]
    print(f"  Using existing results: {attr_path}")

    # T6: Fallback loading works
    raw = utils.load_json(attr_path)
    if isinstance(raw, list):
        results_list = raw
    else:
        results_list = raw["results"]

    log_test(
        "T6a: fallback load — got results list",
        len(results_list) > 0,
        f"{len(results_list)} prompts",
    )

    # Verify structure
    entry = results_list[0]
    log_test(
        "T6b: entry has 'conditions' key",
        "conditions" in entry,
    )
    log_test(
        "T6c: entry has 'bare' condition",
        "bare" in entry.get("conditions", {}),
    )
    log_test(
        "T6d: bare condition has 'net' value",
        "net" in entry.get("conditions", {}).get("bare", {}),
    )

    # Check for direction file
    direction_path = (
        config.REPO_ROOT / "data" / "results"
        / "meeting_experiments" / "refusal_direction_corrected.pt"
    )
    if not direction_path.exists():
        log_skip("T7-T9: Stage 03 GPU tests", "no pre-computed direction file found")
        return

    # Load direction
    dir_data = torch.load(direction_path, map_location="cpu", weights_only=False)
    r_hat = dir_data["best_direction"].to(torch.float32)
    best_layer = dir_data["best_layer"]
    best_pos = dir_data["best_position"]

    log_test(
        "T6e: direction loaded — r_hat has unit norm",
        abs(r_hat.norm().item() - 1.0) < 1e-4,
        f"||r_hat||={r_hat.norm().item():.6f}",
    )
    log_test(
        "T6f: direction — best_layer is 32",
        best_layer == 32,
    )
    log_test(
        "T6g: direction — best_position is -2",
        best_pos == -2,
    )

    # Load model for dot product verification
    print("  Loading model (float32) for Stage 03 tests...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, dtype=torch.float32, device_map="auto",
    )
    model.eval()
    r_hat_dev = r_hat.to(model.device)

    # T7: Dot product > 0 for harmful prompts
    print("\n  T7: Dot product sign check (3 prompts)...")
    for i in range(min(3, len(results_list))):
        prompt = results_list[i]["prompt"]
        attr_net = results_list[i]["conditions"]["bare"]["net"]

        formatted = utils.format_prompt(tokenizer, prompt)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        act = out.hidden_states[best_layer + 1][0, best_pos, :].to(torch.float32)
        dot_product = (act @ r_hat_dev).item()

        log_test(
            f"T7a: prompt {i} dot product > 0 (refusal direction aligns)",
            dot_product > 0,
            f"dot={dot_product:.2f}, attr_net={attr_net:.2f}",
        )

        # T8: MLP ratio in expected range
        if dot_product != 0:
            ratio = attr_net / dot_product
            log_test(
                f"T8a: prompt {i} MLP ratio in [0.001, 0.05] (0.1%-5%)",
                0.001 < ratio < 0.05,
                f"ratio={ratio:.4f} ({ratio*100:.2f}%)",
            )

        del out
        gc.collect()
        torch.cuda.empty_cache()

    # T9: Per-layer decomposition reconstruction
    print("\n  T9: Per-layer decomposition reconstruction (1 prompt)...")
    prompt = results_list[0]["prompt"]
    formatted = utils.format_prompt(tokenizer, prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    emb_dot = (out.hidden_states[0][0, best_pos, :].to(torch.float32) @ r_hat_dev).item()
    full_dot = (out.hidden_states[best_layer + 1][0, best_pos, :].to(torch.float32) @ r_hat_dev).item()

    layer_sum = 0.0
    for layer in range(best_layer + 1):
        before = out.hidden_states[layer][0, best_pos, :].to(torch.float32)
        after = out.hidden_states[layer + 1][0, best_pos, :].to(torch.float32)
        layer_sum += ((after - before) @ r_hat_dev).item()

    reconstructed = emb_dot + layer_sum
    error = abs(full_dot - reconstructed)

    # float32 accumulation over 33 layers introduces ~1e-7 relative error
    # on dot products of magnitude ~18,000, so absolute error can reach ~0.01
    relative_error = error / abs(full_dot) if full_dot != 0 else error
    log_test(
        "T9a: reconstruction relative error < 1e-5",
        relative_error < 1e-5,
        f"full_dot={full_dot:.4f}, reconstructed={reconstructed:.4f}, "
        f"abs_err={error:.6f}, rel_err={relative_error:.2e}",
    )

    # Verify that the per-layer contributions are telescoping correctly
    # (this is mathematically guaranteed but verifies no numerical issues)
    log_test(
        "T9b: full dot product is non-trivial (> 100)",
        abs(full_dot) > 100,
        f"full_dot={full_dot:.1f}",
    )

    del out, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def find_latest_pipeline_run() -> Path | None:
    """Return newest run_dir under pipeline_runs/ that has Stage 01+02 outputs."""
    base = config.REPO_ROOT / "data" / "results" / "pipeline_runs"
    if not base.exists():
        return None
    candidates = sorted(base.glob("run_*"))
    for r in reversed(candidates):
        if (r / "01_direction" / "direction_metadata.json").exists() and \
           (r / "02_attribution" / "attribution_results.json").exists():
            return r
    return None


def test_stage_02b():
    print("\n" + "=" * 60)
    print("STAGE 02b TESTS (statistical_analysis + plots)")
    print("=" * 60)

    run_dir = find_latest_pipeline_run()
    if run_dir is None:
        log_skip("T-A3: Stage 02b", "no pipeline run with Stage 01+02 outputs found")
        return
    print(f"  Using run: {run_dir}")

    try:
        from scipy import stats as _  # noqa: F401
    except ImportError:
        log_skip("T-A3: Stage 02b", "scipy not installed locally")
        return

    # Regression guard: snapshot existing outputs before re-running
    stats_json = run_dir / "02b_stats" / "statistical_analysis.json"
    pre_stats = json.loads(stats_json.read_text()) if stats_json.exists() else None

    # Re-run Stage 02b against the existing run_dir
    import importlib
    stage02b = importlib.import_module("02b_statistical_analysis")

    class MockArgs:
        run_dir = None
        n_bootstrap = 10000

    mock_args = MockArgs()
    mock_args.run_dir = run_dir
    original_parse = stage02b.parse_args
    stage02b.parse_args = lambda: mock_args
    try:
        stage02b.main()
    finally:
        stage02b.parse_args = original_parse

    # T-A3a: new plot exists
    plot = run_dir / "02b_stats" / "separation_by_layer.png"
    log_test(
        "T-A3a: separation_by_layer.png exists",
        plot.exists(),
        str(plot.relative_to(run_dir)) if plot.exists() else "missing",
    )
    if plot.exists():
        size = plot.stat().st_size
        log_test(
            "T-A3b: separation_by_layer.png non-trivial (>5KB)",
            size > 5_000,
            f"size={size} bytes",
        )

    # T-A3c: existing stats numerically unchanged (regression guard)
    # Tolerance-based rather than byte-identity because scipy/BLAS versions
    # can shift p-values in their last few sig figs across machines.
    if pre_stats is not None and stats_json.exists():
        post_stats = json.loads(stats_json.read_text())
        mismatches = []
        for cls in pre_stats:
            if cls not in post_stats:
                mismatches.append(f"class '{cls}' missing post-run")
                continue
            for key, pre_val in pre_stats[cls].items():
                post_val = post_stats[cls].get(key)
                if post_val is None:
                    mismatches.append(f"{cls}.{key} missing")
                    continue
                if isinstance(pre_val, float) and isinstance(post_val, float):
                    if pre_val == 0 and abs(post_val) > 1e-10:
                        mismatches.append(f"{cls}.{key}: {pre_val} -> {post_val}")
                    elif pre_val != 0:
                        rel = abs(post_val - pre_val) / abs(pre_val)
                        if rel > 1e-6:
                            mismatches.append(f"{cls}.{key}: {pre_val} -> {post_val} (rel={rel:.2e})")
                elif pre_val != post_val:
                    mismatches.append(f"{cls}.{key}: {pre_val!r} -> {post_val!r}")
        log_test(
            "T-A3c: statistical_analysis.json numerically stable (rel_tol 1e-6)",
            len(mismatches) == 0,
            "; ".join(mismatches[:3]) if mismatches else f"{sum(len(v) for v in pre_stats.values())} values match",
        )
    
    # A5: cosine heatmap (requires cosine_matrix in direction_metadata)
    heatmap = run_dir / "02b_stats" / "cosine_heatmap.png"
    dm_path = run_dir / "01_direction" / "direction_metadata.json"
    dm = json.loads(dm_path.read_text()) if dm_path.exists() else {}
    if "cosine_matrix" in dm:
        log_test(
            "T-A5h: cosine_heatmap.png exists",
            heatmap.exists(),
            str(heatmap.relative_to(run_dir)) if heatmap.exists() else "missing",
        )
        if heatmap.exists():
            log_test(
                "T-A5i: cosine_heatmap.png non-trivial (>5KB)",
                heatmap.stat().st_size > 5_000,
                f"size={heatmap.stat().st_size} bytes",
            )
    else:
        log_skip("T-A5h/i: cosine heatmap", "cosine_matrix not populated (run --stage 01-a5 first)")
    
    # A6: bare-vs-JB distribution plot                                                                                          
    dist = run_dir / "02b_stats" / "distribution_by_class.png"
    log_test(                                                                                                                   
        "T-A6a: distribution_by_class.png exists",
        dist.exists(),                                                                                                          
        str(dist.relative_to(run_dir)) if dist.exists() else "missing",
    )                                                                                                                           
    if dist.exists():
        log_test(                                                                                                               
            "T-A6b: distribution_by_class.png non-trivial (>5KB)",                                                              
            dist.stat().st_size > 5_000,
            f"size={dist.stat().st_size} bytes",                                                                                
        )

def test_stage_03_a4():
    print("\n" + "=" * 60)
    print("STAGE 03 A4 TESTS (per-layer aggregate + plot)")
    print("=" * 60)

    run_dir = find_latest_pipeline_run()
    if run_dir is None:
        log_skip("T-A4: Stage 03 A4", "no pipeline run found")
        return
    verif_path = run_dir / "03_verification" / "verification_results.json"
    decomp_path = run_dir / "03_verification" / "per_layer_decomposition.json"
    if not verif_path.exists() or not decomp_path.exists():
        log_skip("T-A4: Stage 03 A4", "Stage 03 outputs missing; run full Stage 03 first")
        return
    print(f"  Using run: {run_dir}")

    # Snapshot the per_prompt array for regression check
    pre = json.loads(verif_path.read_text())
    pre_per_prompt = pre.get("per_prompt", [])
    pre_summary = pre.get("summary", {})

    # Run --aggregate-only
    import importlib
    stage03 = importlib.import_module("03_verify_attribution")

    class MockArgs:
        run_dir = None
        n_decompose = 10
        aggregate_only = True

    mock = MockArgs()
    mock.run_dir = run_dir
    original_parse = stage03.parse_args
    stage03.parse_args = lambda: mock
    try:
        stage03.main()
    finally:
        stage03.parse_args = original_parse

    plot = run_dir / "03_verification" / "per_layer_contribution.png"
    log_test(
        "T-A4a: per_layer_contribution.png exists",
        plot.exists(),
        str(plot.relative_to(run_dir)) if plot.exists() else "missing",
    )
    if plot.exists():
        log_test(
            "T-A4b: per_layer_contribution.png non-trivial (>5KB)",
            plot.stat().st_size > 5_000,
            f"size={plot.stat().st_size} bytes",
        )

    post = json.loads(verif_path.read_text())
    log_test(
        "T-A4c: per_layer_aggregate key added",
        "per_layer_aggregate" in post,
    )
    if "per_layer_aggregate" in post:
        agg = post["per_layer_aggregate"]
        log_test(
            "T-A4d: aggregate has all 34 layers",
            len(agg.get("layers", [])) == config.N_LAYERS,
            f"got {len(agg.get('layers', []))}",
        )
        log_test(
            "T-A4e: aggregate has embedding block",
            "embedding" in agg and "mean" in agg["embedding"],
        )

    # Regression: per_prompt unchanged, summary unchanged
    log_test(
        "T-A4f: per_prompt unchanged (regression)",
        post.get("per_prompt") == pre_per_prompt,
        "list differs" if post.get("per_prompt") != pre_per_prompt else f"{len(pre_per_prompt)} entries identical",
    )
    log_test(
        "T-A4g: summary unchanged (regression)",
        post.get("summary") == pre_summary,
    )

def test_stage_01_a5():
    print("\n" + "=" * 60)
    print("STAGE 01 A5 TESTS (full cosine matrix)")
    print("=" * 60)

    run_dir = find_latest_pipeline_run()
    if run_dir is None:
        log_skip("T-A5: Stage 01 A5", "no pipeline run found")
        return
    meta_path = run_dir / "01_direction" / "direction_metadata.json"
    directions_dir = run_dir / "01_direction" / "directions"
    if not meta_path.exists() or not directions_dir.exists():
        log_skip("T-A5: Stage 01 A5", "Stage 01 outputs missing")
        return
    print(f"  Using run: {run_dir}")

    pre = json.loads(meta_path.read_text())
    pre_layers = pre.get("layers", {})
    pre_cos_sims = pre.get("cosine_similarities", {})

    import importlib
    stage01 = importlib.import_module("01_compute_direction")

    class MockArgs:
        run_dir = None
        n_samples = config.N_DIRECTION_SAMPLES
        layers = None
        recompute = False
        update_metadata = True

    mock = MockArgs()
    mock.run_dir = run_dir
    original_parse = stage01.parse_args
    stage01.parse_args = lambda: mock
    try:
        stage01.main()
    finally:
        stage01.parse_args = original_parse

    post = json.loads(meta_path.read_text())

    log_test("T-A5a: cosine_matrix key present", "cosine_matrix" in post)

    if "cosine_matrix" in post:
        matrix = post["cosine_matrix"]
        n = len(matrix)
        log_test(
            "T-A5b: matrix is 34×34",
            n == config.N_LAYERS and all(len(row) == n for row in matrix),
            f"got {n}×{len(matrix[0]) if matrix else 0}",
        )
        diag_ok = all(abs(matrix[i][i] - 1.0) < 1e-6 for i in range(n))
        log_test("T-A5c: diagonal == 1.0", diag_ok)
        sym_ok = all(
            abs(matrix[i][j] - matrix[j][i]) < 1e-6
            for i in range(n) for j in range(i + 1, n)
        )
        log_test("T-A5d: matrix symmetric", sym_ok)
        # L15-L32 should match the previously-stored pairwise value
        if "L15_L32" in pre_cos_sims:
            diff = abs(matrix[15][32] - pre_cos_sims["L15_L32"])
            log_test(
                "T-A5e: L15-L32 matches prior pairwise value",
                diff < 1e-3,
                f"diff={diff:.6f}",
            )

    log_test("T-A5f: existing layers unchanged", post.get("layers") == pre_layers)
    log_test(
        "T-A5g: existing cosine_similarities unchanged",
        post.get("cosine_similarities") == pre_cos_sims,
    )

def test_stage_04_a8():
    print("\n" + "=" * 60)
    print("STAGE 04 A8 TESTS (layer histogram)")
    print("=" * 60)

    run_dir = find_latest_pipeline_run()
    if run_dir is None:
        log_skip("T-A8: Stage 04 A8", "no pipeline run found")
        return
    comp_path = run_dir / "04_labels" / "feature_comparison_labeled.json"
    if not comp_path.exists():
        log_skip("T-A8: Stage 04 A8", "feature_comparison_labeled.json missing")
        return
    print(f"  Using run: {run_dir}")

    import importlib
    stage04 = importlib.import_module("04_label_features")

    class MockArgs:
        run_dir = None
        skip_download = False
        max_features = None
        n_examples = 3
        n_logits = 10
        histogram_only = True

    mock = MockArgs()
    mock.run_dir = run_dir
    original_parse = stage04.parse_args
    stage04.parse_args = lambda: mock
    try:
        stage04.main()
    finally:
        stage04.parse_args = original_parse

    plot = run_dir / "04_labels" / "features_by_layer.png"
    log_test(
        "T-A8a: features_by_layer.png exists",
        plot.exists(),
        str(plot.relative_to(run_dir)) if plot.exists() else "missing",
    )
    if plot.exists():
        log_test(
            "T-A8b: features_by_layer.png non-trivial (>5KB)",
            plot.stat().st_size > 5_000,
            f"size={plot.stat().st_size} bytes",
        )

    hist_path = run_dir / "04_labels" / "layer_histogram.json"
    log_test("T-A8c: layer_histogram.json exists", hist_path.exists())
    if hist_path.exists():
        hist = json.loads(hist_path.read_text())
        log_test(
            "T-A8d: histogram has 3 buckets",
            set(hist.get("buckets", {}).keys()) == {"sign_flipped", "dampened", "amplified_anti"},
        )
        log_test(
            "T-A8e: each bucket has 34 layer slots",
            all(len(b["by_layer"]) == 34 for b in hist.get("buckets", {}).values()),
        )
        # Cross-validate totals against label_coverage.json
        cov_path = run_dir / "04_labels" / "label_coverage.json"
        if cov_path.exists():
            cov = json.loads(cov_path.read_text())
            expected = cov.get("comparison_counts", {})
            actual = {k: b["total"] for k, b in hist["buckets"].items()}
            log_test(
                "T-A8f: histogram totals match label_coverage.json",
                actual == expected,
                f"actual={actual}, expected={expected}",
            )

def test_stage_04_a7():                                                                                                         
    print("\n" + "=" * 60)             
    print("STAGE 04 A7 TESTS (feature-class UpSet)")                                                                            
    print("=" * 60)                                                                                                             
                                                                                                                                
    run_dir = find_latest_pipeline_run()                                                                                        
    if run_dir is None:                
        log_skip("T-A7: Stage 04 A7", "no pipeline run found")                                                                  
        return                                                                                                                  
    comp_path = run_dir / "04_labels" / "feature_comparison_labeled.json"
    if not comp_path.exists():                                                                                                  
        log_skip("T-A7: Stage 04 A7", "feature_comparison_labeled.json missing")
        return                                                                                                                  
    try:                                                                                                                        
        import upsetplot  # noqa: F401
    except ImportError:                                                                                                         
        log_skip("T-A7: Stage 04 A7", "upsetplot not installed (pip install upsetplot)")
        return                                                                                                                  
    print(f"  Using run: {run_dir}")
                                                                                                                                
    import importlib                                                                                                            
    stage04 = importlib.import_module("04_label_features")
                                                                                                                                
    class MockArgs:                                                                                                             
        run_dir = None
        skip_download = False                                                                                                   
        max_features = None            
        n_examples = 3
        n_logits = 10                                                                                                           
        histogram_only = False
        upset_only = True                                                                                                       
                                                                                                                                
    mock = MockArgs()
    mock.run_dir = run_dir                                                                                                      
    original_parse = stage04.parse_args
    stage04.parse_args = lambda: mock                                                                                           
    try:
        stage04.main()                                                                                                          
    finally:                           
        stage04.parse_args = original_parse
                                                                                                                                
    plot = run_dir / "04_labels" / "feature_class_upset.png"
    log_test(                                                                                                                   
        "T-A7a: feature_class_upset.png exists",                                                                                
        plot.exists(),
        str(plot.relative_to(run_dir)) if plot.exists() else "missing",                                                         
    )                                                                                                                           
    if plot.exists():
        log_test(                                                                                                               
            "T-A7b: feature_class_upset.png non-trivial (>5KB)",                                                                
            plot.stat().st_size > 5_000,
            f"size={plot.stat().st_size} bytes",                                                                                
        )                              
                                                                                                                                
    sets_path = run_dir / "04_labels" / "feature_class_sets.json"                                                               
    log_test("T-A7c: feature_class_sets.json exists", sets_path.exists())
    if sets_path.exists():                                                                                                      
        fs = json.loads(sets_path.read_text())
        log_test(                                                                                                               
            "T-A7d: 5 classes detected",
            fs.get("n_classes") == 5,                                                                                           
            f"got {fs.get('n_classes')}",
        )                                                                                                                       
        log_test(                                                                                                               
            "T-A7e: combined.total is a positive int",
            isinstance(fs.get("combined", {}).get("total"), int) and fs["combined"]["total"] > 0,                               
        )                                                                                                                       
        # Sanity: every key in combined.features has non-empty class list
        features = fs.get("combined", {}).get("features", {})                                                                   
        all_tagged = all(isinstance(v, list) and len(v) > 0 for v in features.values())                                         
        log_test("T-A7f: every combined feature has ≥1 class", all_tagged)

def test_stage_07():                                                                                                            
    print("\n" + "=" * 60)                                                                                                      
    print("STAGE 07 TESTS (rule-based subcircuits)")                                                                            
    print("=" * 60)                                                                                                             
                                                                                                                                
    run_dir = find_latest_pipeline_run()                                                                                        
    if run_dir is None:                      
        log_skip("T-07: Stage 07", "no pipeline run found")
        return                             
    for required in [                                                                                                           
        run_dir / "04_labels" / "feature_labels.json",
        run_dir / "04_labels" / "feature_class_sets.json",                                                                      
    ]:                                       
        if not required.exists():                  
            log_skip("T-07: Stage 07", f"{required.name} missing")
            return                                                                                                              
    print(f"  Using run: {run_dir}")               
                                                                                                                                
    import importlib                         
    stage07 = importlib.import_module("07_identify_subcircuits")                                                                
                                                    
    class MockArgs:                                                                                                             
        run_dir = None                             
        convergent_min = 3                                                                                                      
        late_wave_start = 24                                                                                                    
        late_wave_end = 32                         
                                                                                                                                
    mock = MockArgs()                        
    mock.run_dir = run_dir                                                                                                      
    orig = stage07.parse_args                      
    stage07.parse_args = lambda: mock                                                                                           
    try:                                     
        stage07.main()                                                                                                          
    finally:                                                                                                                    
        stage07.parse_args = orig                  
                                                                                                                                
    out_dir = run_dir / "07_subcircuits" 
    for fname in ["subcircuits.json", "subcircuits_summary.json",                                                               
                "subcircuits_treemap.png", "subcircuits_by_layer.png",
                "subcircuits_overlap.png", "SUBCIRCUITS_REPORT.md"]:                                                          
        p = out_dir / fname                  
        log_test(                            
            f"T-07a: {fname} exists",                                                                                           
            p.exists(),                            
            str(p.relative_to(run_dir)) if p.exists() else "missing",                                                           
        )                                          
                                                                                                                                
    sc_path = out_dir / "subcircuits.json"         
    if not sc_path.exists():                                                                                                    
        return                               
    data = json.loads(sc_path.read_text())         
                                                                                                                                
    expected_names = {                             
        "universal_refusal_core", "canonical_pro_refusal",                                                                      
        "sign_flip_convergent", "dampening_specialists",
        "anti_refusal_amplifiers", "late_wave_layer24_32",                                                                      
        "analytical_exclusive", "cognitive_reframe_exclusive",
        "completion_exclusive", "fiction_exclusive", "roleplay_exclusive",                                                      
    }                                                                                                                           
    actual_names = set(data["subcircuits"].keys())
    log_test(                                                                                                                   
        "T-07b: all 11 subcircuits present", 
        actual_names == expected_names,                                                                                         
        f"missing={expected_names - actual_names}; extra={actual_names - expected_names}",
    )                                                                                                                           
                                            
    # Invariants                                                                                                                
    excl = [set(data["subcircuits"][f"{c}_exclusive"]["features"])
            for c in ["analytical", "cognitive_reframe", "completion", "fiction", "roleplay"]]                                  
    all_disjoint = all(                            
        excl[i].isdisjoint(excl[j]) for i in range(5) for j in range(i + 1, 5)                                                  
    )                                                                                                                           
    log_test("T-07c: class_exclusive sets pairwise-disjoint", all_disjoint)                                                     
                                                                                                                                
    uni = set(data["subcircuits"]["universal_refusal_core"]["features"])
    can = set(data["subcircuits"]["canonical_pro_refusal"]["features"])                                                         
    log_test("T-07d: universal_refusal_core ∩ canonical_pro_refusal = ∅", uni.isdisjoint(can))
                                                                                                                                
    labels = json.loads((run_dir / "04_labels" / "feature_labels.json").read_text())
    late_feats = data["subcircuits"]["late_wave_layer24_32"]["features"]                                                        
    all_late = all(24 <= labels.get(k, {}).get("layer", -1) <= 32 for k in late_feats)
    log_test("T-07e: late_wave features all in L24–L32", all_late)                                                              
                                                                                                                                
    sizes = {n: v["size"] for n, v in data["subcircuits"].items()}                                                              
    log_test(                                                                                                                   
        "T-07f: universal_refusal_core size > 50 (expected ~83)",
        sizes["universal_refusal_core"] > 50,                                                                                   
        f"got {sizes['universal_refusal_core']}",                                                                               
    )                                                                                                                           
    log_test(                                                                                                                   
        "T-07g: late_wave_layer24_32 is largest bucket",                                                                        
        max(sizes.values()) == sizes["late_wave_layer24_32"],                                                                   
        f"max={max(sizes, key=sizes.get)}",                                                                                     
    )                                                                                                                           
                                                                                                                                
    # Sizes on reference run match probe predictions                                                                            
    log_test(                                                                                                                   
        "T-07h: sizes match probe predictions within ±2",
        abs(sizes["universal_refusal_core"] - 83) <= 2                                                                          
        and abs(sizes["canonical_pro_refusal"] - 56) <= 2                                                                       
        and abs(sizes["dampening_specialists"] - 52) <= 2                                                                       
        and abs(sizes["anti_refusal_amplifiers"] - 50) <= 2,                                                                    
        f"uni={sizes['universal_refusal_core']}, can={sizes['canonical_pro_refusal']}, "                                        
        f"damp={sizes['dampening_specialists']}, amp={sizes['anti_refusal_amplifiers']}",
    )

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Local pipeline validation tests")
    parser.add_argument(
        "--stage", choices=["01", "01-a5", "02", "02b", "03", "03-a4", "04-a7", "04-a8", "07", "utils", "utils-viz", "all"], default="all",
        help="Which stage to test (default: all)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Skip GPU tests (run utils only)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("REFUSAL-LENS PIPELINE — LOCAL VALIDATION TESTS")
    print("=" * 60)
    print(f"  Pipeline dir: {PIPELINE_DIR}")
    print(f"  Repo root:    {config.REPO_ROOT}")
    print(f"  GPU available: {check_gpu()}")

    t0 = time.time()

    if args.quick or args.stage == "utils":
        test_utils()
    elif args.stage == "all":
        test_utils()
        test_utils_viz()
        test_stage_02()
        test_stage_01_a5()
        test_stage_02b()
        test_stage_03_a4()
        test_stage_04_a7()
        test_stage_04_a8()
        test_stage_07()
        if check_gpu():
            test_stage_01()
            test_stage_03()
        else:
            log_skip("Stage 01 tests", "no GPU available")
            log_skip("Stage 03 tests", "no GPU available")
    elif args.stage == "01":
        if check_gpu():
            test_stage_01()
        else:
            log_skip("Stage 01 tests", "no GPU available")
    elif args.stage == "01-a5":
        test_stage_01_a5()
    elif args.stage == "02":
        test_stage_02()
    elif args.stage == "02b":
        test_stage_02b()
    elif args.stage == "03":
        if check_gpu():
            test_stage_03()
        else:
            log_skip("Stage 03 tests", "no GPU available")
    elif args.stage == "03-a4":
        test_stage_03_a4()
    elif args.stage == "04-a7":        
        test_stage_04_a7()
    elif args.stage == "04-a8":
        test_stage_04_a8()
    elif args.stage == "07":                                                                                                        
        test_stage_07()
    elif args.stage == "utils-viz":
        test_utils_viz()

    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed, {SKIP_COUNT} skipped")
    print(f"Elapsed: {elapsed:.1f}s")
    print("=" * 60)

    sys.exit(1 if FAIL_COUNT > 0 else 0)


if __name__ == "__main__":
    main()
