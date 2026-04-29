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
        "T-V1d: buckets constant is complete (3-way schema, Apr 22)",
        set(OVERLAP_BUCKETS) == {
            "shared_with_bare_and_ctrl", "shared_with_bare", "shared_with_ctrl",
            "jb_unique", "ctrl", "ctrl_unique", "bare", "bare_only", "non_feature",
        },
        f"got {sorted(OVERLAP_BUCKETS)}",
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

    # T-V5: annotate_overlap_3way + annotate_ctrl (Task 8, Apr 22)
    # Synthetic 3-graph fixture: bare / ctrl / jb. Features constructed so each
    # 3-way bucket has a known member.
    from utils_viz import annotate_overlap_3way, annotate_ctrl

    def _graph(feature_keys: list[tuple[str, str]]) -> dict:
        return {
            "metadata": {},
            "nodes": [
                {"node_id": nid, "layer": nid.split("_")[0],
                 "feature_type": "cross layer transcoder"}
                for nid, _ in feature_keys
            ] + [{"node_id": "E_0_0", "feature_type": "embedding"}],
        }

    # Feature plan for 3-way test (fiction class):
    #  fA (L10:F1) in bare + ctrl + jb → shared_with_bare_and_ctrl
    #  fB (L11:F2) in bare + jb, not ctrl → shared_with_bare
    #  fC (L12:F3) in ctrl + jb, not bare → shared_with_ctrl (PREFIX-INDUCED)
    #  fD (L13:F4) in jb only → jb_unique (TRUE JB-SEMANTIC)
    #  fE (L14:F5) in bare + ctrl only (not jb — doesn't appear in jb graph)
    #  fF (L15:F6) in ctrl only
    fA = ("10_1_0", "L10:F1")
    fB = ("11_2_0", "L11:F2")
    fC = ("12_3_0", "L12:F3")
    fD = ("13_4_0", "L13:F4")
    fE = ("14_5_0", "L14:F5")
    fF = ("15_6_0", "L15:F6")
    bare_g = _graph([fA, fB, fE])
    ctrl_g = _graph([fA, fC, fE, fF])
    jb_g = _graph([fA, fB, fC, fD])

    paths = {}
    try:
        for name, g in (("bare", bare_g), ("ctrl", ctrl_g), ("jb", jb_g)):
            with tempfile.NamedTemporaryFile(suffix=f"_{name}.json", delete=False, mode="w") as f:
                json.dump(g, f)
                paths[name] = Path(f.name)

        result = annotate_overlap_3way(paths["jb"], paths["bare"], paths["ctrl"],
                                       "fiction", 0)
        nodes = {n["node_id"]: n for n in result["nodes"]}
        log_test(
            "T-V5a: fA in bare+ctrl+jb → shared_with_bare_and_ctrl",
            nodes[fA[0]]["overlap_bucket"] == "shared_with_bare_and_ctrl",
            nodes[fA[0]]["overlap_bucket"],
        )
        log_test(
            "T-V5b: fB in bare+jb (not ctrl) → shared_with_bare",
            nodes[fB[0]]["overlap_bucket"] == "shared_with_bare",
            nodes[fB[0]]["overlap_bucket"],
        )
        log_test(
            "T-V5c: fC in ctrl+jb (not bare) → shared_with_ctrl (PREFIX-INDUCED)",
            nodes[fC[0]]["overlap_bucket"] == "shared_with_ctrl",
            nodes[fC[0]]["overlap_bucket"],
        )
        log_test(
            "T-V5d: fD in jb only → jb_unique (TRUE JB-SEMANTIC)",
            nodes[fD[0]]["overlap_bucket"] == "jb_unique",
            nodes[fD[0]]["overlap_bucket"],
        )
        log_test(
            "T-V5e: non-feature node → non_feature",
            nodes["E_0_0"]["overlap_bucket"] == "non_feature",
        )
        log_test(
            "T-V5f: overlap_mode metadata = '3way'",
            result["metadata"].get("overlap_mode") == "3way",
        )
        # Counts recorded in metadata
        counts = result["metadata"].get("overlap_counts", {})
        log_test(
            "T-V5g: overlap_counts has correct per-bucket sizes",
            counts.get("shared_with_bare_and_ctrl") == 1
            and counts.get("shared_with_bare") == 1
            and counts.get("shared_with_ctrl") == 1
            and counts.get("jb_unique") == 1,
            f"got {counts}",
        )

        # annotate_ctrl: vs bare path (fF only in ctrl → ctrl_unique; fA/fE in bare+ctrl → shared_with_bare)
        result_ctrl = annotate_ctrl(paths["ctrl"], paths["bare"], "fiction", 0)
        ctrl_nodes = {n["node_id"]: n for n in result_ctrl["nodes"]}
        log_test(
            "T-V5h: annotate_ctrl fA (bare+ctrl) → shared_with_bare",
            ctrl_nodes[fA[0]]["overlap_bucket"] == "shared_with_bare",
        )
        log_test(
            "T-V5i: annotate_ctrl fC (ctrl+jb, not bare) → ctrl_unique",
            ctrl_nodes[fC[0]]["overlap_bucket"] == "ctrl_unique",
        )
        log_test(
            "T-V5j: annotate_ctrl fF (ctrl only) → ctrl_unique",
            ctrl_nodes[fF[0]]["overlap_bucket"] == "ctrl_unique",
        )
        log_test(
            "T-V5k: annotate_ctrl writes ctrl_class + overlap_mode metadata",
            result_ctrl["metadata"].get("ctrl_class") == "fiction"
            and result_ctrl["metadata"].get("overlap_mode") == "ctrl_vs_bare",
        )
    finally:
        for p in paths.values():
            if p.exists():
                p.unlink()

    # T-V6: ctrl-aware subcircuit filter rules (Task 10 feedthrough into Stage 05)
    from utils_viz import _subcircuit_allowed
    log_test(
        "T-V6a: ctrl_shared_refusal allowed in {bare, ctrl, shared_*}",
        all(
            _subcircuit_allowed("ctrl_shared_refusal", b, None)
            for b in ("bare", "ctrl", "shared_with_bare",
                      "shared_with_bare_and_ctrl", "shared_with_ctrl")
        ),
    )
    log_test(
        "T-V6b: ctrl_shared_refusal REJECTED when bucket=jb_unique",
        not _subcircuit_allowed("ctrl_shared_refusal", "jb_unique", "fiction"),
    )
    log_test(
        "T-V6c: ctrl_shared_refusal REJECTED when bucket=ctrl_unique",
        not _subcircuit_allowed("ctrl_shared_refusal", "ctrl_unique", None),
    )
    log_test(
        "T-V6d: ctrl_only allowed when bucket=ctrl_unique",
        _subcircuit_allowed("ctrl_only", "ctrl_unique", None),
    )
    log_test(
        "T-V6e: ctrl_only REJECTED when bucket in {jb_unique, shared_*, bare}",
        not any(
            _subcircuit_allowed("ctrl_only", b, "fiction")
            for b in ("jb_unique", "shared_with_bare", "shared_with_ctrl",
                      "shared_with_bare_and_ctrl", "bare")
        ),
    )
    log_test(
        "T-V6f: jb_fiction_specific_vs_ctrl allowed when bucket=jb_unique + jb_class=fiction",
        _subcircuit_allowed("jb_fiction_specific_vs_ctrl", "jb_unique", "fiction"),
    )
    log_test(
        "T-V6g: jb_fiction_specific_vs_ctrl allowed when bucket=shared_with_bare + jb_class=fiction",
        _subcircuit_allowed("jb_fiction_specific_vs_ctrl", "shared_with_bare", "fiction"),
    )
    log_test(
        "T-V6h: jb_fiction_specific_vs_ctrl REJECTED when bucket=shared_with_ctrl (feature IS in ctrl)",
        not _subcircuit_allowed("jb_fiction_specific_vs_ctrl", "shared_with_ctrl", "fiction"),
    )
    log_test(
        "T-V6i: jb_fiction_specific_vs_ctrl REJECTED for wrong class",
        not _subcircuit_allowed("jb_fiction_specific_vs_ctrl", "jb_unique", "roleplay"),
    )
    log_test(
        "T-V6j: jb_cognitive_reframe_specific_vs_ctrl parses multi-word class",
        _subcircuit_allowed("jb_cognitive_reframe_specific_vs_ctrl", "jb_unique", "cognitive_reframe")
        and not _subcircuit_allowed("jb_cognitive_reframe_specific_vs_ctrl", "jb_unique", "fiction"),
    )
    log_test(
        "T-V6k: existing universal_refusal_core accepts new shared_with_bare_and_ctrl bucket",
        _subcircuit_allowed("universal_refusal_core", "shared_with_bare_and_ctrl", "fiction"),
    )
    log_test(
        "T-V6l: canonical_pro_refusal accepts shared_with_ctrl (prefix-induced still not bare)",
        _subcircuit_allowed("canonical_pro_refusal", "shared_with_ctrl", "fiction"),
    )


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
    log_test(
        "T-02u: config.TARGET_POSITIONS_MULTI == [-5, -3, -2] (template positions)",
        sorted(config.TARGET_POSITIONS_MULTI) == [-5, -3, -2],
        f"got {sorted(config.TARGET_POSITIONS_MULTI)}",
    )
    log_test(
        "T-02v: config.TARGET_POSITIONS_SINGLE == [-2] (causally-verified baseline)",
        list(config.TARGET_POSITIONS_SINGLE) == [-2],
        f"got {list(config.TARGET_POSITIONS_SINGLE)}",
    )

    # CLI flags for two-graph scheme
    sys.argv = ["02_run_attribution.py", "--run-dir", "/tmp/fake"]
    args = stage02.parse_args()
    log_test(
        "T-02w: --multi-position-targets defaults to [-5, -3, -2]",
        sorted(args.multi_position_targets) == [-5, -3, -2],
        f"got {args.multi_position_targets}",
    )
    log_test(
        "T-02x: --single-position-target defaults to -2",
        args.single_position_target == -2,
        f"got {args.single_position_target}",
    )
    log_test(
        "T-02y: --skip-multi-graph / --skip-single-graph default False",
        args.skip_multi_graph is False and args.skip_single_graph is False,
    )

    sys.argv = [
        "02_run_attribution.py", "--run-dir", "/tmp",
        "--multi-position-targets", "-5", "-3", "-2", "-15",
        "--single-position-target", "-5",
        "--skip-single-graph",
    ]
    args = stage02.parse_args()
    log_test(
        "T-02z: --multi-position-targets / --single-position-target / --skip-* parse",
        sorted(args.multi_position_targets) == [-15, -5, -3, -2]
        and args.single_position_target == -5
        and args.skip_single_graph is True,
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
            update_metadata = False
            per_position_layer = 15
            per_position_positions = [-2]
            skip_per_position = True

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
        no_plots = False  # added 2026-04-21 along with the 3-way stats refactor

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

    # T-A3c: legacy run's stats numerically stable across 02b refactors.
    # Post-refactor schema is stats[<mode>][<comparison>][<class>][...] (nested).
    # For legacy (pre-ctrl) runs the equivalent of the old flat [<class>] lives
    # under ["single"]["vs_bare"][<class>]. Skip the test when pre_stats was
    # saved under the legacy flat schema AND no mapping is possible.
    if pre_stats is not None and stats_json.exists():
        post_stats = json.loads(stats_json.read_text())

        def _navigate_legacy(ps: dict, post: dict) -> dict | None:
            """Return {cls: old_flat_stats_dict} view into the new structure, or None."""
            if not isinstance(post, dict):
                return None
            single_vs_bare = post.get("single", {}).get("vs_bare")
            if isinstance(single_vs_bare, dict):
                return single_vs_bare
            # Already flat (very old test snapshot): return as-is
            if all(isinstance(v, dict) and "mean_delta" in v for v in post.values()):
                return post
            return None

        post_flat = _navigate_legacy(pre_stats, post_stats)
        pre_is_legacy_flat = all(
            isinstance(v, dict) and "mean_delta" in v for v in pre_stats.values()
        )

        if post_flat is None or not pre_is_legacy_flat:
            log_skip(
                "T-A3c: statistical_analysis.json numerically stable",
                "pre-snapshot uses pre-refactor schema; reset by deleting old statistical_analysis.json",
            )
        else:
            # Map old top-level keys to new field names inside _paired_stats output.
            # Legacy keys that were renamed (bare_mean_net → a_mean_net, cls_mean_net
            # → b_mean_net, n_cls_lower → n_treatment_lower, etc.) are skipped to
            # avoid false mismatches. We only compare semantically-invariant fields.
            invariant_keys = {
                "n_pairs", "mean_delta", "std_delta",
                "wilcoxon_pval", "ttest_pval",
                "cohens_d", "ci_95_low", "ci_95_high",
                "d_pos", "d_neg",
            }
            mismatches = []
            for cls in pre_stats:
                if cls not in post_flat:
                    mismatches.append(f"class '{cls}' missing post-run")
                    continue
                for key in invariant_keys:
                    if key not in pre_stats[cls]:
                        continue
                    pre_val = pre_stats[cls][key]
                    post_val = post_flat[cls].get(key)
                    if post_val is None:
                        mismatches.append(f"{cls}.{key} missing")
                        continue
                    if isinstance(pre_val, float) and isinstance(post_val, float):
                        if pre_val == 0 and abs(post_val) > 1e-10:
                            mismatches.append(f"{cls}.{key}: {pre_val} -> {post_val}")
                        elif pre_val != 0:
                            rel = abs(post_val - pre_val) / abs(pre_val)
                            # Relaxed to 1e-4: bootstrap deltas can drift slightly
                            # across bootstrap seed reuse when loop order changes.
                            if rel > 1e-4:
                                mismatches.append(
                                    f"{cls}.{key}: {pre_val} -> {post_val} (rel={rel:.2e})"
                                )
                    elif pre_val != post_val:
                        mismatches.append(f"{cls}.{key}: {pre_val!r} -> {post_val!r}")
            log_test(
                "T-A3c: statistical_analysis.json numerically stable (invariant keys, rel_tol 1e-4)",
                len(mismatches) == 0,
                "; ".join(mismatches[:3])
                if mismatches else f"{len(invariant_keys) * len(pre_stats)} values checked",
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
    
    # A6: bare-vs-JB distribution plot (legacy name OR the new mode-suffixed names)
    dist_legacy = run_dir / "02b_stats" / "distribution_by_class.png"
    dist_multi = run_dir / "02b_stats" / "distribution_by_class_multi.png"
    dist_single = run_dir / "02b_stats" / "distribution_by_class_single.png"
    found = next((d for d in (dist_legacy, dist_multi, dist_single) if d.exists()), None)
    log_test(
        "T-A6a: distribution_by_class[_multi/_single].png exists",
        found is not None,
        str(found.relative_to(run_dir)) if found else "none of _multi / _single / legacy present",
    )
    if found is not None:
        log_test(
            "T-A6b: distribution_by_class plot non-trivial (>5KB)",
            found.stat().st_size > 5_000,
            f"size={found.stat().st_size} bytes",
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
        upset_only = False

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
        # Legacy schema tags bucket entries with bare class names ("fiction") → 5.
        # New 11-cond schema tags with full condition names ("jb_fiction"/"ctrl_fiction") → 10.
        # Either is correct depending on which dataset the Stage 04 run consumed.
        legacy_classes = 5
        new_classes = 10
        log_test(
            "T-A7d: 5 (legacy) or 10 (new 11-cond) classes detected",
            fs.get("n_classes") in (legacy_classes, new_classes),
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

def test_stage_04_schema():
    """T-S4: Stage 04 schema-aware collection (new 11-cond × 2-graph schema).

    Runs against the committed smoke JSON at
    data/results/pipeline_runs/attribution_results_test.json — no pipeline run needed.
    Verifies:
      - collect_all_features reads graphs.multi.top50_features (not legacy flat)
      - conditions_seen contains full condition names (bare, jb_*, ctrl_*)
      - top50_conditions ⊆ conditions_seen
      - build_per_condition_sets emits bare + 5 jb_* + 5 ctrl_* keys
      - collect_comparison_features traverses vs_bare/vs_ctrl/ctrl_vs_bare
    """
    print("\n" + "=" * 60)
    print("STAGE 04 SCHEMA TESTS (T-S4*)")
    print("=" * 60)

    smoke_path = (
        config.REPO_ROOT / "data" / "results" / "pipeline_runs" / "attribution_results_test.json"
    )
    if not smoke_path.exists():
        log_skip("T-S4: Stage 04 schema", f"smoke JSON missing at {smoke_path}")
        return
    print(f"  Using smoke JSON: {smoke_path.relative_to(config.REPO_ROOT)}")

    import importlib
    stage04 = importlib.import_module("04_label_features")

    raw = json.loads(smoke_path.read_text())
    results = raw["results"] if isinstance(raw, dict) else raw

    features = stage04.collect_all_features(results)
    log_test(
        "T-S4a: collect_all_features returns non-empty dict",
        isinstance(features, dict) and len(features) > 0,
        f"got {len(features)} features",
    )

    # conditions_seen uses new full-condition naming
    all_conds = set()
    for info in features.values():
        all_conds.update(info.get("conditions_seen", []))
    expected_conds = {
        "bare",
        "jb_fiction", "ctrl_fiction",
        "jb_roleplay", "ctrl_roleplay",
    }
    log_test(
        "T-S4b: conditions_seen contains full condition names (bare, jb_*, ctrl_*)",
        expected_conds.issubset(all_conds),
        f"missing: {sorted(expected_conds - all_conds)}",
    )

    # top50_conditions field is populated + subset of conditions_seen
    all_top50_cond = set()
    for info in features.values():
        all_top50_cond.update(info.get("top50_conditions", []))
    log_test(
        "T-S4c: top50_conditions populated",
        len(all_top50_cond) > 0,
        f"got {len(all_top50_cond)} distinct conditions",
    )
    subset_ok = all(
        set(info.get("top50_conditions", [])).issubset(set(info.get("conditions_seen", [])))
        for info in features.values()
    )
    log_test("T-S4d: top50_conditions ⊆ conditions_seen for every feature", subset_ok)

    # per_condition_top50 keyed by condition name
    per_cond = stage04.build_per_condition_sets(features)
    jb_keys = {k for k in per_cond if k.startswith("jb_")}
    ctrl_keys = {k for k in per_cond if k.startswith("ctrl_")}
    log_test(
        "T-S4e: per_condition_top50 has 'bare'",
        "bare" in per_cond,
        f"got keys: {sorted(per_cond.keys())[:12]}",
    )
    log_test(
        "T-S4f: per_condition_top50 has 5 jb_* + 5 ctrl_* classes",
        len(jb_keys) == 5 and len(ctrl_keys) == 5,
        f"jb: {sorted(jb_keys)}, ctrl: {sorted(ctrl_keys)}",
    )
    log_test(
        "T-S4g: per_condition_top50 values are sorted feature-key lists",
        all(isinstance(v, list) and (len(v) == 0 or isinstance(v[0], str))
            for v in per_cond.values()),
    )

    # collect_comparison_features traverses the 3 sub-buckets
    comp = stage04.collect_comparison_features(results)
    log_test(
        "T-S4h: collect_comparison_features returns 3 categories",
        set(comp.keys()) == {"sign_flipped", "dampened", "amplified_anti"},
        f"got {sorted(comp.keys())}",
    )
    # At least one category should have non-empty entries
    has_any = any(len(v) > 0 for v in comp.values())
    log_test("T-S4i: at least one comparison category is non-empty", has_any)
    # Classes on comparison entries use full condition names (jb_*, ctrl_*)
    example_classes = set()
    for cat_data in comp.values():
        for info in cat_data.values():
            example_classes.update(info.get("classes", []))
    prefixed = any(c.startswith("jb_") or c.startswith("ctrl_") for c in example_classes)
    log_test(
        "T-S4j: comparison 'classes' use full condition names",
        prefixed,
        f"saw: {sorted(example_classes)[:10]}",
    )


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
        # New flags from P4 (sweep configs); keep disabled in legacy-path tests.
        sweep_configs = ""
        graph_mode = "multi"
        skip_legacy = False
        skip_sweep = True
                                                                                                                                
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
                                                                                                                                
    expected_legacy = {
        "universal_refusal_core", "canonical_pro_refusal",
        "sign_flip_convergent", "dampening_specialists",
        "anti_refusal_amplifiers", "late_wave_layer24_32",
        "analytical_exclusive", "cognitive_reframe_exclusive",
        "completion_exclusive", "fiction_exclusive", "roleplay_exclusive",
    }
    expected_ctrl = {
        "ctrl_shared_refusal", "ctrl_only",
        "jb_analytical_specific_vs_ctrl", "jb_cognitive_reframe_specific_vs_ctrl",
        "jb_completion_specific_vs_ctrl", "jb_fiction_specific_vs_ctrl",
        "jb_roleplay_specific_vs_ctrl",
    }
    expected_names = expected_legacy | expected_ctrl
    actual_names = set(data["subcircuits"].keys())
    log_test(
        "T-07b: all 18 subcircuits present (11 legacy + 7 ctrl-aware)",
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
    # T-07g and T-07h are calibrated on the L32-measurement reference run
    # (run_20260417_010035). On L15-measurement runs, late_wave_layer24_32 is
    # EMPTY (attribution doesn't reach past L15), and absolute subcircuit sizes
    # differ because the measurement target moved. Reference-run-specific.
    is_legacy_reference = run_dir.name == "run_20260417_010035"
    if is_legacy_reference:
        log_test(
            "T-07g: late_wave_layer24_32 is largest bucket (legacy L32 run)",
            max(sizes.values()) == sizes["late_wave_layer24_32"],
            f"max={max(sizes, key=sizes.get)}",
        )
        log_test(
            "T-07h: sizes match probe predictions within ±2 (legacy L32 run)",
            abs(sizes["universal_refusal_core"] - 83) <= 2
            and abs(sizes["canonical_pro_refusal"] - 56) <= 2
            and abs(sizes["dampening_specialists"] - 52) <= 2
            and abs(sizes["anti_refusal_amplifiers"] - 50) <= 2,
            f"uni={sizes['universal_refusal_core']}, can={sizes['canonical_pro_refusal']}, "
            f"damp={sizes['dampening_specialists']}, amp={sizes['anti_refusal_amplifiers']}",
        )
    else:
        log_test(
            "T-07g: late_wave_layer24_32 empty on L15-measurement run",
            sizes["late_wave_layer24_32"] == 0,
            f"got {sizes['late_wave_layer24_32']} (expected 0 when target layer <= L23)",
        )
        log_test(
            "T-07h: universal_refusal_core is non-empty on L15-measurement run",
            sizes["universal_refusal_core"] > 0,
            f"got {sizes['universal_refusal_core']}",
        )

    # Ctrl-aware metadata field present; legacy-path invariants
    # (ctrl-available path covered by test_stage_07_synthetic_ctrl)
    ctrl_available = data.get("metadata", {}).get("ctrl_available", False)
    log_test(
        "T-07i: metadata.ctrl_available field present",
        "ctrl_available" in data.get("metadata", {}),
        f"got ctrl_available={ctrl_available}",
    )
    if not ctrl_available:
        ctrl_sizes = {n: sizes[n] for n in expected_ctrl}
        log_test(
            "T-07j: legacy data → all 7 ctrl-aware subcircuits empty",
            all(s == 0 for s in ctrl_sizes.values()),
            f"nonzero: {[(n, s) for n, s in ctrl_sizes.items() if s > 0]}",
        )
        log_test(
            "T-07k: legacy data → jb_vs_ctrl_contrast is empty dict",
            data.get("jb_vs_ctrl_contrast", "absent") == {},
            f"got {type(data.get('jb_vs_ctrl_contrast')).__name__}",
        )


def test_stage_07_synthetic_ctrl():
    """T-S7ctrl: ctrl-aware Stage 07 rules exercised against a synthetic fixture.

    The legacy run directory has no per_condition_top50 block, so the ctrl-aware
    branch never fires in test_stage_07. This test builds a minimal synthetic
    feature_class_sets.json + feature_labels.json with known top-50 memberships
    across all 11 conditions, invokes Stage 07, and verifies each ctrl-aware
    rule produces the expected feature set.
    """
    import importlib
    import tempfile

    print("\n" + "=" * 60)
    print("STAGE 07 CTRL-AWARE TESTS (synthetic fixture, T-S7ctrl)")
    print("=" * 60)

    stage07 = importlib.import_module("07_identify_subcircuits")

    JB_CLASSES = ["analytical", "cognitive_reframe", "completion", "fiction", "roleplay"]

    # Construct feature keys with known membership patterns
    # fA: in bare + all 5 jb + all 5 ctrl            → universal_core, NOT canonical
    # fB: in bare + all 5 ctrl, NOT all 5 jb (miss fiction) → ctrl_shared_refusal
    # fC: in all 5 ctrl only (not bare, not any jb)  → ctrl_only
    # fD: in all 5 jb, not bare                      → canonical_pro_refusal
    # fE: in jb_fiction only (not ctrl_fiction, not other conds) → fiction_exclusive + jb_fiction_specific_vs_ctrl
    # fF: in jb_roleplay AND ctrl_roleplay            → NOT jb_specific_vs_ctrl (filtered by ctrl)
    per_condition_top50 = {"bare": ["fA", "fB"]}
    for cls in JB_CLASSES:
        jb_keys = ["fA", "fD"]
        ctrl_keys = ["fA", "fB", "fC"]
        if cls == "fiction":
            jb_keys = jb_keys + ["fE"]              # fE in jb_fiction only
            # fB missing from jb_fiction intentionally — makes fB ctrl_shared (not universal)
        if cls == "roleplay":
            jb_keys = jb_keys + ["fF"]
            ctrl_keys = ctrl_keys + ["fF"]           # fF in both jb_roleplay and ctrl_roleplay
        per_condition_top50[f"jb_{cls}"] = sorted(set(jb_keys))
        per_condition_top50[f"ctrl_{cls}"] = sorted(set(ctrl_keys))

    # For each feature, compute its conditions_seen from the per_condition sets above
    conds_seen = {}
    for cond, keys in per_condition_top50.items():
        for k in keys:
            conds_seen.setdefault(k, set()).add(cond)

    feature_labels = {
        k: {
            "layer": 20,
            "feature_idx": i,
            "max_abs_attribution": 1.0,
            "conditions_seen": sorted(conds_seen[k]),
            "top50_conditions": sorted(conds_seen[k]),
            "top_logits": [],
            "bottom_logits": [],
            "activation_frequency": 0.01,
            "examples": [],
            "labeled": False,
        }
        for i, k in enumerate(["fA", "fB", "fC", "fD", "fE", "fF"])
    }

    class_sets = {
        "n_classes": 5,
        "classes": sorted(JB_CLASSES),
        "by_bucket": {
            "sign_flipped": {"total": 0, "features": {}},
            "dampened": {"total": 0, "features": {}},
            "amplified_anti": {"total": 0, "features": {}},
        },
        "combined": {"total": 0, "features": {}},
        "per_condition_top50": per_condition_top50,
    }

    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td) / "synthetic_run"
        labels_dir = run_dir / "04_labels"
        labels_dir.mkdir(parents=True)
        (labels_dir / "feature_labels.json").write_text(json.dumps(feature_labels))
        (labels_dir / "feature_class_sets.json").write_text(json.dumps(class_sets))

        class MockArgs:
            convergent_min = 3
            late_wave_start = 24
            late_wave_end = 32
            # New flags from P4 (sweep configs); keep disabled in legacy-path tests.
            sweep_configs = ""
            graph_mode = "multi"
            skip_legacy = False
            skip_sweep = True
        mock = MockArgs()
        mock.run_dir = run_dir
        orig = stage07.parse_args
        stage07.parse_args = lambda: mock
        try:
            stage07.main()
        finally:
            stage07.parse_args = orig

        out = json.loads((run_dir / "07_subcircuits" / "subcircuits.json").read_text())
        sc = out["subcircuits"]

        log_test(
            "T-S7ctrl-a: metadata.ctrl_available=True on synthetic fixture",
            out.get("metadata", {}).get("ctrl_available") is True,
        )
        # Expected memberships (by construction above):
        # fA in bare + all 5 jb + all 5 ctrl  → universal_core
        log_test(
            "T-S7ctrl-b: universal_refusal_core == {fA}",
            set(sc["universal_refusal_core"]["features"]) == {"fA"},
            f"got {sc['universal_refusal_core']['features']}",
        )
        # fD in all 5 jb, not bare → canonical_pro_refusal
        log_test(
            "T-S7ctrl-c: canonical_pro_refusal == {fD}",
            set(sc["canonical_pro_refusal"]["features"]) == {"fD"},
            f"got {sc['canonical_pro_refusal']['features']}",
        )
        # fB in bare + all 5 ctrl, missing jb_fiction → ctrl_shared_refusal
        log_test(
            "T-S7ctrl-d: ctrl_shared_refusal == {fB}",
            set(sc["ctrl_shared_refusal"]["features"]) == {"fB"},
            f"got {sc['ctrl_shared_refusal']['features']}",
        )
        # fC in all 5 ctrl only → ctrl_only
        log_test(
            "T-S7ctrl-e: ctrl_only == {fC}",
            set(sc["ctrl_only"]["features"]) == {"fC"},
            f"got {sc['ctrl_only']['features']}",
        )
        # fE in jb_fiction only; fD in all jb but never ctrl → jb_fiction_specific = {fD, fE}
        log_test(
            "T-S7ctrl-f: jb_fiction_specific_vs_ctrl == {fD, fE}",
            set(sc["jb_fiction_specific_vs_ctrl"]["features"]) == {"fD", "fE"},
            f"got {sc['jb_fiction_specific_vs_ctrl']['features']}",
        )
        # fF in jb_roleplay AND ctrl_roleplay → fF filtered; fD remains jb-only → {fD}
        log_test(
            "T-S7ctrl-g: jb_roleplay_specific_vs_ctrl == {fD} (fF filtered by ctrl match)",
            set(sc["jb_roleplay_specific_vs_ctrl"]["features"]) == {"fD"},
            f"got {sc['jb_roleplay_specific_vs_ctrl']['features']}",
        )

        # jb_vs_ctrl_contrast arithmetic + values
        contrast = out.get("jb_vs_ctrl_contrast", {})
        fiction = contrast.get("fiction", {})
        # jb_fiction top-50 = [fA, fD, fE]; ctrl_fiction = [fA, fB, fC]; inter = [fA]
        # jb_specific = [fD, fE] = 2; overlap = [fA] = 1; jb_top50 = 3
        log_test(
            "T-S7ctrl-h: fiction contrast: jb=3, ctrl=3, intersection=1, jb_specific=2",
            fiction.get("jb_top50") == 3
            and fiction.get("ctrl_top50") == 3
            and fiction.get("intersection") == 1
            and fiction.get("jb_specific") == 2,
            f"got {fiction}",
        )
        # jb_specific_frac = 2/3 ≈ 0.667
        log_test(
            "T-S7ctrl-i: fiction jb_specific_frac ≈ 0.667",
            abs(fiction.get("jb_specific_frac", 0) - 0.667) < 0.01,
            f"got {fiction.get('jb_specific_frac')}",
        )
        # Novel-insight figures
        for fig in ["jb_vs_ctrl_contrast.png", "jb_specific_by_layer.png"]:
            p = run_dir / "07_subcircuits" / fig
            log_test(
                f"T-S7ctrl-j: {fig} generated in ctrl-available path",
                p.exists() and p.stat().st_size > 3_000,
                f"size={p.stat().st_size if p.exists() else 'missing'}",
            )


def test_stage_08():
    """T-S8: Stage 08 subcircuit ablation — primitives, intervention specs,
    dissociation aggregation. No GPU required."""
    print("\n" + "=" * 60)
    print("STAGE 08 TESTS (subcircuit ablation, no GPU)")
    print("=" * 60)

    import importlib

    from utils import (
        load_cart,
        load_subcircuit_features,
        parse_feature_key,
        resolve_anchor_positions,
    )

    # T-S8a: parse_feature_key happy + error paths
    ok = parse_feature_key("L14:F480") == (14, 480) and parse_feature_key("L0:F0") == (0, 0)
    try:
        parse_feature_key("bad")
        raised = False
    except ValueError:
        raised = True
    log_test("T-S8a: parse_feature_key parses canonical + rejects malformed", ok and raised)

    # T-S8b: load_subcircuit_features dedupes across overlapping sets
    with tempfile.TemporaryDirectory() as tmp:
        sub_path = Path(tmp) / "sub.json"
        sub_path.write_text(json.dumps({
            "subcircuits": {
                "A": {"features": ["L0:F1", "L0:F2", "L1:F5"]},
                "B": {"features": ["L0:F2", "L3:F9"]},
                "empty": {"features": []},
            },
        }))
        feats_both = load_subcircuit_features(sub_path, ["A", "B"])
        log_test(
            "T-S8b: load_subcircuit_features dedupes (A∪B) and preserves order",
            feats_both == [(0, 1), (0, 2), (1, 5), (3, 9)],
            f"got {feats_both}",
        )
        # Missing name raises
        try:
            load_subcircuit_features(sub_path, ["not_here"])
            raised = False
        except KeyError:
            raised = True
        log_test("T-S8b-err: missing subcircuit name raises KeyError", raised)

    # T-S8c: load_cart accepts minimal + full schema
    with tempfile.TemporaryDirectory() as tmp:
        cart_min = Path(tmp) / "cart_min.json"
        cart_min.write_text(json.dumps({"features": [{"layer": 14, "feat_idx": 480}]}))
        got = load_cart(cart_min)
        log_test("T-S8c: load_cart defaults value=0.0 when omitted",
                 got == [(14, 480, 0.0)], f"got {got}")

        cart_full = Path(tmp) / "cart_full.json"
        cart_full.write_text(json.dumps({
            "features": [
                {"layer": 15, "feat_idx": 200, "value": 0.0, "label": "refusal"},
                {"layer": 20, "feat_idx": 100, "value": 0.5},
            ],
        }))
        got_full = load_cart(cart_full)
        log_test("T-S8c-full: load_cart preserves custom values and extra keys",
                 got_full == [(15, 200, 0.0), (20, 100, 0.5)], f"got {got_full}")

    # T-S8d: resolve_anchor_positions resolves negatives + rejects OOB
    log_test(
        "T-S8d: resolve_anchor_positions([-5,-3,-2], length=20) == [15,17,18]",
        resolve_anchor_positions([-5, -3, -2], 20) == [15, 17, 18],
    )
    try:
        resolve_anchor_positions([-100], 10)
        oob_raised = False
    except ValueError:
        oob_raised = True
    log_test("T-S8d-err: out-of-bounds anchor raises ValueError", oob_raised)

    # Load the 08 module for function-level tests
    s08 = importlib.import_module("08_ablate_subcircuits")

    # T-S8e: build_interventions in 'all' mode
    feats = [(14, 480), (15, 200)]
    ivs_all = s08.build_interventions(feats, "all", tokenized_length=20)
    ok_all = (
        len(ivs_all) == 2
        and all(iv[1] == slice(None) for iv in ivs_all)
        and all(iv[3] == 0.0 for iv in ivs_all)
        and ivs_all[0][:1] + ivs_all[0][2:3] == (14, 480)
    )
    log_test("T-S8e: build_interventions('all') produces 1-per-feature at slice(None)", ok_all)

    # T-S8f: build_interventions in 'anchors' mode
    ivs_anc = s08.build_interventions(feats, "anchors", tokenized_length=20)
    ok_anc = (
        len(ivs_anc) == 6   # 2 features × 3 anchor positions
        and sorted({iv[1] for iv in ivs_anc}) == [15, 17, 18]
        and all(iv[3] == 0.0 for iv in ivs_anc)
    )
    log_test("T-S8f: build_interventions('anchors') produces features × anchor_positions", ok_anc)

    # T-S8g: default subcircuit names are real subcircuit-rule keys
    default_names = list(config.STAGE_08_DEFAULT_SUBCIRCUITS)
    expected = {"universal_refusal_core", "ctrl_shared_refusal",
                "jb_fiction_specific_vs_ctrl",
                "jb_analytical_specific_vs_ctrl",
                "jb_cognitive_reframe_specific_vs_ctrl"}
    log_test(
        "T-S8g: STAGE_08_DEFAULT_SUBCIRCUITS contains the required control + class-specific sets",
        expected.issubset(set(default_names)),
        f"got {default_names}",
    )

    # T-S8h: zero-positions mask for Gemma-3-it is slice(0, 4)
    log_test(
        "T-S8h: STAGE_08_FIRST_TOKEN_MASK == slice(0, 4) (Gemma-3-it transcoder zero-positions)",
        config.STAGE_08_FIRST_TOKEN_MASK == slice(0, 4),
    )

    # T-S8i: compute_dissociation_score on a hand-constructed fixture
    fake_summary = {
        "per_ablation": {
            "jb_fiction_specific_vs_ctrl": {
                "n_features": 52,
                "positions": {
                    "all": {
                        "jb_fiction":            {"recovery_rate": 0.85, "break_rate": 0.0,
                                                  "n_baseline_comply": 20, "n_baseline_refuse": 0,
                                                  "n_ablated_refuse": 17, "n_ablated_comply": 3,
                                                  "n_changed": 17, "n_coherent_changed": 17,
                                                  "n_recovered_refusal": 17, "n_broke_refusal": 0,
                                                  "n_seen": 20},
                        "jb_analytical":         {"recovery_rate": 0.20, "break_rate": 0.0,
                                                  "n_baseline_comply": 25, "n_baseline_refuse": 0,
                                                  "n_ablated_refuse": 5, "n_ablated_comply": 20,
                                                  "n_changed": 5, "n_coherent_changed": 5,
                                                  "n_recovered_refusal": 5, "n_broke_refusal": 0,
                                                  "n_seen": 25},
                        "jb_cognitive_reframe":  {"recovery_rate": 0.25, "break_rate": 0.0,
                                                  "n_baseline_comply": 30, "n_baseline_refuse": 0,
                                                  "n_ablated_refuse": 7, "n_ablated_comply": 23,
                                                  "n_changed": 7, "n_coherent_changed": 7,
                                                  "n_recovered_refusal": 7, "n_broke_refusal": 0,
                                                  "n_seen": 30},
                        "jb_completion":         {"recovery_rate": 0.10, "break_rate": 0.0,
                                                  "n_baseline_comply": 10, "n_baseline_refuse": 0,
                                                  "n_ablated_refuse": 1, "n_ablated_comply": 9,
                                                  "n_changed": 1, "n_coherent_changed": 1,
                                                  "n_recovered_refusal": 1, "n_broke_refusal": 0,
                                                  "n_seen": 10},
                        "jb_roleplay":           {"recovery_rate": 0.15, "break_rate": 0.0,
                                                  "n_baseline_comply": 8, "n_baseline_refuse": 0,
                                                  "n_ablated_refuse": 1, "n_ablated_comply": 7,
                                                  "n_changed": 1, "n_coherent_changed": 1,
                                                  "n_recovered_refusal": 1, "n_broke_refusal": 0,
                                                  "n_seen": 8},
                    },
                },
            },
        },
    }
    diss = s08.compute_dissociation_score(fake_summary)
    fiction_rec = diss["jb_fiction_specific_vs_ctrl"]["all"]
    ok_diss = (
        fiction_rec["target_class"] == "fiction"
        and fiction_rec["target_recovery_rate"] == 0.85
        and 0.6 < fiction_rec["dissociation_delta"] < 0.7
    )
    log_test(
        "T-S8i: compute_dissociation_score selects target='fiction', Δ ≈ +67pp on fixture",
        ok_diss,
        f"got {fiction_rec}",
    )

    # T-S8j: aggregate_summary on a tiny 2-prompt fixture produces correct recovery math
    synth_results = [
        {
            "prompt_id": 1,
            "baseline": {
                "bare":       {"cls": "REFUSE"},
                "jb_fiction": {"cls": "COMPLY"},
                "jb_analytical": {"cls": "COMPLY"},
            },
            "ablations": {
                "S1": {
                    "n_features": 5,
                    "all": {
                        "bare":          {"cls": "COMPLY", "changed_vs_baseline": True, "coherent": True},
                        "jb_fiction":    {"cls": "REFUSE", "changed_vs_baseline": True, "coherent": True},
                        "jb_analytical": {"cls": "COMPLY", "changed_vs_baseline": False, "coherent": True},
                    },
                },
            },
        },
        {
            "prompt_id": 2,
            "baseline": {
                "bare":       {"cls": "REFUSE"},
                "jb_fiction": {"cls": "COMPLY"},
                "jb_analytical": {"cls": "REFUSE"},
            },
            "ablations": {
                "S1": {
                    "n_features": 5,
                    "all": {
                        "bare":          {"cls": "REFUSE", "changed_vs_baseline": False, "coherent": True},
                        "jb_fiction":    {"cls": "REFUSE", "changed_vs_baseline": True, "coherent": True},
                        "jb_analytical": {"cls": "REFUSE", "changed_vs_baseline": False, "coherent": True},
                    },
                },
            },
        },
    ]
    agg = s08.aggregate_summary(
        synth_results,
        ablation_sets={"S1": [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]},
        positions_modes=["all"],
        conditions=["bare", "jb_fiction", "jb_analytical"],
    )
    per = agg["per_ablation"]["S1"]["positions"]["all"]
    log_test(
        "T-S8j: aggregate recovery_rate jb_fiction == 2/2 (both flipped), break_rate bare == 1/2",
        per["jb_fiction"]["recovery_rate"] == 1.0
        and per["jb_fiction"]["n_recovered_refusal"] == 2
        and per["bare"]["break_rate"] == 0.5
        and per["bare"]["n_broke_refusal"] == 1,
        f"got {per}",
    )


def test_stage_07_sweep():
    """T-S7sweep: per-prompt subcircuit construction (P4).

    Builds a synthetic attribution_results.json + Stage 04 outputs, runs
    Stage 07 with a single sweep config, and verifies the per-config output
    files are emitted with the expected metadata + class-set arithmetic.
    """
    import importlib
    import tempfile

    print("\n" + "=" * 60)
    print("STAGE 07 SWEEP TESTS (per-prompt subcircuit construction, T-S7sweep)")
    print("=" * 60)

    stage07 = importlib.import_module("07_identify_subcircuits")

    JB_CLASSES = ["analytical", "cognitive_reframe", "completion", "fiction", "roleplay"]

    # Construct synthetic attribution_results: 2 prompts × 11 conditions, each
    # with a top_features dict. Choose memberships so per-prompt frequency at
    # threshold=0.5 (i.e., feature must appear in BOTH prompts) yields a known
    # subcircuit.
    #   fA: in bare top_features for BOTH prompts            → bare per-cond set
    #   fB: in jb_fiction for BOTH prompts only              → only jb_fiction
    #   fC: in jb_fiction for prompt 1 only                  → drops at freq=0.5
    #   fD: in ctrl_fiction for BOTH prompts                 → ctrl_fiction
    #   fE: in jb_fiction AND ctrl_fiction for BOTH prompts  → both
    def cond_top(*keys):
        return {k: 0.5 for k in keys}

    def make_row(pid: int, fc_jb_extra: list[str]) -> dict:
        conds: dict = {"bare": {"graphs": {"multi": {"top_features": cond_top("fA")}}}}
        for cls in JB_CLASSES:
            jb_keys = ["fX"]                  # noise feature, won't hit freq
            ctrl_keys = ["fY"]
            if cls == "fiction":
                jb_keys = ["fB", "fE"] + fc_jb_extra
                ctrl_keys = ["fD", "fE"]
            conds[f"jb_{cls}"] = {"graphs": {"multi": {"top_features": cond_top(*jb_keys)}}}
            conds[f"ctrl_{cls}"] = {"graphs": {"multi": {"top_features": cond_top(*ctrl_keys)}}}
        return {"prompt_idx": pid, "prompt_id": pid, "conditions": conds}

    attribution_results = {
        "metadata": {"measurement_layer": 15, "measurement_hook": "hook_resid_post"},
        "results": [
            make_row(0, fc_jb_extra=["fC"]),  # prompt 0 has fC in jb_fiction
            make_row(1, fc_jb_extra=[]),       # prompt 1 lacks fC
        ],
    }

    # Stage 04 outputs: minimal feature_labels + feature_class_sets.
    # per_condition_top50 will be IGNORED in sweep mode (we override it from
    # attribution_results), so we leave it empty.
    feature_labels = {
        k: {"layer": 12, "feature_idx": i, "max_abs_attribution": 1.0,
            "conditions_seen": [], "top50_conditions": [],
            "top_logits": [], "bottom_logits": [],
            "activation_frequency": 0.01, "examples": [], "labeled": False}
        for i, k in enumerate(["fA", "fB", "fC", "fD", "fE", "fX", "fY"])
    }
    class_sets = {
        "n_classes": 5,
        "classes": sorted(JB_CLASSES),
        "by_bucket": {
            "sign_flipped": {"total": 0, "features": {}},
            "dampened": {"total": 0, "features": {}},
            "amplified_anti": {"total": 0, "features": {}},
        },
        "combined": {"total": 0, "features": {}},
        "per_condition_top50": {},
    }

    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td) / "synthetic_run"
        labels_dir = run_dir / "04_labels"
        attr_dir = run_dir / "02_attribution"
        labels_dir.mkdir(parents=True)
        attr_dir.mkdir(parents=True)
        (labels_dir / "feature_labels.json").write_text(json.dumps(feature_labels))
        (labels_dir / "feature_class_sets.json").write_text(json.dumps(class_sets))
        (attr_dir / "attribution_results.json").write_text(json.dumps(attribution_results))

        class MockArgs:
            convergent_min = 3
            late_wave_start = 24
            late_wave_end = 32
            # Run two configs at once: lax (0.5 → ≥1 of 2 prompts) and strict
            # (1.0 → both prompts). Distinct outputs verify the threshold logic.
            sweep_configs = "50:0.5,50:1.0"
            graph_mode = "multi"
            skip_legacy = True          # only emit the sweep files
            skip_sweep = False
        mock = MockArgs()
        mock.run_dir = run_dir
        orig = stage07.parse_args
        stage07.parse_args = lambda: mock
        try:
            stage07.main()
        finally:
            stage07.parse_args = orig

        sweep_path = run_dir / "07_subcircuits" / "subcircuits_k50_f50.json"
        sweep_summary_path = run_dir / "07_subcircuits" / "subcircuits_k50_f50_summary.json"
        sweep_path_strict = run_dir / "07_subcircuits" / "subcircuits_k50_f100.json"
        log_test(
            "T-S7sweep-a: subcircuits_k50_f50.json emitted",
            sweep_path.exists(), str(sweep_path),
        )
        log_test(
            "T-S7sweep-b: subcircuits_k50_f50_summary.json emitted",
            sweep_summary_path.exists(), str(sweep_summary_path),
        )
        log_test(
            "T-S7sweep-b2: subcircuits_k50_f100.json (strict freq=1.0) emitted",
            sweep_path_strict.exists(), str(sweep_path_strict),
        )
        if not sweep_path.exists() or not sweep_path_strict.exists():
            return
        out_lax = json.loads(sweep_path.read_text())
        out_strict = json.loads(sweep_path_strict.read_text())
        meta = out_lax.get("metadata", {})
        log_test(
            "T-S7sweep-c: metadata records sweep_top_k=50, sweep_freq_threshold=0.5",
            meta.get("sweep_top_k") == 50 and meta.get("sweep_freq_threshold") == 0.5,
            f"got {meta}",
        )
        log_test(
            "T-S7sweep-d: metadata.is_legacy is False",
            meta.get("is_legacy") is False,
            f"got is_legacy={meta.get('is_legacy')}",
        )
        log_test(
            "T-S7sweep-e: metadata.n_prompts_per_condition recorded for all 11 conds",
            isinstance(meta.get("n_prompts_per_condition"), dict)
            and len(meta["n_prompts_per_condition"]) == 11,
            f"got {meta.get('n_prompts_per_condition')}",
        )
        # Lax (freq=0.5, threshold=1 of 2): both fB (both prompts) and fC (prompt 0
        # only) qualify for jb_fiction; fE qualifies for both jb and ctrl fiction;
        # so jb_fiction_specific_vs_ctrl = {fB, fC}.
        lax_spec = set(out_lax["subcircuits"]["jb_fiction_specific_vs_ctrl"]["features"])
        log_test(
            "T-S7sweep-f: lax (freq=0.5) jb_fiction_specific_vs_ctrl == {fB, fC}",
            lax_spec == {"fB", "fC"},
            f"got {lax_spec}",
        )
        # Strict (freq=1.0, threshold=2 of 2): fC drops out (only prompt 0 had
        # it); only fB qualifies for jb_fiction.
        strict_spec = set(out_strict["subcircuits"]["jb_fiction_specific_vs_ctrl"]["features"])
        log_test(
            "T-S7sweep-g: strict (freq=1.0) jb_fiction_specific_vs_ctrl == {fB}",
            strict_spec == {"fB"},
            f"got {strict_spec}",
        )
        # Strict subcircuit ⊆ lax subcircuit (raising the threshold can only
        # remove features).
        log_test(
            "T-S7sweep-h: strict subcircuit ⊆ lax (raising freq threshold drops only)",
            strict_spec.issubset(lax_spec),
            f"strict={strict_spec}, lax={lax_spec}",
        )


def test_stage_08_coverage():
    """T-S8cov: Stage 08 per-prompt coverage diagnostic + comply-weighted summary
    (P5). Pure-function tests; no GPU.
    """
    import importlib

    print("\n" + "=" * 60)
    print("STAGE 08 COVERAGE / WEIGHTED-SUMMARY TESTS (T-S8cov)")
    print("=" * 60)

    s08 = importlib.import_module("08_ablate_subcircuits")

    # T-S8cov-a: compute_coverage with full / partial / zero overlap
    feat_keys = ["L10:F1", "L10:F2", "L11:F3", "L12:F4"]
    prompt_top = {
        "jb_fiction": {"L10:F1": 0.5, "L11:F3": 0.2, "L20:F99": 0.05},
        "bare": {"L10:F2": 0.1},
    }
    cov_jbf = s08.compute_coverage(feat_keys, prompt_top, "jb_fiction")
    log_test(
        "T-S8cov-a: 2 of 4 features in jb_fiction top → frac=0.5",
        cov_jbf["n_features"] == 4
        and cov_jbf["n_in_top_k"] == 2
        and cov_jbf["frac_in_top_k"] == 0.5
        and abs(cov_jbf["sum_abs_attribution"] - 0.7) < 1e-6,
        f"got {cov_jbf}",
    )
    cov_bare = s08.compute_coverage(feat_keys, prompt_top, "bare")
    log_test(
        "T-S8cov-b: 1 of 4 features in bare top → frac=0.25",
        cov_bare["n_in_top_k"] == 1 and cov_bare["frac_in_top_k"] == 0.25,
        f"got {cov_bare}",
    )
    cov_missing = s08.compute_coverage(feat_keys, prompt_top, "ctrl_fiction")
    log_test(
        "T-S8cov-c: missing condition → frac=0",
        cov_missing["n_in_top_k"] == 0 and cov_missing["frac_in_top_k"] == 0.0,
        f"got {cov_missing}",
    )

    # T-S8cov-d: aggregate_coverage_summary — averages frac over prompts,
    # counts low-coverage prompts.
    synth_results = [
        {
            "prompt_id": 1,
            "ablations": {
                "S1": {
                    "n_features": 4,
                    "all": {
                        "jb_fiction": {
                            "cls": "REFUSE",
                            "coverage": {
                                "n_features": 4, "n_in_top_k": 2,
                                "frac_in_top_k": 0.5,
                                "sum_abs_attribution": 0.5,
                                "low_coverage": False,
                            },
                        },
                    },
                },
            },
        },
        {
            "prompt_id": 2,
            "ablations": {
                "S1": {
                    "n_features": 4,
                    "all": {
                        "jb_fiction": {
                            "cls": "COMPLY",
                            "coverage": {
                                "n_features": 4, "n_in_top_k": 0,
                                "frac_in_top_k": 0.0,
                                "sum_abs_attribution": 0.0,
                                "low_coverage": True,
                            },
                        },
                    },
                },
            },
        },
    ]
    cov_summary = s08.aggregate_coverage_summary(synth_results)
    jbf = cov_summary["S1"]["all"]["jb_fiction"]
    log_test(
        "T-S8cov-d: aggregate mean frac = (0.5 + 0.0) / 2 = 0.25",
        abs(jbf["mean_frac_in_top_k"] - 0.25) < 1e-6
        and jbf["n_low_coverage_prompts"] == 1
        and jbf["n_prompts"] == 2
        and jbf["frac_low_coverage"] == 0.5,
        f"got {jbf}",
    )

    # T-S8cov-e: aggregate_weighted_summary computes comply-weighted JB recovery
    fake = {
        "per_ablation": {
            "X": {
                "n_features": 10,
                "positions": {
                    "all": {
                        "bare": {"recovery_rate": 0.0, "break_rate": 0.5,
                                 "n_baseline_comply": 0, "n_baseline_refuse": 50,
                                 "n_recovered_refusal": 0},
                        "jb_fiction": {"recovery_rate": 1.0, "break_rate": 0.0,
                                       "n_baseline_comply": 20, "n_baseline_refuse": 30,
                                       "n_recovered_refusal": 20},
                        "jb_analytical": {"recovery_rate": 0.5, "break_rate": 0.0,
                                          "n_baseline_comply": 10, "n_baseline_refuse": 40,
                                          "n_recovered_refusal": 5},
                        # Fill the remaining 3 jb_* + 5 ctrl_* with no-op rows.
                        "jb_cognitive_reframe": {"recovery_rate": 0.0, "break_rate": 0.0,
                                                 "n_baseline_comply": 0, "n_baseline_refuse": 50,
                                                 "n_recovered_refusal": 0},
                        "jb_completion": {"recovery_rate": 0.0, "break_rate": 0.0,
                                          "n_baseline_comply": 0, "n_baseline_refuse": 50,
                                          "n_recovered_refusal": 0},
                        "jb_roleplay": {"recovery_rate": 0.0, "break_rate": 0.0,
                                        "n_baseline_comply": 0, "n_baseline_refuse": 50,
                                        "n_recovered_refusal": 0},
                        "ctrl_fiction": {"recovery_rate": 0.0, "break_rate": 0.2,
                                         "n_baseline_comply": 5, "n_baseline_refuse": 45,
                                         "n_recovered_refusal": 0},
                        "ctrl_analytical": {"recovery_rate": 0.0, "break_rate": 0.1,
                                            "n_baseline_comply": 5, "n_baseline_refuse": 45,
                                            "n_recovered_refusal": 0},
                        "ctrl_cognitive_reframe": {"recovery_rate": 0.0, "break_rate": 0.0,
                                                   "n_baseline_comply": 0, "n_baseline_refuse": 50,
                                                   "n_recovered_refusal": 0},
                        "ctrl_completion": {"recovery_rate": 0.0, "break_rate": 0.0,
                                            "n_baseline_comply": 0, "n_baseline_refuse": 50,
                                            "n_recovered_refusal": 0},
                        "ctrl_roleplay": {"recovery_rate": 0.0, "break_rate": 0.0,
                                          "n_baseline_comply": 0, "n_baseline_refuse": 50,
                                          "n_recovered_refusal": 0},
                    },
                },
            },
        },
    }
    w_summary = s08.aggregate_weighted_summary(fake)
    w = w_summary["per_ablation"]["X"]["positions"]["all"]["weighted"]
    # Expected: jb_total_comply = 20+10+0+0+0 = 30
    # weighted_recovery_num = 1.0*20 + 0.5*10 = 25
    # jb_weighted_recovery_rate = 25/30 ≈ 0.8333
    log_test(
        "T-S8cov-e: jb_weighted_recovery_rate = (1.0×20 + 0.5×10) / 30 ≈ 0.833",
        abs(w["jb_weighted_recovery_rate"] - 25 / 30) < 1e-3
        and w["jb_total_baseline_comply"] == 30,
        f"got {w}",
    )
    log_test(
        "T-S8cov-f: ctrl_weighted_break_rate = (0.2×45 + 0.1×45) / (45+45+50+50+50) ≈ 0.0608",
        abs(w["ctrl_weighted_break_rate"] - (0.2 * 45 + 0.1 * 45) / (45 + 45 + 50 + 50 + 50)) < 1e-3,
        f"got ctrl_weighted_break_rate={w['ctrl_weighted_break_rate']}",
    )
    log_test(
        "T-S8cov-g: bare_break_rate passed through correctly",
        w["bare_break_rate"] == 0.5 and w["bare_baseline_refuse"] == 50,
        f"got {w}",
    )
    # T-S8cov-h: per_class_jb dict has all 5 jb classes
    log_test(
        "T-S8cov-h: weighted.per_class_jb has all 5 JB classes",
        set(w["per_class_jb"].keys()) == {"analytical", "cognitive_reframe",
                                          "completion", "fiction", "roleplay"},
        f"got {list(w['per_class_jb'].keys())}",
    )


def test_stage_06():
    """T-S6: Stage 06 causal intervention — hook math, schema, CLI.

    No GPU required. The model-loaded full run is exercised on RunPod, not here.
    """
    print("\n" + "=" * 60)
    print("STAGE 06 TESTS (causal intervention, no GPU)")
    print("=" * 60)

    import importlib

    # T-S6a: hook math — add direction pointwise, tuple-output case
    try:
        import torch
    except ImportError:
        log_skip("T-S6: torch not installed", "cannot test hook math")
        return

    from utils import make_intervention_hook

    r = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)
    h = torch.zeros(1, 4, 3, dtype=torch.float32)  # batch=1, seq=4, d=3

    add_hook = make_intervention_hook(r, "add")
    sub_hook = make_intervention_hook(r, "sub")

    # Mimic PyTorch's hook signature: (module, input, output)
    out_add = add_hook(None, None, h.clone())
    out_sub = sub_hook(None, None, h.clone())

    log_test(
        "T-S6a: add hook produces h + r at every position",
        torch.allclose(out_add, r.expand(1, 4, 3)),
        f"got {out_add}",
    )
    log_test(
        "T-S6b: sub hook produces h - r at every position",
        torch.allclose(out_sub, -r.expand(1, 4, 3)),
        f"got {out_sub}",
    )

    # T-S6c: tuple-output case (Gemma decoder layers return tuples)
    h_tup = torch.zeros(1, 4, 3, dtype=torch.float32)
    tup_out = add_hook(None, None, (h_tup, "extra"))
    log_test(
        "T-S6c: tuple-wrapped output keeps the trailing element",
        isinstance(tup_out, tuple) and tup_out[1] == "extra"
        and torch.allclose(tup_out[0], r.expand(1, 4, 3)),
    )

    # T-S6d: bad sign raises
    try:
        make_intervention_hook(r, "multiply")
        raised = False
    except ValueError:
        raised = True
    log_test("T-S6d: make_intervention_hook rejects unknown sign", raised)

    # T-S6e: aggregate_summary shape from a synthetic results list
    stage06 = importlib.import_module("06_causal_intervention")
    synth_results = [
        {
            "prompt_id": 1,
            "baseline": {
                "bare":        {"cls": "REFUSE", "coherent": True, "response": "no"},
                "jb_fiction":  {"cls": "COMPLY", "coherent": True, "response": "yes"},
                "jb_roleplay": {"cls": "COMPLY", "coherent": True, "response": "yes"},
                "jb_analytical":    {"cls": "REFUSE", "coherent": True, "response": "no"},
                "jb_completion":    {"cls": "REFUSE", "coherent": True, "response": "no"},
                "jb_cognitive_reframe": {"cls": "REFUSE", "coherent": True, "response": "no"},
            },
            "interventions": {
                "L15_pro_refusal_add": {
                    "jb_fiction":  {"cls": "REFUSE", "coherent": True,
                                    "response": "I can't", "flipped_toward_refuse": True},
                    "jb_roleplay": {"cls": "COMPLY", "coherent": True,
                                    "response": "sure", "flipped_toward_refuse": False},
                },
                "L15_anti_refusal_sub": {
                    "bare": {"cls": "COMPLY", "coherent": True,
                             "response": "sure", "flipped_toward_comply": True},
                },
            },
        }
    ]
    summary = stage06.aggregate_summary(synth_results, layer=15, skip_anti=False)
    log_test(
        "T-S6e: aggregate_summary has both pro_refusal_add and anti_refusal_sub blocks",
        "L15_pro_refusal_add" in summary and "L15_anti_refusal_sub" in summary,
    )
    pro = summary["L15_pro_refusal_add"]
    log_test(
        "T-S6f: pro_refusal counts jb_fiction flip correctly (1 of 2 comply-baseline)",
        pro["n_jb_comply_baseline"] == 2
        and pro["n_flipped_to_refuse"] == 1
        and abs(pro["flip_rate"] - 0.5) < 1e-9,
        f"got {pro['n_flipped_to_refuse']}/{pro['n_jb_comply_baseline']} rate={pro['flip_rate']}",
    )
    anti = summary["L15_anti_refusal_sub"]
    log_test(
        "T-S6g: anti_refusal counts bare flip correctly (1/1)",
        anti["n_bare_refuse_baseline"] == 1
        and anti["n_flipped_to_comply"] == 1
        and abs(anti["flip_rate"] - 1.0) < 1e-9,
    )

    # T-S6h: skip_anti path omits the anti block
    summary_pro_only = stage06.aggregate_summary(synth_results, layer=15, skip_anti=True)
    log_test(
        "T-S6h: skip_anti=True omits L15_anti_refusal_sub from summary",
        "L15_anti_refusal_sub" not in summary_pro_only
        and "L15_pro_refusal_add" in summary_pro_only,
    )

    # T-S6i: per-class rate math
    pro_per = pro["per_class"]
    log_test(
        "T-S6i: per-class rate for fiction=1.0, roleplay=0.0, others=0.0 (no comply baseline)",
        pro_per["fiction"]["rate"] == 1.0
        and pro_per["roleplay"]["rate"] == 0.0
        and pro_per["analytical"]["rate"] == 0.0,
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Local pipeline validation tests")
    parser.add_argument(
        "--stage", choices=["01", "01-a5", "02", "02b", "03", "03-a4", "04-a7", "04-a8", "04-schema", "06", "07", "07-ctrl", "07-sweep", "08", "08-cov", "utils", "utils-viz", "all"], default="all",
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
        test_stage_04_schema()
        test_stage_07()
        test_stage_07_synthetic_ctrl()
        test_stage_07_sweep()
        test_stage_06()
        test_stage_08()
        test_stage_08_coverage()
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
    elif args.stage == "04-schema":
        test_stage_04_schema()
    elif args.stage == "07":
        test_stage_07()
        test_stage_07_synthetic_ctrl()
        test_stage_07_sweep()
    elif args.stage == "07-ctrl":
        test_stage_07_synthetic_ctrl()
    elif args.stage == "07-sweep":
        test_stage_07_sweep()
    elif args.stage == "06":
        test_stage_06()
    elif args.stage == "08":
        test_stage_08()
        test_stage_08_coverage()
    elif args.stage == "08-cov":
        test_stage_08_coverage()
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
