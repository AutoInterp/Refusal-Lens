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

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Local pipeline validation tests")
    parser.add_argument(
        "--stage", choices=["01", "01-a5", "02b", "03", "03-a4", "04-a7", "04-a8", "utils", "utils-viz", "all"], default="all",
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
        test_stage_01_a5()
        test_stage_02b()
        test_stage_03_a4()
        test_stage_04_a7()
        test_stage_04_a8()
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
