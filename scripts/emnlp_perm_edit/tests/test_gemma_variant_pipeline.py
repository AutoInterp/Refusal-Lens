"""CPU unit tests for the Gemma-variant pipeline (no GPU, no HF)."""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))                 # scripts/emnlp_perm_edit
sys.path.insert(0, str(HERE.parents[2] / "pipeline"))    # scripts/pipeline

from ensure_gemma_variant_directions import build_variant_directions  # noqa: E402
from verify_variant_nets import extract_nets, compare_nets  # noqa: E402


def _fake_r_full(d=2560, spike_dim=443, spike=-2790.53):
    torch.manual_seed(0)
    r = torch.randn(d) * 3.0
    r[spike_dim] = spike
    return r


def test_build_variant_directions():
    r = _fake_r_full()
    v = build_variant_directions(r)
    # all unit-normalized
    for name in ("full", "outlier", "complement"):
        assert abs(float(v[name].norm()) - 1.0) < 1e-5, name
    # outlier is nonzero only at 443
    nz = torch.nonzero(v["outlier"]).flatten().tolist()
    assert nz == [443], nz
    # complement zeros dim 443
    assert float(v["complement"][443]) == 0.0
    # the outlier is the dominant direction (on real Gemma data ‖outlier‖/‖full‖ ≈ 0.90;
    # the synthetic fixture's exact ratio depends on the noise scale, so assert dominance
    # rather than pinning to 0.90 — this is the real invariant build_variant_directions relies on).
    assert r[443].abs().item() / r.norm().item() > 0.85
    print("PASS test_build_variant_directions")


def test_extract_and_compare_nets():
    attribution = {"results": [
        {"prompt_idx": 0, "conditions": {
            "bare": {"graphs": {"single": {"net": 900.0}}},
            "jb_fiction": {"graphs": {"single": {"net": 500.0}}}}},
        {"prompt_idx": 1, "conditions": {
            "bare": {"graphs": {"single": {"net": 920.0}}}}},
    ]}
    recs = extract_nets(attribution)
    assert {"prompt_idx": 0, "condition": "bare", "net": 900.0} in recs
    assert len(recs) == 3
    ref = [{"prompt_idx": 0, "condition": "bare", "net": 908.0},
           {"prompt_idx": 0, "condition": "jb_fiction", "net": 480.0},
           {"prompt_idx": 1, "condition": "bare", "net": 915.0}]
    good = compare_nets(recs, ref)
    assert good["ok"] is True, good
    # a sign-flipped / wrong-magnitude run must fail the gate
    bad = compare_nets([{"prompt_idx": 0, "condition": "bare", "net": -48000.0},
                        {"prompt_idx": 1, "condition": "bare", "net": -47000.0}], ref)
    assert bad["ok"] is False, bad
    print("PASS test_extract_and_compare_nets")


from assemble_compare_frontend import parse_condition, build_compare_manifest  # noqa: E402

import importlib.util  # noqa: E402


def _load_push_module():
    path = HERE.parents[2] / "pipeline" / "push_graph_data.py"
    spec = importlib.util.spec_from_file_location("push_graph_data", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_resolve_run_name():
    push = _load_push_module()
    assert push.resolve_run_name("run_gemma_complement_L15", Path("/x/gemma_var_complement")) == "run_gemma_complement_L15"
    assert push.resolve_run_name(None, Path("/x/gemma_var_complement")) == "gemma_var_complement"
    print("PASS test_resolve_run_name")


def test_parse_condition():
    assert parse_condition("000_bare_single") == ("000", "bare")
    assert parse_condition("012_jb_fiction_single") == ("012", "jb_fiction")
    assert parse_condition("003_ctrl_analytical") == ("003", "ctrl_analytical")
    assert parse_condition("garbage") is None
    print("PASS test_parse_condition")


def test_build_compare_manifest_intersection():
    colA = {"label": "G-cmpl", "dir": "run_gemma_complement_L15/05_frontend",
            "model": "gemma", "target": "complement",
            "graphs": [{"slug": "000_bare_single", "prompt": "p0"},
                       {"slug": "000_jb_fiction_single", "prompt": "p0"}]}
    colB = {"label": "Qwen", "dir": "run_emnlp_qwen_L18_20260522/05_frontend",
            "model": "qwen", "target": "full",
            "graphs": [{"slug": "000_bare_single", "prompt": "p0"}]}  # no jb_fiction
    m = build_compare_manifest([colA, colB], title="t")
    assert [p["idx"] for p in m["prompts"]] == ["000"]
    assert m["conditions"] == ["bare"]                    # jb_fiction not in ALL columns
    assert m["columns"][0]["slugmap"]["000_bare"] == "000_bare_single"
    assert "000_jb_fiction" not in m["columns"][0]["slugmap"]
    print("PASS test_build_compare_manifest_intersection")


if __name__ == "__main__":
    test_build_variant_directions()
    test_extract_and_compare_nets()
    test_resolve_run_name()
    test_parse_condition()
    test_build_compare_manifest_intersection()
    print("ALL PASS")
