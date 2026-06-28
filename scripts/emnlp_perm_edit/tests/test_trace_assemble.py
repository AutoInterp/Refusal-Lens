"""Integration smoke for the trace assembler against the real complement graphs."""
import gzip, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/pipeline"))
from assemble_trace_frontend import build_pair_entry, load_graph  # noqa: E402

GD = ROOT / "data/results/compare_3way/run_gemma_complement_L15/05_frontend/graph_data"


def test_build_pair_entry_on_real_roleplay_flip():
    if not GD.exists():
        import pytest; pytest.skip("complement graphs not on disk")
    bare = load_graph(GD / "004_bare_single.json.gz")
    jb = load_graph(GD / "004_jb_roleplay_single.json.gz")
    cfg = {"top_n": 20, "delta": 0.30, "model_token_gate": False}
    pair, baked_bare, baked_jb = build_pair_entry(4, "jb_roleplay", "social engineering", bare, jb, cfg)
    assert pair["bare_slug"] == "004_bare_single" and pair["jb_slug"] == "004_jb_roleplay_single"
    # at least one refusal-centric feature exists in a real refusal graph
    classes = {n.get("rl_trace_class") for n in baked_jb["nodes"] if n.get("feature_type") == "cross layer transcoder"}
    assert "refusal_centric" in classes or "suppression" in classes
    assert any(r["class"] != "neutral" for r in pair["evidence"])
