import pytest
from many_shot import assemble_many_shot


def _pool(n=40):
    # base_id 100 is the "target" collision; ids 1..n are demos
    return [{"src": "v3_judged", "record_idx": i, "base_id": i,
             "base": f"harmful question {i}?", "response": f"compliant answer {i}."}
            for i in range(1, n + 1)]


def _target():
    return {"base_id": 7, "base": "TARGET harmful request?", "topic": "fraud"}


def test_k_demos_plus_target_last():
    text, refs = assemble_many_shot(_target(), _pool(), k=5, seed=0)
    assert len(refs) == 5
    assert text.count("User:") == 6            # 5 demos + 1 target
    assert text.count("Assistant:") == 6
    assert text.rstrip().endswith("Assistant:")  # generation continues from here
    assert "TARGET harmful request?" in text.split("User:")[-1]


def test_excludes_same_base_id():
    tgt = _target()
    _, refs = assemble_many_shot(tgt, _pool(), k=10, seed=0)
    assert all(r["base_id"] != tgt["base_id"] for r in refs)


def test_deterministic_under_seed():
    a, ra = assemble_many_shot(_target(), _pool(), k=8, seed=0)
    b, rb = assemble_many_shot(_target(), _pool(), k=8, seed=0)
    assert a == b and ra == rb


def test_seed_changes_selection():
    _, ra = assemble_many_shot(_target(), _pool(), k=8, seed=0)
    _, rb = assemble_many_shot(_target(), _pool(), k=8, seed=1)
    assert [r["record_idx"] for r in ra] != [r["record_idx"] for r in rb]


def test_demo_char_cap_truncates():
    pool = [{"src": "s", "record_idx": 1, "base_id": 1,
             "base": "q?", "response": "X" * 500}]
    text, _ = assemble_many_shot(_target(), pool, k=1, seed=0, demo_char_cap=100)
    # the demo answer is capped to 100 X's (target has no answer body)
    assert "X" * 100 in text and "X" * 101 not in text


def test_raises_when_pool_too_small():
    with pytest.raises(ValueError):
        assemble_many_shot(_target(), _pool(3), k=10, seed=0)
