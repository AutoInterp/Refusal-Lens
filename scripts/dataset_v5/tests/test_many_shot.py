import json

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


def test_load_comply_pool_filters_and_maps(tmp_path):
    from many_shot import load_comply_pool
    j = {"metadata": {}, "generations": [
        {"record_idx": 0, "class": "c", "kind": "attack", "judge": "COMPLY",
         "base_id": 5, "base": "q5?", "response": "a5", "n_chars": 2},
        {"record_idx": 1, "class": "c", "kind": "attack", "judge": "REFUSE",
         "base_id": 6, "base": "q6?", "response": "r6", "n_chars": 2},          # dropped: REFUSE
        {"record_idx": 2, "class": "c", "kind": "control", "judge": "COMPLY",
         "base_id": 7, "base": "q7?", "response": "a7", "n_chars": 2},          # dropped: not attack
        {"record_idx": 3, "class": "m", "kind": "attack", "judge": "COMPLY",
         "base_id": None, "base": "q8?", "response": "a8", "n_chars": 2},
    ]}
    p = tmp_path / "judged.json"
    p.write_text(json.dumps(j))
    pool = load_comply_pool([p])
    assert len(pool) == 2                                    # only the two COMPLY attack rows
    assert {r["record_idx"] for r in pool} == {0, 3}
    r0 = next(r for r in pool if r["record_idx"] == 0)
    assert r0 == {"src": "judged", "record_idx": 0, "base_id": 5, "base": "q5?", "response": "a5"}
    assert next(r for r in pool if r["record_idx"] == 3)["base_id"] is None


def test_load_comply_pool_class_filter_keeps_only_named_classes(tmp_path):
    # faithful-MSJ pool: keep only refusal_suppression COMPLY (genuinely-harmful demos),
    # drop the soft many_shot/gcg COMPLY rows. `base` stays the RAW question.
    from many_shot import load_comply_pool
    j = {"metadata": {}, "generations": [
        {"record_idx": 0, "class": "refusal_suppression", "kind": "attack",
         "judge": "COMPLY", "base_id": 1, "base": "raw q1?",
         "attack_text": "WRAPPED q1", "response": "Sure, here is <harmful a1>"},
        {"record_idx": 1, "class": "many_shot_icl", "kind": "attack", "judge": "COMPLY",
         "base_id": 2, "base": "raw q2?", "response": "soft deflection"},   # dropped: class
        {"record_idx": 2, "class": "gcg_per_prompt", "kind": "attack", "judge": "COMPLY",
         "base_id": 3, "base": "raw q3?", "response": "a3"},                # dropped: class
    ]}
    p = tmp_path / "v5_judged.json"
    p.write_text(json.dumps(j))
    pool = load_comply_pool([p], classes={"refusal_suppression"})
    assert len(pool) == 1
    assert pool[0]["base"] == "raw q1?"                       # raw question, not the wrapper
    assert pool[0]["response"] == "Sure, here is <harmful a1>"
