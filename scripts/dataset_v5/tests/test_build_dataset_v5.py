from build_dataset_v5 import build_records, build_sweep

BASES = [{"base_id": i, "base": f"harmful {i}?", "topic": "fraud"} for i in range(1, 6)]
POOL = [{"src": "v3_judged", "record_idx": i, "base_id": i,
         "base": f"demo q {i}?", "response": f"demo a {i}."} for i in range(1, 41)]
GCG = {
    "per_prompt": {str(i): {"suffix": f"SUF{i}", "suffix_pruned": f"SP{i}",
                            "pruned_n_tokens": 5, "final_loss": 0.1, "target": f"T{i}"}
                   for i in range(1, 6)},
    "config": {"suffix_len": 30, "num_steps": 500},
}
COMMON = {"base_id", "base", "topic", "class", "class_type",
          "attack_text", "control", "source", "faithful"}


def test_four_records_per_base_with_gcg():
    recs = build_records(BASES, POOL, GCG, k=4, seed=0)
    assert len(recs) == 20                        # 5 bases x 4 classes
    assert [r["class"] for r in recs[:4]] == \
        ["gcg_per_prompt", "many_shot_icl", "refusal_suppression", "refusal_suppression_prefill"]
    for r in recs:
        assert COMMON <= set(r)
        assert r["class_type"] == "sota_transform"
        assert r["control"] == {"type": "floor_ceiling"}


def test_gcg_attack_text_uses_full_suffix():
    recs = build_records(BASES, POOL, GCG, k=4, seed=0)
    pp = [r for r in recs if r["class"] == "gcg_per_prompt"]
    assert pp[0]["attack_text"] == "harmful 1? SUF1"
    assert pp[0]["gcg"]["suffix_pruned"] == "SP1"
    assert len(pp) == 5 and len({r["attack_text"] for r in pp}) == 5   # distinct per prompt


def test_gcg_record_carries_target():
    recs = build_records(BASES, POOL, GCG, k=4, seed=0)
    pp = [r for r in recs if r["class"] == "gcg_per_prompt"][0]
    assert pp["attack_text"] == "harmful 1? SUF1"
    assert pp["gcg"]["target"] == "T1"


def test_many_shot_record_shape():
    recs = build_records(BASES, POOL, GCG, k=4, seed=0)
    ms = [r for r in recs if r["class"] == "many_shot_icl"][0]
    assert ms["many_shot"]["n_shots"] == 4
    assert len(ms["many_shot"]["shot_refs"]) == 4
    assert ms["attack_text"].rstrip().endswith("Assistant:")


def test_refusal_suppression_shape():
    recs = build_records(BASES, POOL, GCG, k=4, seed=0)
    rs = [r for r in recs if r["class"] == "refusal_suppression"][0]
    assert "harmful 1?" in rs["attack_text"]                     # base embedded
    assert 'Begin your response with exactly: "Sure, here is"' in rs["attack_text"]
    assert "cannot" in rs["attack_text"]                         # the rule list
    assert "prefill" not in rs                                   # natural tier: no prefill
    pf = [r for r in recs if r["class"] == "refusal_suppression_prefill"][0]
    assert pf["attack_text"] == rs["attack_text"]                # same wrapper
    assert pf["prefill"] == "Sure, here is"                      # forced tier


def test_placeholder_mode_without_gcg():
    recs = build_records(BASES, POOL, None, k=4, seed=0)
    pp = [r for r in recs if r["class"] == "gcg_per_prompt"]
    assert pp[0]["attack_text"] == "harmful 1?"           # base only, no suffix
    assert pp[0]["gcg"]["suffix"] == ""


def test_limit_scopes_bases():
    recs = build_records(BASES, POOL, GCG, k=4, seed=0, limit=2)
    assert len({r["base_id"] for r in recs}) == 2
    assert len(recs) == 8                         # 2 bases x 4 classes


def test_sweep_is_nested_and_many_shot_only():
    # ks=(4,8) straddles random.sample's internal-algorithm boundary. This test loops
    # over EVERY swept base (not just base_id 1, which nests coincidentally under the old
    # rng.sample too) so it genuinely fails if the Step-0 shuffle+slice fix is reverted:
    # base_id 2 is non-nested under rng.sample but nested under shuffle+slice.
    sweep = build_sweep(BASES, POOL, ks=(4, 8), n_bases=3, seed=0)
    assert {r["class"] for r in sweep} == {"many_shot_icl"}
    assert {r["sweep_k"] for r in sweep} == {4, 8}
    for bid in (1, 2, 3):
        byk = {r["sweep_k"]: r for r in sweep if r["base_id"] == bid}
        refs4 = [x["record_idx"] for x in byk[4]["many_shot"]["shot_refs"]]
        refs8 = [x["record_idx"] for x in byk[8]["many_shot"]["shot_refs"]]
        assert refs8[:4] == refs4, f"base {bid} shot sets not nested"
