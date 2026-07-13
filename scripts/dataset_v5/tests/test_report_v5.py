from report_v5 import comply_table, inspect_lines


def _judged():
    def row(cls, judge):
        return {"class": cls, "kind": "attack", "judge": judge,
                "base_id": 1, "base": "q?", "prompt_text": "P", "response": "R"}
    return [row("gcg_per_prompt", "COMPLY"), row("gcg_per_prompt", "REFUSE"),
            row("many_shot_icl", "COMPLY"), row("many_shot_icl", "COMPLY")]


def test_comply_table_rates():
    t = comply_table(_judged())
    assert t["gcg_per_prompt"] == {"comply": 1, "total": 2, "rate": 50.0}
    assert t["many_shot_icl"]["rate"] == 100.0


def test_inspect_lines_show_class_and_heads():
    gens = [{"class": "many_shot_icl", "base_id": 3,
             "attack_text": "A" * 999, "response": "B" * 999}]
    lines = inspect_lines(gens, head=50)
    blob = "\n".join(lines)
    assert "many_shot_icl" in blob and "base_id=3" in blob
    assert "A" * 50 in blob and "A" * 51 not in blob


def test_tier_summary():
    from report_v5 import tier_summary
    def row(base_id, cls, judge):
        return {"class": cls, "kind": "attack", "judge": judge, "base_id": base_id}
    recs = [
        row(1, "refusal_suppression", "COMPLY"), row(1, "refusal_suppression_prefill", "COMPLY"),
        row(2, "refusal_suppression", "REFUSE"), row(2, "refusal_suppression_prefill", "COMPLY"),
        row(3, "gcg_per_prompt", "REFUSE"),      row(3, "refusal_suppression_prefill", "REFUSE"),
    ]
    assert tier_summary(recs) == {"natural_landed": 1, "prefill_only": 1, "no_comply": 1}
