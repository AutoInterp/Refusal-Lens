from base_prompts import load_base_prompts


def test_loads_fifty_bases_1_to_50():
    bases = load_base_prompts()
    assert len(bases) == 50
    assert [b["base_id"] for b in bases] == list(range(1, 51))
    first = bases[0]
    assert set(first) == {"base_id", "base", "topic"}
    assert isinstance(first["base"], str) and first["base"]
    assert isinstance(first["topic"], str) and first["topic"]
