from pathlib import Path

PKG = Path(__file__).resolve().parents[1]


def test_gcg_optimize_invariants():
    src = (PKG / "gcg_optimize.py").read_text()
    for needle in ['import nanogcg', "google/gemma-3-4b-it", "prune_suffix",
                   "suffix_len",
                   '"per_prompt"', '"smoke"', '"--smoke"',
                   "use_prefix_cache=False", "from gcg_prep import",
                   "get_nonascii_toks", "allow_non_ascii=False", "_gcg_target"]:
        assert needle in src, needle
    # default L=30 per the spec (robust-alignment tier)
    assert "default=30" in src


def test_generate_matches_tejas_settings():
    src = (PKG / "generate.py").read_text()
    for needle in ["google/gemma-3-4b-it", "torch.bfloat16", "do_sample=False",
                   "max_new_tokens", "apply_chat_template", 'device_map="cuda"',
                   "sweep_k"]:   # bf16 (not Tejas's fp32): fp32 OOMs on 35k-token many-shot prompts
        assert needle in src, needle
