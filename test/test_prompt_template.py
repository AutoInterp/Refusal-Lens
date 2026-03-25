"""Tests for refusal_lens.prompt_template — prompt categories, library, and experiment sets."""
from __future__ import annotations

from refusal_lens.prompt_template import (
    PromptCategory,
    PromptTemplate,
    PromptTemplateLibrary,
)


# ---------------------------------------------------------------------------
# PromptCategory enum
# ---------------------------------------------------------------------------

class TestPromptCategory:
    def test_original_four_categories(self):
        assert PromptCategory.DIRECT_HARM.value == "Direct Harm"
        assert PromptCategory.CONTEXTUAL_EDUCATIONAL.value == "Contextual/Educational"
        assert PromptCategory.ROLEPLAY.value == "Roleplay"
        assert PromptCategory.HYPOTHETICAL_FICTIONAL.value == "Hypothetical/Fictional"

    def test_new_three_categories(self):
        assert PromptCategory.ROLEPLAY_JAILBREAK.value == "Roleplay Jailbreak"
        assert PromptCategory.MATCHED_HARMFUL.value == "Matched Harmful"
        assert PromptCategory.COMPLEX_BENIGN.value == "Complex Benign"

    def test_total_category_count(self):
        assert len(PromptCategory) == 7


# ---------------------------------------------------------------------------
# PromptTemplate dataclass
# ---------------------------------------------------------------------------

class TestPromptTemplate:
    def test_required_fields(self):
        pt = PromptTemplate(topic="t", category=PromptCategory.DIRECT_HARM, template="hello")
        assert pt.topic == "t"
        assert pt.template == "hello"

    def test_to_dict_includes_category_value(self):
        pt = PromptTemplate(
            topic="t",
            category=PromptCategory.ROLEPLAY_JAILBREAK,
            template="x",
        )
        d = pt.to_dict()
        assert d["category"] == "Roleplay Jailbreak"

    def test_format(self):
        pt = PromptTemplate(topic="t", category=PromptCategory.DIRECT_HARM, template="say {word}")
        assert pt.format(word="hi") == "say hi"


# ---------------------------------------------------------------------------
# Prompt lists on PromptTemplateLibrary
# ---------------------------------------------------------------------------

class TestPromptLists:
    def test_roleplay_jailbreak_count(self):
        assert len(PromptTemplateLibrary.ROLEPLAY_JAILBREAK_PROMPTS) == 8

    def test_matched_harmful_count(self):
        assert len(PromptTemplateLibrary.MATCHED_HARMFUL_PROMPTS) == 8

    def test_complex_benign_count(self):
        assert len(PromptTemplateLibrary.COMPLEX_BENIGN_PROMPTS) == 8

    def test_clean_harmless_count(self):
        assert len(PromptTemplateLibrary.CLEAN_HARMLESS_PROMPTS) == 8

    def test_all_prompts_are_nonempty_strings(self):
        for name in [
            "ROLEPLAY_JAILBREAK_PROMPTS",
            "MATCHED_HARMFUL_PROMPTS",
            "COMPLEX_BENIGN_PROMPTS",
            "CLEAN_HARMLESS_PROMPTS",
        ]:
            prompts = getattr(PromptTemplateLibrary, name)
            for i, p in enumerate(prompts):
                assert isinstance(p, str) and len(p) > 10, f"{name}[{i}] is too short"

    def test_lists_have_equal_length(self):
        """Jailbreak and matched harmful must be paired 1:1."""
        assert len(PromptTemplateLibrary.ROLEPLAY_JAILBREAK_PROMPTS) == len(
            PromptTemplateLibrary.MATCHED_HARMFUL_PROMPTS
        )

    def test_no_duplicates_within_lists(self):
        for name in [
            "ROLEPLAY_JAILBREAK_PROMPTS",
            "MATCHED_HARMFUL_PROMPTS",
            "COMPLEX_BENIGN_PROMPTS",
            "CLEAN_HARMLESS_PROMPTS",
        ]:
            prompts = getattr(PromptTemplateLibrary, name)
            assert len(prompts) == len(set(prompts)), f"duplicates in {name}"

    def test_no_overlap_between_lists(self):
        """The four prompt sets should be disjoint."""
        rp = set(PromptTemplateLibrary.ROLEPLAY_JAILBREAK_PROMPTS)
        mh = set(PromptTemplateLibrary.MATCHED_HARMFUL_PROMPTS)
        cb = set(PromptTemplateLibrary.COMPLEX_BENIGN_PROMPTS)
        ch = set(PromptTemplateLibrary.CLEAN_HARMLESS_PROMPTS)
        assert rp.isdisjoint(mh)
        assert rp.isdisjoint(cb)
        assert rp.isdisjoint(ch)
        assert mh.isdisjoint(cb)
        assert mh.isdisjoint(ch)
        assert cb.isdisjoint(ch)


# ---------------------------------------------------------------------------
# get_matched_pairs()
# ---------------------------------------------------------------------------

class TestGetMatchedPairs:
    def test_returns_list_of_tuples(self):
        lib = PromptTemplateLibrary()
        pairs = lib.get_matched_pairs()
        assert isinstance(pairs, list)
        for pair in pairs:
            assert isinstance(pair, tuple)
            assert len(pair) == 2

    def test_count_matches_prompt_lists(self):
        lib = PromptTemplateLibrary()
        pairs = lib.get_matched_pairs()
        assert len(pairs) == 8

    def test_first_element_is_jailbreak(self):
        lib = PromptTemplateLibrary()
        pairs = lib.get_matched_pairs()
        for jb, _ in pairs:
            assert jb in PromptTemplateLibrary.ROLEPLAY_JAILBREAK_PROMPTS

    def test_second_element_is_harmful(self):
        lib = PromptTemplateLibrary()
        pairs = lib.get_matched_pairs()
        for _, mh in pairs:
            assert mh in PromptTemplateLibrary.MATCHED_HARMFUL_PROMPTS

    def test_ordering_preserved(self):
        lib = PromptTemplateLibrary()
        pairs = lib.get_matched_pairs()
        for i, (jb, mh) in enumerate(pairs):
            assert jb == PromptTemplateLibrary.ROLEPLAY_JAILBREAK_PROMPTS[i]
            assert mh == PromptTemplateLibrary.MATCHED_HARMFUL_PROMPTS[i]


# ---------------------------------------------------------------------------
# get_experimental_prompts()
# ---------------------------------------------------------------------------

class TestGetExperimentalPrompts:
    def test_returns_dict(self):
        lib = PromptTemplateLibrary()
        exp = lib.get_experimental_prompts()
        assert isinstance(exp, dict)

    def test_has_four_keys(self):
        lib = PromptTemplateLibrary()
        exp = lib.get_experimental_prompts()
        assert set(exp.keys()) == {
            "roleplay_jailbreak",
            "matched_harmful",
            "clean_harmless",
            "complex_benign",
        }

    def test_each_value_is_list_of_strings(self):
        lib = PromptTemplateLibrary()
        exp = lib.get_experimental_prompts()
        for key, val in exp.items():
            assert isinstance(val, list), f"{key} is not a list"
            for item in val:
                assert isinstance(item, str), f"{key} contains non-string"

    def test_lengths_match_class_attributes(self):
        lib = PromptTemplateLibrary()
        exp = lib.get_experimental_prompts()
        assert len(exp["roleplay_jailbreak"]) == len(
            PromptTemplateLibrary.ROLEPLAY_JAILBREAK_PROMPTS
        )
        assert len(exp["matched_harmful"]) == len(
            PromptTemplateLibrary.MATCHED_HARMFUL_PROMPTS
        )
        assert len(exp["clean_harmless"]) == len(
            PromptTemplateLibrary.CLEAN_HARMLESS_PROMPTS
        )
        assert len(exp["complex_benign"]) == len(
            PromptTemplateLibrary.COMPLEX_BENIGN_PROMPTS
        )

    def test_returns_copies_not_originals(self):
        """Mutating the returned lists should not affect the class."""
        lib = PromptTemplateLibrary()
        exp = lib.get_experimental_prompts()
        original_first = PromptTemplateLibrary.ROLEPLAY_JAILBREAK_PROMPTS[0]
        exp["roleplay_jailbreak"][0] = "MUTATED"
        assert PromptTemplateLibrary.ROLEPLAY_JAILBREAK_PROMPTS[0] == original_first


# ---------------------------------------------------------------------------
# Expected outcomes for new categories
# ---------------------------------------------------------------------------

class TestExpectedOutcomes:
    def test_roleplay_jailbreak_outcome(self):
        lib = PromptTemplateLibrary()
        pt = lib.create_prompt(
            topic="test",
            category=PromptCategory.ROLEPLAY_JAILBREAK,
            topic_request="do something",
        )
        assert pt.expected_outcome == "Accept (Jailbreak Bypass)"

    def test_matched_harmful_outcome(self):
        lib = PromptTemplateLibrary()
        pt = lib.create_prompt(
            topic="test",
            category=PromptCategory.MATCHED_HARMFUL,
            topic_request="do something",
        )
        assert pt.expected_outcome == "Strong Refusal"

    def test_complex_benign_outcome(self):
        lib = PromptTemplateLibrary()
        pt = lib.create_prompt(
            topic="test",
            category=PromptCategory.COMPLEX_BENIGN,
            topic_request="do something",
        )
        assert pt.expected_outcome == "Accept (Harmless)"


# ---------------------------------------------------------------------------
# DEFAULT_TEMPLATES includes new categories
# ---------------------------------------------------------------------------

class TestDefaultTemplates:
    def test_new_categories_in_defaults(self):
        assert PromptCategory.ROLEPLAY_JAILBREAK in PromptTemplateLibrary.DEFAULT_TEMPLATES
        assert PromptCategory.MATCHED_HARMFUL in PromptTemplateLibrary.DEFAULT_TEMPLATES
        assert PromptCategory.COMPLEX_BENIGN in PromptTemplateLibrary.DEFAULT_TEMPLATES

    def test_all_seven_categories_have_templates(self):
        for cat in PromptCategory:
            assert cat in PromptTemplateLibrary.DEFAULT_TEMPLATES, f"{cat} missing from DEFAULT_TEMPLATES"


# ---------------------------------------------------------------------------
# generate_prompts_for_topic includes new categories
# ---------------------------------------------------------------------------

class TestGeneratePromptsForTopic:
    def test_generates_seven_prompts(self):
        lib = PromptTemplateLibrary()
        prompts = lib.generate_prompts_for_topic(
            topic="test",
            topic_request="do something",
            harmful_keywords=["test"],
        )
        assert len(prompts) == 7

    def test_all_categories_represented(self):
        lib = PromptTemplateLibrary()
        prompts = lib.generate_prompts_for_topic(
            topic="test",
            topic_request="do something",
            harmful_keywords=["test"],
        )
        categories = {p.category for p in prompts}
        assert categories == set(PromptCategory)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_original_framing_lists_unchanged(self):
        assert len(PromptTemplateLibrary.CONTEXT_FRAMINGS) == 4
        assert len(PromptTemplateLibrary.ROLE_FRAMINGS) == 4
        assert len(PromptTemplateLibrary.HYPOTHETICAL_FRAMINGS) == 4
        assert len(PromptTemplateLibrary.DEFAULT_EXPERT_ROLES) == 5

    def test_get_standard_prompts_still_works(self):
        lib = PromptTemplateLibrary()
        prompts = lib.get_standard_prompts()
        # 5 topics * 7 categories = 35
        assert len(prompts) == 35

    def test_create_prompt_direct_harm_unchanged(self):
        lib = PromptTemplateLibrary()
        pt = lib.create_prompt(
            topic="test",
            category=PromptCategory.DIRECT_HARM,
            topic_request="do something bad",
        )
        assert pt.expected_outcome == "Strong Refusal"
        assert pt.template == "do something bad"


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------

class TestPackageExports:
    def test_all_exported(self):
        import refusal_lens
        assert hasattr(refusal_lens, "PromptCategory")
        assert hasattr(refusal_lens, "PromptTemplate")
        assert hasattr(refusal_lens, "PromptTemplateLibrary")

    def test_in_all(self):
        import refusal_lens
        assert "PromptCategory" in refusal_lens.__all__
        assert "PromptTemplate" in refusal_lens.__all__
        assert "PromptTemplateLibrary" in refusal_lens.__all__
