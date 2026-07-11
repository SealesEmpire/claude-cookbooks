"""Brief batching and splitting."""

import pytest
from research_service.briefs import plan_briefs, render_briefing
from research_service.config import BriefPolicy


def test_batches_small_facts():
    facts = [f"fact {i}" for i in range(10)]
    briefs = plan_briefs(facts, BriefPolicy(target_batch_size=4, max_brief_chars=600))
    assert [len(b.facts) for b in briefs] == [4, 4, 2]


def test_oversize_fact_gets_own_brief():
    big = "x" * 700
    facts = ["a", "b", big, "c"]
    briefs = plan_briefs(facts, BriefPolicy(target_batch_size=4, max_brief_chars=600))
    assert [b.facts for b in briefs] == [("a", "b"), (big,), ("c",)]


def test_all_facts_preserved_in_order():
    facts = [f"f{i}" for i in range(7)]
    briefs = plan_briefs(facts, BriefPolicy(target_batch_size=3, max_brief_chars=600))
    assert [f for b in briefs for f in b.facts] == facts


def test_invalid_batch_size():
    with pytest.raises(ValueError):
        plan_briefs(["a"], BriefPolicy(target_batch_size=0))


def test_render_briefing_with_briefs():
    briefs = plan_briefs(["fee for park A", "fee for park B"], BriefPolicy(target_batch_size=1))
    text = render_briefing("The question.", briefs)
    assert text.startswith("The question.")
    assert "Brief 1:" in text and "Brief 2:" in text
    assert "- fee for park A" in text


def test_render_briefing_without_briefs_is_question():
    assert render_briefing("Just the question.", []) == "Just the question."
