"""Eval harness: scoring answers against the gold set."""

from pathlib import Path

from research_service.evals import GoldFact, GoldSet, load_gold, score_answer

GOLD_PATH = Path(__file__).resolve().parent.parent / "evals" / "gold_national_parks.yaml"


def test_gold_set_loads():
    gold = load_gold(GOLD_PATH)
    assert len(gold.facts) == 10
    assert any(f.entity == "Great Smoky Mountains" for f in gold.facts)
    assert all(f.source_domain == "nps.gov" for f in gold.facts)


def test_perfect_answer_scores_full():
    gold = GoldSet(
        question="q",
        facts=(
            GoldFact(entity="Death Valley", expected=("$30",), source_domain="nps.gov"),
            GoldFact(entity="Yellowstone", expected=("$35",), source_domain="nps.gov"),
        ),
    )
    answer = (
        "Death Valley: $30 per vehicle (https://www.nps.gov/deva/planyourvisit/fees.htm)\n"
        "Yellowstone: $35 per vehicle (https://www.nps.gov/yell/planyourvisit/fees.htm)\n"
    )
    result = score_answer(answer, gold)
    assert result.coverage == 1.0
    assert result.citation_rate == 1.0


def test_missing_entity_and_missing_citation():
    gold = GoldSet(
        question="q",
        facts=(
            GoldFact(entity="Death Valley", expected=("$30",), source_domain="nps.gov"),
            GoldFact(entity="Yellowstone", expected=("$35",), source_domain="nps.gov"),
        ),
    )
    answer = "Death Valley costs $30 per vehicle according to a travel blog."
    result = score_answer(answer, gold)
    assert result.coverage == 0.5  # Yellowstone absent
    assert result.citation_rate == 0.0  # no nps.gov near Death Valley either


def test_wrong_value_not_counted():
    gold = GoldSet(
        question="q",
        facts=(GoldFact(entity="Yellowstone", expected=("$35",), source_domain=""),),
    )
    result = score_answer("Yellowstone entry is $20.", gold)
    assert result.coverage == 0.0
    # entity was mentioned and no domain required, so citation passes
    assert result.citation_rate == 1.0
