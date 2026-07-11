"""Evaluation harness: score an answer against a gold set.

Coverage questions have a checkable shape — a fixed list of facts, each
of which must appear in the answer and carry an authoritative citation.
The gold set is YAML; the score is fraction of facts found plus fraction
of facts cited from the required domain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class GoldFact:
    """One required fact: the entity, values to find, required source domain."""

    entity: str
    expected: tuple[str, ...]  # substrings that must all appear near the entity
    source_domain: str = ""


@dataclass(frozen=True)
class GoldSet:
    question: str
    facts: tuple[GoldFact, ...]


@dataclass(frozen=True)
class FactScore:
    entity: str
    found: bool
    cited: bool


@dataclass(frozen=True)
class EvalResult:
    scores: tuple[FactScore, ...]

    @property
    def coverage(self) -> float:
        return sum(s.found for s in self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def citation_rate(self) -> float:
        return sum(s.cited for s in self.scores) / len(self.scores) if self.scores else 0.0


def load_gold(path: str | Path) -> GoldSet:
    raw = yaml.safe_load(Path(path).read_text())
    facts = tuple(
        GoldFact(
            entity=f["entity"],
            expected=tuple(f.get("expected", [])),
            source_domain=f.get("source_domain", ""),
        )
        for f in raw["facts"]
    )
    return GoldSet(question=raw["question"], facts=facts)


def _entity_window(answer: str, entity: str, window: int = 600) -> str:
    """The slice of the answer around the entity mention."""
    m = re.search(re.escape(entity), answer, re.IGNORECASE)
    if not m:
        return ""
    return answer[max(0, m.start() - window) : m.end() + window]


def score_answer(answer: str, gold: GoldSet) -> EvalResult:
    scores = []
    for fact in gold.facts:
        window = _entity_window(answer, fact.entity)
        found = bool(window) and all(exp.lower() in window.lower() for exp in fact.expected)
        cited = bool(window) and (
            not fact.source_domain or fact.source_domain.lower() in window.lower()
        )
        scores.append(FactScore(entity=fact.entity, found=found, cited=cited))
    return EvalResult(scores=tuple(scores))
