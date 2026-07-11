"""Brief granularity: batch small facts, split large ones.

The notebook's caveat: delegation has a floor cost per worker thread, so
splitting the same work into more, narrower briefs raises the bill.
These helpers turn a flat list of facts to verify into briefs sized by a
configurable policy, and render them as an instruction block the
coordinator's prompt (or the user message) can carry.
"""

from __future__ import annotations

from dataclasses import dataclass

from research_service.config import BriefPolicy


@dataclass(frozen=True)
class Brief:
    """One delegation unit: a batch of facts for a single worker."""

    facts: tuple[str, ...]

    def render(self) -> str:
        return "\n".join(f"- {fact}" for fact in self.facts)


def plan_briefs(facts: list[str], policy: BriefPolicy) -> list[Brief]:
    """Group facts into briefs of at most ``target_batch_size``.

    A fact longer than ``max_brief_chars`` is a brief on its own — it is
    already a big reading job and batching it with others just serializes
    them behind it.
    """
    if policy.target_batch_size < 1:
        raise ValueError("target_batch_size must be >= 1")

    briefs: list[Brief] = []
    batch: list[str] = []
    for fact in facts:
        if len(fact) > policy.max_brief_chars:
            if batch:
                briefs.append(Brief(facts=tuple(batch)))
                batch = []
            briefs.append(Brief(facts=(fact,)))
            continue
        batch.append(fact)
        if len(batch) >= policy.target_batch_size:
            briefs.append(Brief(facts=tuple(batch)))
            batch = []
    if batch:
        briefs.append(Brief(facts=tuple(batch)))
    return briefs


def render_briefing(question: str, briefs: list[Brief]) -> str:
    """Render the question plus a suggested delegation split.

    The coordinator remains free to re-plan, but seeding the split keeps
    brief granularity at the configured optimum instead of whatever the
    model improvises.
    """
    if not briefs:
        return question
    lines = [question, "", "Suggested delegation split (one brief per worker):"]
    for i, brief in enumerate(briefs, 1):
        lines.append(f"Brief {i}:")
        lines.append(brief.render())
    return "\n".join(lines)
