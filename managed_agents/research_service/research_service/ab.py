"""Solo-vs-team A/B mode.

The notebook measured the split once, by hand. Here it is a repeatable
regression check: run the same question through the coordinator team and
through one frontier agent with the same tools, and report both bills
and both wall-clocks, so the cost/latency claims stay verified as models
and prices change.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from research_service.agents import create_solo
from research_service.config import PricingConfig, TeamConfig
from research_service.events import fold_event
from research_service.metering import SessionCostReport, session_report
from research_service.orchestration import ResearchOrchestrator
from research_service.store import RunState


@dataclass(frozen=True)
class ArmResult:
    answer: str
    duration_s: float
    cost: SessionCostReport


@dataclass(frozen=True)
class ABResult:
    team: ArmResult
    solo: ArmResult

    @property
    def cost_ratio(self) -> float:
        """solo / team — the notebook observed roughly 2.5x."""
        if not self.team.cost.total_usd:
            return float("inf")
        return self.solo.cost.total_usd / self.team.cost.total_usd

    @property
    def speed_ratio(self) -> float:
        """solo / team wall-clock — the notebook observed roughly 3x."""
        if not self.team.duration_s:
            return float("inf")
        return self.solo.duration_s / self.team.duration_s


def run_solo(
    client, team: TeamConfig, pricing: PricingConfig, environment_id: str, question: str
) -> ArmResult:
    """One frontier agent, same tools, same verification standard."""
    solo_id = create_solo(client, team.solo, team.betas)
    session = client.beta.sessions.create(
        agent=solo_id, environment_id=environment_id, betas=list(team.betas)
    )
    client.beta.sessions.events.send(
        session.id,
        betas=list(team.betas),
        events=[{"type": "user.message", "content": [{"type": "text", "text": question}]}],
    )

    answer = ""
    t0 = time.monotonic()
    with client.beta.sessions.events.stream(session.id, betas=list(team.betas)) as stream:
        for raw in stream:
            progress = fold_event(raw)
            if progress is None:
                continue
            if progress.kind == "coordinator_message":
                answer = progress.text
            elif progress.kind == "run_finished":
                break
    duration = time.monotonic() - t0

    cost = session_report(
        client,
        session.id,
        primary_model=team.solo.model,
        default_worker_model=team.solo.model,
        pricing=pricing,
        betas=team.betas,
    )
    return ArmResult(answer=answer, duration_s=duration, cost=cost)


def run_ab(orchestrator: ResearchOrchestrator, question: str) -> ABResult:
    """Run both arms on the same question and compare."""
    t0 = time.monotonic()
    team_state: RunState = orchestrator.run(question)
    team_duration = time.monotonic() - t0

    team_cost = session_report(
        orchestrator.client,
        team_state.session_id,
        primary_model=orchestrator.team.coordinator.model,
        worker_model_for=orchestrator._worker_model_for,
        default_worker_model=orchestrator.team.workers[0].model,
        pricing=orchestrator.pricing,
        betas=orchestrator.team.betas,
    )
    team = ArmResult(answer=team_state.final_answer, duration_s=team_duration, cost=team_cost)

    solo = run_solo(
        orchestrator.client,
        orchestrator.team,
        orchestrator.pricing,
        environment_id=team_state.environment_id,
        question=question,
    )
    return ABResult(team=team, solo=solo)
