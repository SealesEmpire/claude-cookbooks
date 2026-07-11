"""Provision agents on the Managed Agents API from a TeamConfig.

The roster is snapshotted when the coordinator is created — if you
change a worker's definition, re-provision the coordinator too, which is
why this module always creates workers first and the coordinator last.
"""

from __future__ import annotations

from dataclasses import dataclass

from research_service.config import SoloSpec, TeamConfig, WorkerSpec


def toolset_config(tools: tuple[str, ...] | list[str]) -> list[dict]:
    """Everything off except the named tools.

    Scoping is also the security boundary: workers read arbitrary
    (untrusted) web pages, so a worker that can only search, fetch, and
    report back is the blast radius you want for that input.
    """
    if not tools:
        return []
    return [
        {
            "type": "agent_toolset_20260401",
            "default_config": {"enabled": False},
            "configs": [{"name": name, "enabled": True} for name in tools],
        }
    ]


@dataclass(frozen=True)
class ProvisionedTeam:
    coordinator_id: str
    worker_ids: dict[str, str]  # worker name -> agent id
    solo_id: str | None = None


def create_worker(client, spec: WorkerSpec, betas: tuple[str, ...]) -> str:
    agent = client.beta.agents.create(
        name=spec.name,
        model=spec.model,
        tools=toolset_config(spec.tools),
        system=spec.system,
        betas=list(betas),
    )
    return agent.id


def create_solo(client, spec: SoloSpec, betas: tuple[str, ...]) -> str:
    agent = client.beta.agents.create(
        name=spec.name,
        model=spec.model,
        tools=toolset_config(spec.tools),
        system=spec.system,
        betas=list(betas),
    )
    return agent.id


def provision_team(client, team: TeamConfig, include_solo: bool = False) -> ProvisionedTeam:
    """Create every worker, then the coordinator with the full roster."""
    worker_ids = {spec.name: create_worker(client, spec, team.betas) for spec in team.workers}

    coordinator = client.beta.agents.create(
        name=team.coordinator.name,
        model=team.coordinator.model,
        multiagent={
            "type": "coordinator",
            "agents": [{"type": "agent", "id": agent_id} for agent_id in worker_ids.values()],
        },
        system=team.coordinator.rendered_system(),
        betas=list(team.betas),
    )

    solo_id = create_solo(client, team.solo, team.betas) if include_solo else None
    return ProvisionedTeam(coordinator_id=coordinator.id, worker_ids=worker_ids, solo_id=solo_id)
