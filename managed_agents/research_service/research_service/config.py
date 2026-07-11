"""Typed configuration for the research service.

The notebook hardcoded models, prompts, toolsets, and prices inline.
Here they load from two YAML files — ``config/team.yaml`` (the roster)
and ``config/pricing.yaml`` (the versioned pricing table) — so teams can
be re-composed and prices updated without code changes.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
PRICING_MAX_AGE_DAYS = 180

logger = logging.getLogger("research_service")


class ConfigError(ValueError):
    """Raised when a config file is missing required fields or malformed."""


@dataclass(frozen=True)
class WorkerSpec:
    """One worker type in the coordinator's roster."""

    name: str
    model: str
    system: str
    tools: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class CoordinatorSpec:
    """The coordinator: no tools of its own, only a roster."""

    name: str
    model: str
    system: str
    max_parallel_workers: int = 6

    def rendered_system(self) -> str:
        return self.system.replace("{max_parallel_workers}", str(self.max_parallel_workers))


@dataclass(frozen=True)
class SoloSpec:
    """The A/B baseline: one frontier agent with the workers' tools."""

    name: str
    model: str
    system: str
    tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class BriefPolicy:
    """Brief granularity knobs — delegation has a floor cost."""

    target_batch_size: int = 4
    max_brief_chars: int = 600


@dataclass(frozen=True)
class TeamConfig:
    betas: tuple[str, ...]
    coordinator: CoordinatorSpec
    workers: tuple[WorkerSpec, ...]
    solo: SoloSpec
    briefs: BriefPolicy = field(default_factory=BriefPolicy)

    def worker(self, name: str) -> WorkerSpec:
        for w in self.workers:
            if w.name == name:
                return w
        raise KeyError(name)


@dataclass(frozen=True)
class ModelRates:
    """$ / MTok for one model."""

    input: float
    output: float


@dataclass(frozen=True)
class CacheMultipliers:
    """Multipliers applied to the input rate for cache activity."""

    ephemeral_5m_write: float = 1.25
    ephemeral_1h_write: float = 2.0
    read: float = 0.1


@dataclass(frozen=True)
class PricingConfig:
    version: str
    models: dict[str, ModelRates]
    cache_multipliers: CacheMultipliers = field(default_factory=CacheMultipliers)
    source: str = ""

    def rates(self, model: str) -> ModelRates:
        try:
            return self.models[model]
        except KeyError:
            raise ConfigError(
                f"no pricing configured for model {model!r}; add it to pricing.yaml"
            ) from None


def _require(mapping: dict, key: str, where: str):
    if key not in mapping:
        raise ConfigError(f"missing required key {key!r} in {where}")
    return mapping[key]


def load_team(path: str | Path | None = None) -> TeamConfig:
    """Load and validate the team roster from YAML.

    Model names may be overridden with the same environment variables the
    notebook used: ``COOKBOOK_COORDINATOR_MODEL`` and ``COOKBOOK_WORKER_MODEL``.
    """
    path = Path(path) if path else DEFAULT_CONFIG_DIR / "team.yaml"
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} did not parse to a mapping")

    coordinator_model_override = os.environ.get("COOKBOOK_COORDINATOR_MODEL")
    worker_model_override = os.environ.get("COOKBOOK_WORKER_MODEL")

    c = _require(raw, "coordinator", str(path))
    coordinator = CoordinatorSpec(
        name=_require(c, "name", "coordinator"),
        model=coordinator_model_override or _require(c, "model", "coordinator"),
        system=_require(c, "system", "coordinator"),
        max_parallel_workers=int(c.get("max_parallel_workers", 6)),
    )

    workers = []
    for w in _require(raw, "workers", str(path)):
        workers.append(
            WorkerSpec(
                name=_require(w, "name", "worker"),
                model=worker_model_override or _require(w, "model", w.get("name", "worker")),
                system=_require(w, "system", w.get("name", "worker")),
                tools=tuple(w.get("tools", []) or []),
                description=w.get("description", ""),
            )
        )
    if not workers:
        raise ConfigError("team.yaml must define at least one worker")

    s = _require(raw, "solo", str(path))
    solo = SoloSpec(
        name=_require(s, "name", "solo"),
        model=coordinator_model_override or _require(s, "model", "solo"),
        system=_require(s, "system", "solo"),
        tools=tuple(s.get("tools", []) or []),
    )

    b = raw.get("briefs", {}) or {}
    briefs = BriefPolicy(
        target_batch_size=int(b.get("target_batch_size", 4)),
        max_brief_chars=int(b.get("max_brief_chars", 600)),
    )

    return TeamConfig(
        betas=tuple(raw.get("betas", []) or []),
        coordinator=coordinator,
        workers=tuple(workers),
        solo=solo,
        briefs=briefs,
    )


def _warn_if_stale(version: str, path: Path, max_age_days: int) -> None:
    """Warn when the pricing table's version date has drifted too old."""
    try:
        as_of = dt.date.fromisoformat(version)
    except ValueError:
        return  # version isn't a date; nothing to check
    age = (dt.date.today() - as_of).days
    if age > max_age_days:
        logger.warning(
            "pricing table %s is %d days old (version %s); "
            "check the pricing page and bump `version`",
            path,
            age,
            version,
        )


def load_pricing(
    path: str | Path | None = None, max_age_days: int = PRICING_MAX_AGE_DAYS
) -> PricingConfig:
    """Load and validate the versioned pricing table from YAML.

    Warns when the table's ``version`` date is older than ``max_age_days``,
    so silently drifting prices get noticed.
    """
    path = Path(path) if path else DEFAULT_CONFIG_DIR / "pricing.yaml"
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} did not parse to a mapping")

    models = {}
    for name, rates in _require(raw, "models", str(path)).items():
        models[name] = ModelRates(
            input=float(_require(rates, "input", f"pricing for {name}")),
            output=float(_require(rates, "output", f"pricing for {name}")),
        )

    m = raw.get("cache_multipliers", {}) or {}
    multipliers = CacheMultipliers(
        ephemeral_5m_write=float(m.get("ephemeral_5m_write", 1.25)),
        ephemeral_1h_write=float(m.get("ephemeral_1h_write", 2.0)),
        read=float(m.get("read", 0.1)),
    )

    version = str(_require(raw, "version", str(path)))
    _warn_if_stale(version, path, max_age_days)

    return PricingConfig(
        version=version,
        models=models,
        cache_multipliers=multipliers,
        source=raw.get("source", ""),
    )
