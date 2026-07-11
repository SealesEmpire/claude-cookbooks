"""Usage metering and cost attribution.

Cost attribution is built into the API: every session thread carries a
typed cumulative ``usage``, the primary thread (``parent_thread_id is
None``) is the coordinator, and child threads are the workers. This
module turns that into per-thread dollar attribution, a session report,
and a live CostTracker the orchestrator polls for its budget guard.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from research_service.config import PricingConfig


def total_input(usage) -> int:
    """All billed input tokens, cache traffic included."""
    cache = getattr(usage, "cache_creation", None)  # None with no cache activity
    return (
        usage.input_tokens
        + usage.cache_read_input_tokens
        + (cache.ephemeral_5m_input_tokens if cache else 0)
        + (cache.ephemeral_1h_input_tokens if cache else 0)
    )


def cost_usd(usage, model: str, pricing: PricingConfig) -> float:
    """Dollar cost of one thread's cumulative usage at a model's rates."""
    rates = pricing.rates(model)
    m = pricing.cache_multipliers
    cache = getattr(usage, "cache_creation", None)
    return (
        usage.input_tokens * rates.input
        + (cache.ephemeral_5m_input_tokens if cache else 0) * rates.input * m.ephemeral_5m_write
        + (cache.ephemeral_1h_input_tokens if cache else 0) * rates.input * m.ephemeral_1h_write
        + usage.cache_read_input_tokens * rates.input * m.read
        + usage.output_tokens * rates.output
    ) / 1e6


@dataclass(frozen=True)
class ThreadCost:
    thread_id: str
    is_primary: bool
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass(frozen=True)
class SessionCostReport:
    threads: tuple[ThreadCost, ...]

    @property
    def total_usd(self) -> float:
        return sum(t.cost_usd for t in self.threads)

    @property
    def primary(self) -> ThreadCost | None:
        return next((t for t in self.threads if t.is_primary), None)

    @property
    def workers(self) -> tuple[ThreadCost, ...]:
        return tuple(t for t in self.threads if not t.is_primary)

    @property
    def worker_input_share(self) -> float:
        """Fraction of input tokens billed at the worker rate."""
        total = sum(t.input_tokens for t in self.threads)
        if not total:
            return 0.0
        return sum(t.input_tokens for t in self.workers) / total

    def to_dict(self) -> dict:
        return {
            "total_usd": round(self.total_usd, 4),
            "worker_input_share": round(self.worker_input_share, 4),
            "threads": [
                {
                    "thread_id": t.thread_id,
                    "is_primary": t.is_primary,
                    "model": t.model,
                    "input_tokens": t.input_tokens,
                    "output_tokens": t.output_tokens,
                    "cost_usd": round(t.cost_usd, 4),
                }
                for t in self.threads
            ],
        }


def session_report(
    client,
    session_id: str,
    primary_model: str,
    worker_model_for: callable | None = None,
    default_worker_model: str = "",
    pricing: PricingConfig | None = None,
    betas: tuple[str, ...] = (),
) -> SessionCostReport:
    """Price every thread of a session.

    ``worker_model_for(thread) -> model`` resolves a child thread's model
    (thread objects don't carry the roster spec); when omitted, every
    child bills at ``default_worker_model``.
    """
    assert pricing is not None
    threads = list(client.beta.sessions.threads.list(session_id, betas=list(betas)))
    costs = []
    for t in threads:
        is_primary = t.parent_thread_id is None
        if is_primary:
            model = primary_model
        elif worker_model_for is not None:
            model = worker_model_for(t)
        else:
            model = default_worker_model
        costs.append(
            ThreadCost(
                thread_id=t.id,
                is_primary=is_primary,
                model=model,
                input_tokens=total_input(t.usage),
                output_tokens=t.usage.output_tokens,
                cost_usd=cost_usd(t.usage, model, pricing),
            )
        )
    return SessionCostReport(threads=tuple(costs))


@dataclass
class CostTracker:
    """Live cost tracking for a running session, with a budget guard.

    The orchestrator calls ``refresh()`` periodically; ``over_budget``
    flips when the cumulative session cost crosses ``budget_usd``.
    """

    client: object
    session_id: str
    primary_model: str
    default_worker_model: str
    pricing: PricingConfig
    betas: tuple[str, ...] = ()
    budget_usd: float | None = None
    worker_model_for: object = None
    last_report: SessionCostReport = field(default_factory=lambda: SessionCostReport(threads=()))

    def refresh(self) -> SessionCostReport:
        self.last_report = session_report(
            self.client,
            self.session_id,
            primary_model=self.primary_model,
            worker_model_for=self.worker_model_for,
            default_worker_model=self.default_worker_model,
            pricing=self.pricing,
            betas=self.betas,
        )
        return self.last_report

    @property
    def total_usd(self) -> float:
        return self.last_report.total_usd

    @property
    def over_budget(self) -> bool:
        return self.budget_usd is not None and self.total_usd > self.budget_usd
