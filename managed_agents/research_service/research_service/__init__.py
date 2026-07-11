"""Research service: the coordinator pattern from CMA_plan_big_execute_small.ipynb as an application.

A frontier model plans and synthesizes; cheap workers do the web reading
in parallel session threads. This package lifts the notebook's inline
code into typed, config-driven, testable modules:

- ``config``        team roster + pricing loaded from YAML
- ``agents``        provision agents on the Managed Agents API from config
- ``events``        fold raw session events into typed progress events
- ``metering``      usage totals, cost attribution, live budget tracking
- ``briefs``        batch/split facts into worker briefs (delegation has a floor cost)
- ``store``         persist run state and distilled findings with source URLs
- ``orchestration`` drive a research run: stream with retry, budget guard, resume
- ``ab``            solo-vs-team A/B comparison as an ongoing regression check
- ``evals``         score answers against a gold set for coverage questions
- ``api``           FastAPI app streaming progress events to a browser frontend
"""

from research_service.config import PricingConfig, TeamConfig, load_pricing, load_team
from research_service.orchestration import BudgetExceededError, ResearchOrchestrator

__all__ = [
    "BudgetExceededError",
    "PricingConfig",
    "ResearchOrchestrator",
    "TeamConfig",
    "load_pricing",
    "load_team",
]
