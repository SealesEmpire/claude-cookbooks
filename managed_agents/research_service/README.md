# Research service: the coordinator pattern as an application

[`CMA_plan_big_execute_small.ipynb`](../CMA_plan_big_execute_small.ipynb) showed the economics once, inline: a frontier model plans and synthesizes, cheap workers do the web reading in parallel session threads, and the split came out roughly 2.5x cheaper and 3x faster than a solo frontier agent held to the same verification standard. This directory develops that notebook into a small production-shaped service (the way [`roadtrip_planner/`](../roadtrip_planner/) does for the session-streaming notebooks).

```
browser ──▶ POST /runs ─────────▶ orchestrator.start(): provision team from config,
   ▲                              create environment + session, send the question
   │ one EventSource
   └── GET /runs/{id}/stream ◀─── orchestrator.stream(): fold session events into
                                  progress events, capture findings + source URLs,
                                  poll cumulative thread usage for the budget guard
```

## What moved out of the notebook, and where

| Notebook inline code | Module | What it became |
| --- | --- | --- |
| Hardcoded models, prompts, toolsets | [`config/team.yaml`](config/team.yaml), [`research_service/config.py`](research_service/config.py) | A validated, re-composable roster. The single search worker became a heterogeneous team — premise-checker, search-worker, fetch-extract-worker, fact-checker, citation-formatter — and the premise-checker runs **before** fan-out, closing the notebook's Kings Canyon caveat (the facts were audited; the decomposition wasn't). |
| `PRICES = {...}` dict | [`config/pricing.yaml`](config/pricing.yaml) | A versioned pricing table with explicit cache-write/read multipliers, because `/v1/models` reports capabilities but not pricing. |
| `agents.create(...)` calls | [`research_service/agents.py`](research_service/agents.py) | Provisioning from config: workers first, coordinator last, because the roster is snapshotted at coordinator creation. |
| The `match ev.type` loop | [`research_service/events.py`](research_service/events.py) | One pure `fold_event` function shared by the orchestrator, the SSE endpoint, and the tests. |
| `total_input` / `cost` / `report` | [`research_service/metering.py`](research_service/metering.py) | Per-thread cost attribution (primary = coordinator, children = workers, priced per roster model) and a live `CostTracker` with a budget guard. |
| — (notebook caveat: "brief granularity has an optimum") | [`research_service/briefs.py`](research_service/briefs.py) | Configurable batching: small facts batch up to `target_batch_size` per brief, any fact over `max_brief_chars` gets its own brief. |
| — | [`research_service/store.py`](research_service/store.py) | Run persistence (resume by run id instead of restarting) and distilled findings with extracted source URLs, so every claim in the answer is auditable. |
| The streaming cell | [`research_service/orchestration.py`](research_service/orchestration.py) | Start / stream / resume with retry on transient stream failures and an abort-if-over-budget guard. Worker-level retry and the parallelism cap ride in the coordinator's prompt (config), because the server does the actual coordination. |
| The solo-agent comparison cells | [`research_service/ab.py`](research_service/ab.py) | A repeatable solo-vs-team A/B mode, so the cost/latency claims stay verified as models and prices change. |
| — | [`research_service/evals.py`](research_service/evals.py), [`evals/gold_national_parks.yaml`](evals/gold_national_parks.yaml) | A gold-set harness for coverage questions: fraction of facts found, fraction cited from the required domain. |
| — | [`research_service/api.py`](research_service/api.py), [`frontend/index.html`](frontend/index.html) | FastAPI + SSE, and a single-file frontend rendering the delegation graph, running cost, findings with sources, and the final answer live. |
| — | [`research_service/logging_utils.py`](research_service/logging_utils.py) | Every progress event, retry, and lifecycle transition as one JSON log line. |

## Run it

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and an Anthropic API key with access to the Managed Agents beta.

```bash
cd managed_agents/research_service
cp .env.example .env          # add ANTHROPIC_API_KEY; optionally a budget
uv sync
uv run uvicorn research_service.api:app --reload
# open http://127.0.0.1:8000 and ask a coverage question
```

The service is meant to run locally: it has no authentication, so don't expose it to untrusted networks as-is.

HTTP surface:

| Endpoint | Effect |
| --- | --- |
| `POST /runs` | start a run (`question` required, ≤4000 chars; optional `facts`) |
| `GET /runs` | run history: id, status, question, created_at, total cost |
| `GET /runs/{id}` | full persisted state of one run |
| `GET /runs/{id}/stream` | SSE progress stream; safe to open from several tabs — one driver per run fans events out, and terminal runs replay their state |
| `POST /runs/{id}/cancel` | abort a run and archive its environment |
| `GET /health` | liveness check |

Each run provisions its own cloud environment; it's archived automatically when the run finishes, aborts, is cancelled, or fails.

Environment knobs:

| Variable | Effect |
| --- | --- |
| `ANTHROPIC_API_KEY` | required |
| `RESEARCH_BUDGET_USD` | abort any run whose cumulative session cost crosses this |
| `RESEARCH_RUN_DIR` | where run state persists (default `.runs`) |
| `COOKBOOK_COORDINATOR_MODEL` / `COOKBOOK_WORKER_MODEL` | override the models in `team.yaml`, same variables the notebook used |

From Python:

```python
import anthropic
from research_service import ResearchOrchestrator, load_pricing, load_team
from research_service.store import RunStore

orchestrator = ResearchOrchestrator(
    client=anthropic.Anthropic(),
    team=load_team(),
    pricing=load_pricing(),
    store=RunStore(".runs"),
    budget_usd=10.0,
)
state = orchestrator.run("For each of the ten largest national parks ...")
print(state.final_answer, state.cost_report["total_usd"])
```

A/B regression check and eval scoring:

```python
from research_service.ab import run_ab
from research_service.evals import load_gold, score_answer

result = run_ab(orchestrator, question)
print(f"solo/team cost ratio: {result.cost_ratio:.1f}x, speed: {result.speed_ratio:.1f}x")

gold = load_gold("evals/gold_national_parks.yaml")
print(score_answer(result.team.answer, gold).coverage)
```

## Tests

The whole orchestration layer is tested against a scripted fake of the Managed Agents client (`tests/conftest.py`) — no API key, no network:

```bash
uv run pytest
```

CI runs the same suite via [`.github/workflows/research-service-tests.yml`](../../.github/workflows/research-service-tests.yml).
