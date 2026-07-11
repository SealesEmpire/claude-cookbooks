"""Drive a research run end to end.

The Managed Agents server does the actual coordination — the coordinator
spawns workers, fans briefs out, and waits on them itself. This module
owns everything around that: provisioning from config, session creation,
streaming with retry on transient stream failures, a live budget guard,
findings capture, and persistence so a run can be resumed by id instead
of restarted.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

from research_service.agents import ProvisionedTeam, provision_team
from research_service.briefs import plan_briefs, render_briefing
from research_service.config import PricingConfig, TeamConfig
from research_service.events import ProgressEvent, fold_event
from research_service.logging_utils import log_event
from research_service.metering import CostTracker
from research_service.store import Finding, RunState, RunStore, extract_urls

TERMINAL_STATUSES = ("finished", "aborted", "failed")


class BudgetExceededError(RuntimeError):
    """The run's cumulative cost crossed the configured budget."""


class ResearchOrchestrator:
    """Start, stream, resume, and meter research runs."""

    def __init__(
        self,
        client,
        team: TeamConfig,
        pricing: PricingConfig,
        store: RunStore,
        budget_usd: float | None = None,
        max_stream_retries: int = 3,
        budget_poll_events: int = 10,
        budget_poll_interval_s: float = 15.0,
        retry_backoff_s: float = 1.0,
    ):
        self.client = client
        self.team = team
        self.pricing = pricing
        self.store = store
        self.budget_usd = budget_usd
        self.max_stream_retries = max_stream_retries
        self.budget_poll_events = budget_poll_events
        self.budget_poll_interval_s = budget_poll_interval_s
        self.retry_backoff_s = retry_backoff_s
        self._team_ids: ProvisionedTeam | None = None
        self._cancel_requested: set[str] = set()

    # -- provisioning -------------------------------------------------

    def ensure_team(self) -> ProvisionedTeam:
        if self._team_ids is None:
            self._team_ids = provision_team(self.client, self.team)
        return self._team_ids

    def _resolve_worker_model(self, thread) -> str:
        """Resolve a child thread's model from the roster by agent name."""
        name = getattr(thread, "agent_name", None)
        if name:
            try:
                return self.team.worker(name).model
            except KeyError:
                pass
        return self.team.workers[0].model

    # -- lifecycle ----------------------------------------------------

    def start(self, question: str, facts: list[str] | None = None) -> RunState:
        """Provision, create environment + session, send the question."""
        team_ids = self.ensure_team()
        env = self.client.beta.environments.create(
            name="research-fanout",
            config={"type": "anthropic_cloud", "networking": {"type": "unrestricted"}},
        )
        session = self.client.beta.sessions.create(
            agent=team_ids.coordinator_id,
            environment_id=env.id,
            betas=list(self.team.betas),
        )

        message = render_briefing(question, plan_briefs(facts or [], self.team.briefs))

        state = self.store.new_run(question)
        state.session_id = session.id
        state.environment_id = env.id
        state.coordinator_id = team_ids.coordinator_id
        state.status = "running"
        self.store.save(state)

        self.client.beta.sessions.events.send(
            session.id,
            betas=list(self.team.betas),
            events=[{"type": "user.message", "content": [{"type": "text", "text": message}]}],
        )
        log_event("run_started", run_id=state.run_id, session_id=session.id)
        return state

    def stream(self, run_id: str) -> Iterator[ProgressEvent]:
        """Yield progress events for a run until it finishes or aborts.

        Safe to call on a freshly started run or on a resumed one — the
        run's state, session id, and captured findings all come from the
        store, so a crashed consumer just calls ``stream(run_id)`` again.
        Session events are cumulative on the server, which is what makes
        the resume cheap: no client-side offset bookkeeping is required.
        """
        state = self.store.load(run_id)
        if state.status in TERMINAL_STATUSES:
            return
        self._cancel_requested.discard(run_id)
        tracker = CostTracker(
            client=self.client,
            session_id=state.session_id,
            primary_model=self.team.coordinator.model,
            default_worker_model=self.team.workers[0].model,
            pricing=self.pricing,
            betas=self.team.betas,
            budget_usd=self.budget_usd,
            resolve_worker_model=self._resolve_worker_model,
        )

        attempts = 0
        events_since_poll = 0
        last_poll = time.monotonic()
        furthest = 0  # most folded events seen in any attempt so far
        while True:
            folded = 0
            try:
                with self.client.beta.sessions.events.stream(
                    state.session_id, betas=list(self.team.betas)
                ) as event_stream:
                    for raw in event_stream:
                        progress = fold_event(raw)
                        if progress is None:
                            continue
                        folded += 1
                        if folded > furthest:
                            # New progress since the last drop: earlier
                            # transient disconnects shouldn't accumulate
                            # against a long run.
                            furthest = folded
                            attempts = 0
                        if run_id in self._cancel_requested:
                            self._cancel_requested.discard(run_id)
                            yield ProgressEvent(kind="run_aborted", text="cancelled by user")
                            return
                        self._record(state, progress)
                        log_event("progress", run_id=run_id, event=progress.to_dict())
                        yield progress

                        if progress.kind == "run_finished":
                            self._finish(state, tracker)
                            return

                        events_since_poll += 1
                        now = time.monotonic()
                        if (
                            events_since_poll >= self.budget_poll_events
                            or now - last_poll >= self.budget_poll_interval_s
                        ):
                            events_since_poll = 0
                            last_poll = now
                            report = tracker.refresh()
                            yield ProgressEvent(
                                kind="budget_status",
                                text=f"${report.total_usd:.2f}"
                                + (f" / ${self.budget_usd:.2f}" if self.budget_usd else ""),
                            )
                            if tracker.over_budget:
                                self._abort(state, tracker)
                                yield ProgressEvent(
                                    kind="run_aborted",
                                    text=f"budget exceeded: ${report.total_usd:.2f}"
                                    f" > ${self.budget_usd:.2f}",
                                )
                                raise BudgetExceededError(
                                    f"run {run_id} cost ${report.total_usd:.2f}, "
                                    f"budget ${self.budget_usd:.2f}"
                                )
                # Stream closed without a run_finished event: reconnect.
                attempts += 1
            except BudgetExceededError:
                raise
            except Exception as exc:  # transient stream failure: reconnect
                attempts += 1
                log_event("stream_retry", run_id=run_id, attempt=attempts, error=str(exc))
                if attempts > self.max_stream_retries:
                    self._fail(state, str(exc))
                    raise
            if attempts > self.max_stream_retries:
                self._fail(state, "stream ended without run_finished")
                raise RuntimeError(f"run {run_id}: stream ended without finishing")
            time.sleep(self.retry_backoff_s * attempts)

    def run(self, question: str, facts: list[str] | None = None) -> RunState:
        """Start a run and drain it to completion; return the final state."""
        state = self.start(question, facts=facts)
        for _ in self.stream(state.run_id):
            pass
        return self.store.load(state.run_id)

    def cancel(self, run_id: str) -> RunState:
        """Abort a run at the user's request and tear its environment down."""
        state = self.store.load(run_id)
        if state.status in TERMINAL_STATUSES:
            return state
        self._cancel_requested.add(run_id)
        state.status = "aborted"
        state.error = "cancelled by user"
        self.store.save(state)
        self._teardown_environment(state)
        log_event("run_cancelled", run_id=run_id)
        return state

    # -- internals ----------------------------------------------------

    def _teardown_environment(self, state: RunState) -> None:
        """Archive the run's environment; best-effort, never fatal."""
        if not state.environment_id:
            return
        try:
            self.client.beta.environments.archive(state.environment_id)
            log_event(
                "environment_archived", run_id=state.run_id, environment_id=state.environment_id
            )
        except Exception as exc:
            log_event(
                "environment_archive_failed",
                run_id=state.run_id,
                environment_id=state.environment_id,
                error=str(exc),
            )

    def _record(self, state: RunState, progress: ProgressEvent) -> None:
        if progress.kind == "coordinator_message":
            state.final_answer = progress.text
        elif progress.kind == "findings_received":
            # Session events are cumulative, so a reconnect or resume
            # replays reports we already captured; don't record twice.
            if any(f.agent == progress.agent and f.text == progress.text for f in state.findings):
                return
            state.findings.append(
                Finding(agent=progress.agent, text=progress.text, urls=extract_urls(progress.text))
            )
        self.store.save(state)

    def _finish(self, state: RunState, tracker: CostTracker) -> None:
        state.status = "finished"
        state.cost_report = tracker.refresh().to_dict()
        self.store.save(state)
        self._teardown_environment(state)
        log_event("run_finished", run_id=state.run_id, total_usd=tracker.total_usd)

    def _abort(self, state: RunState, tracker: CostTracker) -> None:
        state.status = "aborted"
        state.error = f"budget exceeded: ${tracker.total_usd:.2f} > ${self.budget_usd:.2f}"
        state.cost_report = tracker.last_report.to_dict()
        self.store.save(state)
        self._teardown_environment(state)
        log_event("run_aborted", run_id=state.run_id, total_usd=tracker.total_usd)

    def _fail(self, state: RunState, error: str) -> None:
        state.status = "failed"
        state.error = error
        self.store.save(state)
        self._teardown_environment(state)
        log_event("run_failed", run_id=state.run_id, error=error)
