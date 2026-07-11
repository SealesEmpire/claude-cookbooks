"""FastAPI layer: start research runs and stream progress to a browser.

POST /runs starts a run; GET /runs/{id}/stream drives the orchestrator
and relays each ProgressEvent as a server-sent event, mirroring the
notebook's streaming loop. The frontend in ../frontend renders the
delegation graph, running cost, and findings from that stream.

One driver per run: the first stream subscriber starts a pump thread
that drives ``orchestrator.stream()``; additional subscribers (a second
browser tab, a reconnect) attach to the same pump and receive the same
events, so findings are never recorded twice and run-state writes never
race. Each subscriber gets a bounded queue, so a browser that goes away
without closing the socket can't grow memory without limit.

This service is intended to run locally: it has no authentication and
should not be exposed to untrusted networks as-is.

Run it:
    uvicorn research_service.api:app --reload
"""

from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path
from typing import Annotated

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, StringConstraints

from research_service.config import load_pricing, load_team
from research_service.logging_utils import configure_logging
from research_service.orchestration import TERMINAL_STATUSES, ResearchOrchestrator
from research_service.store import RunStore

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
SUBSCRIBER_QUEUE_MAX = 512


class RunRequest(BaseModel):
    question: Annotated[str, StringConstraints(min_length=1, max_length=4_000)]
    facts: list[Annotated[str, StringConstraints(max_length=2_000)]] | None = Field(
        default=None, max_length=200
    )


class RunBroadcaster:
    """Fan one orchestrator.stream() per run out to N SSE subscribers."""

    def __init__(self):
        self._lock = threading.Lock()
        self._subscribers: dict[str, list[queue.Queue]] = {}

    def subscribe(self, orchestrator: ResearchOrchestrator, run_id: str) -> queue.Queue:
        """Attach a bounded queue; the first subscriber starts the pump."""
        q: queue.Queue = queue.Queue(maxsize=SUBSCRIBER_QUEUE_MAX)
        with self._lock:
            first = run_id not in self._subscribers
            self._subscribers.setdefault(run_id, []).append(q)
        if first:
            threading.Thread(target=self._pump, args=(orchestrator, run_id), daemon=True).start()
        return q

    def unsubscribe(self, run_id: str, q: queue.Queue) -> None:
        with self._lock:
            subs = self._subscribers.get(run_id, [])
            if q in subs:
                subs.remove(q)

    def _pump(self, orchestrator: ResearchOrchestrator, run_id: str) -> None:
        try:
            for event in orchestrator.stream(run_id):
                self._publish(run_id, event.to_dict())
        except Exception as exc:
            self._publish(run_id, {"kind": "error", "text": str(exc)})
        finally:
            with self._lock:
                subs = self._subscribers.pop(run_id, [])
            for q in subs:
                self._put(q, None)

    def _publish(self, run_id: str, item: dict) -> None:
        with self._lock:
            subs = list(self._subscribers.get(run_id, []))
        for q in subs:
            self._put(q, item)

    @staticmethod
    def _put(q: queue.Queue, item) -> None:
        """Enqueue without blocking; drop the oldest event if full."""
        while True:
            try:
                q.put_nowait(item)
                return
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass


def default_orchestrator() -> ResearchOrchestrator:
    load_dotenv()
    configure_logging()
    budget = os.environ.get("RESEARCH_BUDGET_USD")
    return ResearchOrchestrator(
        client=anthropic.Anthropic(),
        team=load_team(),
        pricing=load_pricing(),
        store=RunStore(os.environ.get("RESEARCH_RUN_DIR", ".runs")),
        budget_usd=float(budget) if budget else None,
    )


def create_app(orchestrator: ResearchOrchestrator | None = None) -> FastAPI:
    app = FastAPI(title="research-service")
    app.state.orchestrator = orchestrator
    broadcaster = RunBroadcaster()

    def orch() -> ResearchOrchestrator:
        if app.state.orchestrator is None:
            app.state.orchestrator = default_orchestrator()
        return app.state.orchestrator

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/runs")
    def start_run(req: RunRequest) -> dict:
        state = orch().start(req.question, facts=req.facts)
        return {"run_id": state.run_id, "session_id": state.session_id}

    @app.get("/runs")
    def list_runs() -> dict:
        return {"runs": orch().store.summaries()}

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        try:
            return orch().store.load(run_id).to_dict()
        except (FileNotFoundError, ValueError):
            raise HTTPException(status_code=404, detail="run not found") from None

    @app.post("/runs/{run_id}/cancel")
    def cancel_run(run_id: str) -> dict:
        try:
            state = orch().cancel(run_id)
        except (FileNotFoundError, ValueError):
            raise HTTPException(status_code=404, detail="run not found") from None
        return {"run_id": state.run_id, "status": state.status}

    @app.get("/runs/{run_id}/stream")
    def stream_run(run_id: str) -> StreamingResponse:
        o = orch()
        try:
            state = o.store.load(run_id)
        except (FileNotFoundError, ValueError):
            raise HTTPException(status_code=404, detail="run not found") from None

        if state.status in TERMINAL_STATUSES:
            # Replay a terminal run's state instead of re-streaming.
            def replay():
                yield _sse({"kind": "run_state", "state": state.to_dict()})

            return StreamingResponse(replay(), media_type="text/event-stream")

        q = broadcaster.subscribe(o, run_id)

        def relay():
            try:
                while True:
                    item = q.get()
                    if item is None:
                        final = o.store.load(run_id)
                        yield _sse({"kind": "run_state", "state": final.to_dict()})
                        return
                    yield _sse(item)
            finally:
                # Runs on normal completion and when the client goes away
                # (the generator is closed), so dead subscribers don't
                # keep accumulating events.
                broadcaster.unsubscribe(run_id, q)

        return StreamingResponse(relay(), media_type="text/event-stream")

    return app


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


app = create_app()
