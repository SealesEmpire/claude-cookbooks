"""FastAPI layer: start research runs and stream progress to a browser.

POST /runs starts a run; GET /runs/{id}/stream drives the orchestrator
and relays each ProgressEvent as a server-sent event, mirroring the
notebook's streaming loop. The frontend in ../frontend renders the
delegation graph, running cost, and findings from that stream.

Run it:
    uvicorn research_service.api:app --reload
"""

from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from research_service.config import load_pricing, load_team
from research_service.logging_utils import configure_logging
from research_service.orchestration import ResearchOrchestrator
from research_service.store import RunStore

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


class RunRequest(BaseModel):
    question: str
    facts: list[str] | None = None


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

    def orch() -> ResearchOrchestrator:
        if app.state.orchestrator is None:
            app.state.orchestrator = default_orchestrator()
        return app.state.orchestrator

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.post("/runs")
    def start_run(req: RunRequest) -> dict:
        state = orch().start(req.question, facts=req.facts)
        return {"run_id": state.run_id, "session_id": state.session_id}

    @app.get("/runs")
    def list_runs() -> dict:
        return {"runs": orch().store.list_runs()}

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        try:
            return orch().store.load(run_id).to_dict()
        except (FileNotFoundError, ValueError):
            raise HTTPException(status_code=404, detail="run not found") from None

    @app.get("/runs/{run_id}/stream")
    def stream_run(run_id: str) -> StreamingResponse:
        o = orch()
        try:
            state = o.store.load(run_id)
        except (FileNotFoundError, ValueError):
            raise HTTPException(status_code=404, detail="run not found") from None

        if state.status in ("finished", "aborted", "failed"):
            # Replay a terminal run's state instead of re-streaming.
            def replay():
                yield _sse({"kind": "run_state", "state": state.to_dict()})

            return StreamingResponse(replay(), media_type="text/event-stream")

        # The SSE handler drives orchestrator.stream() in a worker
        # thread and relays each event; a dropped browser just
        # reconnects to the same URL and streaming resumes by run id.
        q: queue.Queue = queue.Queue()

        def pump() -> None:
            try:
                for event in o.stream(run_id):
                    q.put(event.to_dict())
            except Exception as exc:
                q.put({"kind": "error", "text": str(exc)})
            finally:
                q.put(None)

        threading.Thread(target=pump, daemon=True).start()

        def relay():
            while True:
                item = q.get()
                if item is None:
                    final = o.store.load(run_id)
                    yield _sse({"kind": "run_state", "state": final.to_dict()})
                    return
                yield _sse(item)

        return StreamingResponse(relay(), media_type="text/event-stream")

    return app


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


app = create_app()
