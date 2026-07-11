"""Persistence: run state and distilled findings with source URLs.

A run's session id, status, findings, final answer, and cost report are
written to disk as JSON so long-running jobs can be resumed instead of
restarted, and so every claim in the final answer stays auditable back
to the worker report and source URLs that produced it.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

_URL_RE = re.compile(r"https?://[^\s\)\]>\"']+")


def extract_urls(text: str) -> list[str]:
    """Pull the source URLs out of a worker's distilled findings."""
    seen: dict[str, None] = {}
    for url in _URL_RE.findall(text or ""):
        seen.setdefault(url.rstrip(".,;"), None)
    return list(seen)


@dataclass
class Finding:
    """One worker report, distilled, with its source URLs."""

    agent: str
    text: str
    urls: list[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)


@dataclass
class RunState:
    run_id: str
    question: str
    status: str = "created"  # created | running | finished | aborted | failed
    session_id: str = ""
    environment_id: str = ""
    coordinator_id: str = ""
    final_answer: str = ""
    findings: list[Finding] = field(default_factory=list)
    cost_report: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class RunStore:
    """JSON-file-per-run persistence under a base directory."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError(f"invalid run id {run_id!r}")
        return self.base_dir / f"{run_id}.json"

    def new_run(self, question: str) -> RunState:
        state = RunState(run_id=uuid.uuid4().hex, question=question)
        self.save(state)
        return state

    def save(self, state: RunState) -> None:
        self._path(state.run_id).write_text(json.dumps(state.to_dict(), indent=2))

    def load(self, run_id: str) -> RunState:
        raw = json.loads(self._path(run_id).read_text())
        raw["findings"] = [Finding(**f) for f in raw.get("findings", [])]
        return RunState(**raw)

    def list_runs(self) -> list[str]:
        return sorted(p.stem for p in self.base_dir.glob("*.json"))
