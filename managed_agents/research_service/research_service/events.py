"""Fold raw Managed Agents session events into typed progress events.

The notebook's inline ``match ev.type`` loop becomes one pure function,
``fold_event``, so the orchestrator, the API's SSE stream, and the tests
all share the same event vocabulary.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field


def text_of(content) -> str:
    """Concatenate the text blocks of an event's content list."""
    return "".join(b.text for b in content or [] if getattr(b, "type", None) == "text")


def clip(s: str, n: int = 160) -> str:
    return s[:n] + ("..." if len(s) > n else "")


@dataclass(frozen=True)
class ProgressEvent:
    """One step of a research run, in the service's own vocabulary.

    kind is one of: coordinator_message, worker_spawned, brief_sent,
    findings_received, run_finished, budget_status, run_aborted.
    """

    kind: str
    text: str = ""
    agent: str = ""
    thread_id: str = ""
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


def fold_event(ev) -> ProgressEvent | None:
    """Map one raw session event to a ProgressEvent, or None to skip it."""
    match ev.type:
        case "agent.message":
            if text := text_of(ev.content).strip():
                return ProgressEvent(kind="coordinator_message", text=text)
            return None
        case "session.thread_created":
            return ProgressEvent(
                kind="worker_spawned",
                agent=getattr(ev, "agent_name", ""),
                thread_id=getattr(ev, "session_thread_id", ""),
            )
        case "agent.thread_message_sent":
            return ProgressEvent(
                kind="brief_sent",
                agent=getattr(ev, "to_agent_name", ""),
                text=text_of(ev.content),
            )
        case "agent.thread_message_received":
            return ProgressEvent(
                kind="findings_received",
                agent=getattr(ev, "from_agent_name", ""),
                text=text_of(ev.content),
            )
        case "session.status_idle":
            return ProgressEvent(kind="run_finished")
        case _:
            return None
