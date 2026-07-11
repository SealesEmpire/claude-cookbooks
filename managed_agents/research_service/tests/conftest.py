"""A fake Managed Agents client for testing the orchestration layer.

The fake mirrors exactly the slice of the SDK surface the service uses:
``beta.agents.create``, ``beta.environments.create``,
``beta.sessions.create``, ``beta.sessions.events.send``,
``beta.sessions.events.stream`` (a context manager yielding scripted
events), and ``beta.sessions.threads.list``.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
from research_service.config import load_pricing, load_team
from research_service.store import RunStore


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def ev(type_: str, **attrs) -> SimpleNamespace:
    return SimpleNamespace(type=type_, **attrs)


def usage(
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read: int = 0,
    cache_5m: int = 0,
    cache_1h: int = 0,
):
    cache = None
    if cache_5m or cache_1h:
        cache = SimpleNamespace(
            ephemeral_5m_input_tokens=cache_5m, ephemeral_1h_input_tokens=cache_1h
        )
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation=cache,
    )


def thread(id_: str, parent: str | None, u, agent_name: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(id=id_, parent_thread_id=parent, usage=u, agent_name=agent_name)


class _FakeStream:
    def __init__(self, events, fail_after: int | None = None):
        self._events = events
        self._fail_after = fail_after

    def __enter__(self):
        def gen():
            for i, event in enumerate(self._events):
                if self._fail_after is not None and i >= self._fail_after:
                    raise ConnectionError("stream dropped")
                yield event

        return gen()

    def __exit__(self, *exc):
        return False


class FakeClient:
    """Scriptable fake of the Managed Agents API client."""

    def __init__(self):
        self._ids = itertools.count(1)
        self.created_agents: list[dict] = []
        self.sent_events: list[dict] = []
        # Each entry is (events, fail_after); streams are consumed in order,
        # and the last entry repeats, so a retry test scripts two entries.
        self.stream_scripts: list[tuple[list, int | None]] = [([], None)]
        self._streams_served = 0
        self.threads: list = []

        agents = SimpleNamespace(create=self._agents_create)
        environments = SimpleNamespace(create=self._environments_create)
        events = SimpleNamespace(send=self._events_send, stream=self._events_stream)
        threads = SimpleNamespace(list=self._threads_list)
        sessions = SimpleNamespace(create=self._sessions_create, events=events, threads=threads)
        self.beta = SimpleNamespace(agents=agents, environments=environments, sessions=sessions)

    # -- scripting ------------------------------------------------------

    def script_stream(self, events, fail_after: int | None = None, reset: bool = True):
        if reset:
            self.stream_scripts = []
        self.stream_scripts.append((list(events), fail_after))

    # -- fake API surface -------------------------------------------------

    def _agents_create(self, **kwargs):
        self.created_agents.append(kwargs)
        return SimpleNamespace(id=f"agent_{next(self._ids)}", **kwargs)

    def _environments_create(self, **kwargs):
        return SimpleNamespace(id=f"env_{next(self._ids)}", **kwargs)

    def _sessions_create(self, **kwargs):
        return SimpleNamespace(id=f"session_{next(self._ids)}", **kwargs)

    def _events_send(self, session_id, **kwargs):
        self.sent_events.append({"session_id": session_id, **kwargs})

    def _events_stream(self, session_id, **kwargs):
        idx = min(self._streams_served, len(self.stream_scripts) - 1)
        self._streams_served += 1
        events, fail_after = self.stream_scripts[idx]
        return _FakeStream(events, fail_after)

    def _threads_list(self, session_id, **kwargs):
        return list(self.threads)


@pytest.fixture
def fake_client():
    return FakeClient()


@pytest.fixture
def team_config():
    return load_team()


@pytest.fixture
def pricing_config():
    return load_pricing()


@pytest.fixture
def run_store(tmp_path):
    return RunStore(tmp_path / "runs")


def happy_path_events(answer: str = "Final synthesized answer.") -> list:
    """The event script of one successful team run."""
    return [
        ev("session.thread_created", agent_name="premise-checker", session_thread_id="th_1"),
        ev(
            "agent.thread_message_sent",
            to_agent_name="premise-checker",
            content=[text_block("Confirm the list of the ten largest parks.")],
        ),
        ev(
            "agent.thread_message_received",
            from_agent_name="premise-checker",
            content=[
                text_block(
                    "List confirmed except #10: Great Smoky Mountains, not Kings Canyon. "
                    "Source: https://www.nps.gov/aboutus/national-park-system.htm"
                )
            ],
        ),
        ev("session.thread_created", agent_name="search-worker", session_thread_id="th_2"),
        ev(
            "agent.thread_message_sent",
            to_agent_name="search-worker",
            content=[text_block("Verify fees for Death Valley and Yellowstone.")],
        ),
        ev(
            "agent.thread_message_received",
            from_agent_name="search-worker",
            content=[
                text_block(
                    "Death Valley $30 (https://www.nps.gov/deva/planyourvisit/fees.htm), "
                    "Yellowstone $35 (https://www.nps.gov/yell/planyourvisit/fees.htm)."
                )
            ],
        ),
        ev("agent.message", content=[text_block(answer)]),
        ev("session.status_idle"),
    ]
