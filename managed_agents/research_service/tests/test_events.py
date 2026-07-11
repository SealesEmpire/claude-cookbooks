"""Event folding: raw session events -> typed progress events."""

from research_service.events import clip, fold_event, text_of

from tests.conftest import ev, text_block


def test_agent_message_becomes_coordinator_message():
    progress = fold_event(ev("agent.message", content=[text_block("  answer  ")]))
    assert progress.kind == "coordinator_message"
    assert progress.text == "answer"


def test_empty_agent_message_skipped():
    assert fold_event(ev("agent.message", content=[text_block("   ")])) is None


def test_thread_created_becomes_worker_spawned():
    progress = fold_event(
        ev("session.thread_created", agent_name="search-worker", session_thread_id="th_9")
    )
    assert progress.kind == "worker_spawned"
    assert progress.agent == "search-worker"
    assert progress.thread_id == "th_9"


def test_delegation_traffic():
    sent = fold_event(
        ev("agent.thread_message_sent", to_agent_name="w", content=[text_block("brief")])
    )
    assert (sent.kind, sent.agent, sent.text) == ("brief_sent", "w", "brief")
    received = fold_event(
        ev("agent.thread_message_received", from_agent_name="w", content=[text_block("report")])
    )
    assert (received.kind, received.agent, received.text) == ("findings_received", "w", "report")


def test_idle_becomes_run_finished():
    assert fold_event(ev("session.status_idle")).kind == "run_finished"


def test_unknown_event_skipped():
    assert fold_event(ev("span.model_request_end")) is None


def test_text_of_and_clip():
    content = [text_block("a"), ev("tool_use"), text_block("b")]
    assert text_of(content) == "ab"
    assert text_of(None) == ""
    assert clip("x" * 200, 10) == "x" * 10 + "..."
    assert clip("short") == "short"


def test_progress_event_serializes():
    d = fold_event(ev("session.status_idle")).to_dict()
    assert d["kind"] == "run_finished"
    assert "ts" in d
