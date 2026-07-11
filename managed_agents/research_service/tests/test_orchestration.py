"""Orchestration: provisioning, streaming, retry, budget guard, resume."""

import pytest
from research_service.agents import provision_team, toolset_config
from research_service.orchestration import BudgetExceededError, ResearchOrchestrator

from tests.conftest import successful_run_events, thread, usage


def make_orchestrator(fake_client, team_config, pricing_config, run_store, **kwargs):
    return ResearchOrchestrator(
        client=fake_client,
        team=team_config,
        pricing=pricing_config,
        store=run_store,
        **kwargs,
    )


def test_toolset_config_scopes_tools():
    cfg = toolset_config(("web_search", "web_fetch"))
    assert cfg[0]["default_config"] == {"enabled": False}
    assert [c["name"] for c in cfg[0]["configs"]] == ["web_search", "web_fetch"]
    assert toolset_config(()) == []


def test_provision_team_creates_workers_then_coordinator(fake_client, team_config):
    provisioned = provision_team(fake_client, team_config)
    assert set(provisioned.worker_ids) == {w.name for w in team_config.workers}
    coordinator_call = fake_client.created_agents[-1]
    assert coordinator_call["name"] == "research-coordinator"
    roster = coordinator_call["multiagent"]["agents"]
    assert {a["id"] for a in roster} == set(provisioned.worker_ids.values())
    assert "{max_parallel_workers}" not in coordinator_call["system"]


def test_happy_path_run(fake_client, team_config, pricing_config, run_store):
    fake_client.script_stream(successful_run_events())
    fake_client.threads = [
        thread("t_p", None, usage(input_tokens=50_000, output_tokens=5_000)),
        thread(
            "t_w",
            "t_p",
            usage(input_tokens=400_000, output_tokens=4_000),
            agent_name="search-worker",
        ),
    ]
    orchestrator = make_orchestrator(fake_client, team_config, pricing_config, run_store)
    state = orchestrator.run("Ten largest parks question")

    assert state.status == "finished"
    assert state.final_answer == "Final synthesized answer."
    assert len(state.findings) == 2  # premise-checker + search-worker reports
    assert state.findings[0].agent == "premise-checker"
    assert "https://www.nps.gov/aboutus/national-park-system.htm" in state.findings[0].urls
    assert state.cost_report["total_usd"] > 0
    # The question actually went to the session.
    sent = fake_client.sent_events[0]["events"][0]
    assert sent["content"][0]["text"].startswith("Ten largest parks question")


def test_facts_become_briefs_in_the_message(fake_client, team_config, pricing_config, run_store):
    fake_client.script_stream(successful_run_events())
    orchestrator = make_orchestrator(fake_client, team_config, pricing_config, run_store)
    orchestrator.start("Q", facts=[f"fact {i}" for i in range(6)])
    text = fake_client.sent_events[0]["events"][0]["content"][0]["text"]
    assert "Suggested delegation split" in text
    assert "Brief 1:" in text and "Brief 2:" in text  # 6 facts, batch size 4


def test_budget_abort(fake_client, team_config, pricing_config, run_store):
    fake_client.script_stream(successful_run_events())
    fake_client.threads = [
        thread("t_p", None, usage(input_tokens=2_000_000, output_tokens=200_000)),
    ]
    orchestrator = make_orchestrator(
        fake_client,
        team_config,
        pricing_config,
        run_store,
        budget_usd=1.0,
        budget_poll_events=1,
    )
    state = orchestrator.start("Q")
    kinds = []
    with pytest.raises(BudgetExceededError):
        for progress in orchestrator.stream(state.run_id):
            kinds.append(progress.kind)
    assert "run_aborted" in kinds
    final = run_store.load(state.run_id)
    assert final.status == "aborted"
    assert "budget exceeded" in final.error


def test_stream_retry_recovers(fake_client, team_config, pricing_config, run_store):
    events = successful_run_events()
    fake_client.script_stream(events, fail_after=2)  # first attempt drops mid-stream
    fake_client.script_stream(events, reset=False)  # reconnect succeeds
    orchestrator = make_orchestrator(
        fake_client, team_config, pricing_config, run_store, retry_backoff_s=0.0
    )
    state = orchestrator.run("Q")
    assert state.status == "finished"
    assert state.final_answer == "Final synthesized answer."


def test_stream_exhausts_retries(fake_client, team_config, pricing_config, run_store):
    fake_client.script_stream(successful_run_events(), fail_after=1)
    orchestrator = make_orchestrator(
        fake_client,
        team_config,
        pricing_config,
        run_store,
        max_stream_retries=2,
        retry_backoff_s=0.0,
    )
    state = orchestrator.start("Q")
    with pytest.raises(ConnectionError):
        list(orchestrator.stream(state.run_id))
    assert run_store.load(state.run_id).status == "failed"


def test_resume_by_run_id(fake_client, team_config, pricing_config, run_store):
    fake_client.script_stream(successful_run_events())
    orchestrator = make_orchestrator(fake_client, team_config, pricing_config, run_store)
    state = orchestrator.start("Q")

    # Simulate a crashed consumer: a *new* orchestrator resumes by id
    # using only what the store persisted.
    resumed = make_orchestrator(fake_client, team_config, pricing_config, run_store)
    kinds = [p.kind for p in resumed.stream(state.run_id)]
    assert kinds[-1] == "run_finished"
    assert run_store.load(state.run_id).status == "finished"


def test_worker_model_resolution(fake_client, team_config, pricing_config, run_store):
    orchestrator = make_orchestrator(fake_client, team_config, pricing_config, run_store)
    haiku_thread = thread("t", "p", usage(), agent_name="fetch-extract-worker")
    unknown_thread = thread("t", "p", usage(), agent_name="mystery")
    assert orchestrator._resolve_worker_model(haiku_thread) == "claude-haiku-4-5"
    assert orchestrator._resolve_worker_model(unknown_thread) == team_config.workers[0].model


def test_retry_counter_resets_after_progress(fake_client, team_config, pricing_config, run_store):
    """Transient drops spread across a long run don't accumulate to failure."""
    events = successful_run_events()
    # Three drops, each one further along than the last (new progress
    # between failures), with max_stream_retries=2: the old always-
    # incrementing counter would fail, the resetting one recovers.
    fake_client.script_stream(events, fail_after=2)
    fake_client.script_stream(events, fail_after=4, reset=False)
    fake_client.script_stream(events, fail_after=6, reset=False)
    fake_client.script_stream(events, reset=False)
    orchestrator = make_orchestrator(
        fake_client,
        team_config,
        pricing_config,
        run_store,
        max_stream_retries=2,
        retry_backoff_s=0.0,
    )
    state = orchestrator.run("Q")
    assert state.status == "finished"


def test_reconnect_does_not_duplicate_findings(fake_client, team_config, pricing_config, run_store):
    """Cumulative event replay after a drop records each finding once."""
    events = successful_run_events()
    fake_client.script_stream(events, fail_after=4)  # drops after first finding
    fake_client.script_stream(events, reset=False)
    orchestrator = make_orchestrator(
        fake_client, team_config, pricing_config, run_store, retry_backoff_s=0.0
    )
    state = orchestrator.run("Q")
    assert state.status == "finished"
    assert len(state.findings) == 2


def test_environment_archived_on_finish(fake_client, team_config, pricing_config, run_store):
    fake_client.script_stream(successful_run_events())
    orchestrator = make_orchestrator(fake_client, team_config, pricing_config, run_store)
    state = orchestrator.run("Q")
    assert state.environment_id in fake_client.archived_environments


def test_environment_archived_on_failure(fake_client, team_config, pricing_config, run_store):
    fake_client.script_stream(successful_run_events(), fail_after=1)
    orchestrator = make_orchestrator(
        fake_client,
        team_config,
        pricing_config,
        run_store,
        max_stream_retries=0,
        retry_backoff_s=0.0,
    )
    state = orchestrator.start("Q")
    with pytest.raises(ConnectionError):
        list(orchestrator.stream(state.run_id))
    assert run_store.load(state.run_id).environment_id in fake_client.archived_environments


def test_cancel_marks_aborted_and_tears_down(fake_client, team_config, pricing_config, run_store):
    fake_client.script_stream(successful_run_events())
    orchestrator = make_orchestrator(fake_client, team_config, pricing_config, run_store)
    state = orchestrator.start("Q")

    cancelled = orchestrator.cancel(state.run_id)
    assert cancelled.status == "aborted"
    assert cancelled.error == "cancelled by user"
    assert cancelled.environment_id in fake_client.archived_environments
    # A cancelled run streams nothing.
    assert list(orchestrator.stream(state.run_id)) == []
    # Cancelling again is a no-op.
    assert orchestrator.cancel(state.run_id).status == "aborted"


def test_cancel_stops_an_active_stream(fake_client, team_config, pricing_config, run_store):
    fake_client.script_stream(successful_run_events())
    orchestrator = make_orchestrator(fake_client, team_config, pricing_config, run_store)
    state = orchestrator.start("Q")

    kinds = []
    for progress in orchestrator.stream(state.run_id):
        kinds.append(progress.kind)
        if len(kinds) == 2:
            orchestrator.cancel(state.run_id)
    assert kinds[-1] == "run_aborted"
    assert run_store.load(state.run_id).status == "aborted"


def test_time_based_budget_poll(fake_client, team_config, pricing_config, run_store):
    """The budget guard also fires on wall-clock time, not just event count."""
    fake_client.script_stream(successful_run_events())
    fake_client.threads = [
        thread("t_p", None, usage(input_tokens=10_000, output_tokens=1_000)),
    ]
    orchestrator = make_orchestrator(
        fake_client,
        team_config,
        pricing_config,
        run_store,
        budget_poll_events=10_000,  # event-count polling effectively off
        budget_poll_interval_s=0.0,
    )
    state = orchestrator.start("Q")
    kinds = [p.kind for p in orchestrator.stream(state.run_id)]
    assert "budget_status" in kinds
