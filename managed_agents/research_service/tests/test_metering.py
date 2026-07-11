"""Metering math and cost attribution, checked against the notebook's formulas."""

from research_service.metering import CostTracker, cost_usd, session_report, total_input

from tests.conftest import thread, usage


def test_total_input_includes_cache_traffic():
    u = usage(input_tokens=1000, cache_read=500, cache_5m=200, cache_1h=100)
    assert total_input(u) == 1800


def test_total_input_without_cache_activity():
    assert total_input(usage(input_tokens=42)) == 42


def test_cost_matches_notebook_formula(pricing_config):
    # sonnet-5: $2 in / $10 out per MTok
    u = usage(input_tokens=1_000_000, output_tokens=100_000, cache_read=1_000_000, cache_5m=100_000)
    expected = (
        1_000_000 * 2.0  # plain input
        + 100_000 * 2.0 * 1.25  # 5m cache writes at 1.25x
        + 1_000_000 * 2.0 * 0.1  # cache reads at 0.1x
        + 100_000 * 10.0  # output
    ) / 1e6
    assert cost_usd(u, "claude-sonnet-5", pricing_config) == expected


def test_session_report_attribution(fake_client, pricing_config):
    fake_client.threads = [
        thread("t_primary", None, usage(input_tokens=10_000, output_tokens=2_000)),
        thread("t_w1", "t_primary", usage(input_tokens=100_000, output_tokens=1_000)),
        thread("t_w2", "t_primary", usage(input_tokens=90_000, output_tokens=1_000)),
    ]
    report = session_report(
        fake_client,
        "session_1",
        primary_model="claude-fable-5",
        default_worker_model="claude-sonnet-5",
        pricing=pricing_config,
    )
    assert report.primary.thread_id == "t_primary"
    assert len(report.workers) == 2
    assert report.worker_input_share == 190_000 / 200_000
    assert report.total_usd > 0
    # frontier tokens bill higher than the same volume at worker rates
    assert report.primary.model == "claude-fable-5"
    assert all(w.model == "claude-sonnet-5" for w in report.workers)


def test_session_report_heterogeneous_worker_models(fake_client, pricing_config, team_config):
    fake_client.threads = [
        thread("t_p", None, usage(input_tokens=1_000)),
        thread("t_h", "t_p", usage(input_tokens=1_000), agent_name="fetch-extract-worker"),
        thread("t_s", "t_p", usage(input_tokens=1_000), agent_name="search-worker"),
    ]

    def worker_model_for(t):
        return team_config.worker(t.agent_name).model

    report = session_report(
        fake_client,
        "session_1",
        primary_model=team_config.coordinator.model,
        worker_model_for=worker_model_for,
        default_worker_model="claude-sonnet-5",
        pricing=pricing_config,
    )
    by_id = {t.thread_id: t.model for t in report.threads}
    assert by_id["t_h"] == "claude-haiku-4-5"
    assert by_id["t_s"] == "claude-sonnet-5"


def test_cost_tracker_budget_guard(fake_client, pricing_config):
    fake_client.threads = [
        thread("t_p", None, usage(input_tokens=1_000_000, output_tokens=100_000)),
    ]
    tracker = CostTracker(
        client=fake_client,
        session_id="session_1",
        primary_model="claude-fable-5",
        default_worker_model="claude-sonnet-5",
        pricing=pricing_config,
        budget_usd=1.0,
    )
    assert not tracker.over_budget  # nothing refreshed yet
    tracker.refresh()
    # 1M input at $10 + 100k output at $50 = $15 > $1
    assert tracker.total_usd == 15.0
    assert tracker.over_budget


def test_cost_tracker_no_budget_never_aborts(fake_client, pricing_config):
    fake_client.threads = [thread("t_p", None, usage(input_tokens=10_000_000))]
    tracker = CostTracker(
        client=fake_client,
        session_id="session_1",
        primary_model="claude-fable-5",
        default_worker_model="claude-sonnet-5",
        pricing=pricing_config,
        budget_usd=None,
    )
    tracker.refresh()
    assert not tracker.over_budget
