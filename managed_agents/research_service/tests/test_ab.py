"""A/B mode: solo vs team on the same question."""

from research_service.ab import ABResult, ArmResult, run_ab
from research_service.metering import SessionCostReport, ThreadCost

from tests.conftest import happy_path_events, thread, usage


def _report(total: float) -> SessionCostReport:
    return SessionCostReport(
        threads=(
            ThreadCost(
                thread_id="t",
                is_primary=True,
                model="m",
                input_tokens=1,
                output_tokens=1,
                cost_usd=total,
            ),
        )
    )


def test_ratios():
    result = ABResult(
        team=ArmResult(answer="a", duration_s=100.0, cost=_report(4.0)),
        solo=ArmResult(answer="b", duration_s=300.0, cost=_report(10.0)),
    )
    assert result.cost_ratio == 2.5
    assert result.speed_ratio == 3.0


def test_run_ab_end_to_end(fake_client, team_config, pricing_config, run_store):
    from tests.test_orchestration import make_orchestrator

    # Both arms consume the same scripted stream; the solo arm's events
    # contain no delegation traffic in a real run, but the folding is
    # identical, so one script exercises both paths.
    fake_client.script_stream(happy_path_events())
    fake_client.threads = [
        thread("t_p", None, usage(input_tokens=10_000, output_tokens=1_000)),
        thread(
            "t_w",
            "t_p",
            usage(input_tokens=100_000, output_tokens=1_000),
            agent_name="search-worker",
        ),
    ]
    orchestrator = make_orchestrator(fake_client, team_config, pricing_config, run_store)
    result = run_ab(orchestrator, "Q")

    assert result.team.answer == "Final synthesized answer."
    assert result.solo.answer == "Final synthesized answer."
    assert result.team.cost.total_usd > 0
    assert result.solo.cost.total_usd > 0
    # Same token volumes, but the solo arm bills every thread at the
    # frontier rate, so it must come out more expensive.
    assert result.cost_ratio > 1.0
    # The solo agent was created with the workers' tools.
    solo_call = fake_client.created_agents[-1]
    assert solo_call["name"] == "solo-researcher"
    tool_names = [c["name"] for c in solo_call["tools"][0]["configs"]]
    assert tool_names == ["web_search", "web_fetch"]
