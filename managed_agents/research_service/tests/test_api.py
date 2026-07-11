"""API layer: run lifecycle over HTTP and the SSE progress stream."""

import json

from fastapi.testclient import TestClient
from research_service.api import create_app

from tests.conftest import successful_run_events, thread, usage
from tests.test_orchestration import make_orchestrator


def make_client(fake_client, team_config, pricing_config, run_store, **kwargs):
    fake_client.script_stream(successful_run_events())
    fake_client.threads = [
        thread("t_p", None, usage(input_tokens=10_000, output_tokens=1_000)),
    ]
    orchestrator = make_orchestrator(fake_client, team_config, pricing_config, run_store, **kwargs)
    return TestClient(create_app(orchestrator))


def sse_payloads(response) -> list[dict]:
    return [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def test_start_and_get_run(fake_client, team_config, pricing_config, run_store):
    client = make_client(fake_client, team_config, pricing_config, run_store)
    created = client.post("/runs", json={"question": "Q"}).json()
    assert created["run_id"]

    run = client.get(f"/runs/{created['run_id']}").json()
    assert run["question"] == "Q"
    assert run["status"] == "running"

    runs = client.get("/runs").json()["runs"]
    summary = next(r for r in runs if r["run_id"] == created["run_id"])
    assert summary["status"] == "running"
    assert summary["question"] == "Q"
    assert summary["created_at"] > 0


def test_unknown_run_404(fake_client, team_config, pricing_config, run_store):
    client = make_client(fake_client, team_config, pricing_config, run_store)
    assert client.get("/runs/doesnotexist").status_code == 404
    assert client.get("/runs/../sneaky").status_code == 404


def test_stream_relays_progress_events(fake_client, team_config, pricing_config, run_store):
    client = make_client(fake_client, team_config, pricing_config, run_store)
    run_id = client.post("/runs", json={"question": "Q"}).json()["run_id"]

    payloads = sse_payloads(client.get(f"/runs/{run_id}/stream"))
    kinds = [p["kind"] for p in payloads]
    assert "worker_spawned" in kinds
    assert "findings_received" in kinds
    assert "coordinator_message" in kinds
    assert kinds[-1] == "run_state"
    assert payloads[-1]["state"]["status"] == "finished"


def test_stream_of_finished_run_replays_state(fake_client, team_config, pricing_config, run_store):
    client = make_client(fake_client, team_config, pricing_config, run_store)
    run_id = client.post("/runs", json={"question": "Q"}).json()["run_id"]
    client.get(f"/runs/{run_id}/stream")  # drain to completion

    payloads = sse_payloads(client.get(f"/runs/{run_id}/stream"))
    assert [p["kind"] for p in payloads] == ["run_state"]
    assert payloads[0]["state"]["final_answer"] == "Final synthesized answer."


def test_index_serves_frontend(fake_client, team_config, pricing_config, run_store):
    client = make_client(fake_client, team_config, pricing_config, run_store)
    response = client.get("/")
    assert response.status_code == 200
    assert "delegation graph" in response.text


def test_health(fake_client, team_config, pricing_config, run_store):
    client = make_client(fake_client, team_config, pricing_config, run_store)
    assert client.get("/health").json() == {"status": "ok"}


def test_run_request_validation(fake_client, team_config, pricing_config, run_store):
    client = make_client(fake_client, team_config, pricing_config, run_store)
    assert client.post("/runs", json={"question": ""}).status_code == 422
    assert client.post("/runs", json={"question": "x" * 5000}).status_code == 422
    assert client.post("/runs", json={"question": "Q", "facts": ["y" * 3000]}).status_code == 422


def test_cancel_endpoint(fake_client, team_config, pricing_config, run_store):
    client = make_client(fake_client, team_config, pricing_config, run_store)
    run_id = client.post("/runs", json={"question": "Q"}).json()["run_id"]

    cancelled = client.post(f"/runs/{run_id}/cancel").json()
    assert cancelled["status"] == "aborted"
    assert client.get(f"/runs/{run_id}").json()["error"] == "cancelled by user"
    assert client.post("/runs/doesnotexist/cancel").status_code == 404

    # A cancelled run's stream replays the terminal state.
    payloads = sse_payloads(client.get(f"/runs/{run_id}/stream"))
    assert [p["kind"] for p in payloads] == ["run_state"]
    assert payloads[0]["state"]["status"] == "aborted"


def test_concurrent_streamers_share_one_driver(fake_client, team_config, pricing_config, run_store):
    """Two browsers on the same run don't double-record findings."""
    import threading

    client = make_client(fake_client, team_config, pricing_config, run_store)
    run_id = client.post("/runs", json={"question": "Q"}).json()["run_id"]

    results: list = [None, None]

    def consume(i: int) -> None:
        results[i] = sse_payloads(client.get(f"/runs/{run_id}/stream"))

    threads = [threading.Thread(target=consume, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    final = run_store.load(run_id)
    assert final.status == "finished"
    assert len(final.findings) == 2  # not doubled by the second consumer
    for payloads in results:
        assert payloads is not None
        assert payloads[-1]["kind"] == "run_state"
        assert payloads[-1]["state"]["status"] == "finished"
