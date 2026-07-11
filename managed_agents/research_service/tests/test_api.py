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

    assert created["run_id"] in client.get("/runs").json()["runs"]


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
