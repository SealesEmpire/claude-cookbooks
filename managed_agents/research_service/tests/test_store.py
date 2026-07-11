"""Run persistence and findings capture."""

import pytest
from research_service.store import Finding, RunStore, extract_urls


def test_extract_urls_dedupes_and_strips_punctuation():
    text = (
        "See https://www.nps.gov/deva/planyourvisit/fees.htm, and "
        "(https://www.nps.gov/yell/planyourvisit/fees.htm). Also "
        "https://www.nps.gov/deva/planyourvisit/fees.htm again."
    )
    assert extract_urls(text) == [
        "https://www.nps.gov/deva/planyourvisit/fees.htm",
        "https://www.nps.gov/yell/planyourvisit/fees.htm",
    ]


def test_extract_urls_empty():
    assert extract_urls("") == []
    assert extract_urls("no links here") == []


def test_run_roundtrip(tmp_path):
    store = RunStore(tmp_path)
    state = store.new_run("What color is the sky?")
    state.status = "running"
    state.session_id = "session_1"
    state.findings.append(
        Finding(
            agent="search-worker", text="blue https://example.com", urls=["https://example.com"]
        )
    )
    store.save(state)

    loaded = store.load(state.run_id)
    assert loaded.question == "What color is the sky?"
    assert loaded.status == "running"
    assert loaded.session_id == "session_1"
    assert loaded.findings[0].agent == "search-worker"
    assert loaded.findings[0].urls == ["https://example.com"]
    assert store.list_runs() == [state.run_id]


def test_invalid_run_id_rejected(tmp_path):
    store = RunStore(tmp_path)
    with pytest.raises(ValueError):
        store.load("../../etc/passwd")


def test_save_is_atomic_leaves_no_temp_files(tmp_path):
    store = RunStore(tmp_path)
    state = store.new_run("Q")
    state.status = "running"
    store.save(state)
    assert [p.name for p in tmp_path.iterdir()] == [f"{state.run_id}.json"]
    assert store.load(state.run_id).status == "running"


def test_summaries_lists_newest_first(tmp_path):
    store = RunStore(tmp_path)
    first = store.new_run("first question")
    second = store.new_run("second question")
    second.status = "finished"
    second.cost_report = {"total_usd": 1.23}
    second.created_at = first.created_at + 10
    store.save(second)

    summaries = store.summaries()
    assert [s["run_id"] for s in summaries] == [second.run_id, first.run_id]
    assert summaries[0]["question"] == "second question"
    assert summaries[0]["status"] == "finished"
    assert summaries[0]["total_usd"] == 1.23
    assert summaries[1]["total_usd"] is None


def test_summaries_skips_corrupt_files(tmp_path):
    store = RunStore(tmp_path)
    state = store.new_run("Q")
    (tmp_path / "corrupt.json").write_text("{not json")
    assert [s["run_id"] for s in store.summaries()] == [state.run_id]
