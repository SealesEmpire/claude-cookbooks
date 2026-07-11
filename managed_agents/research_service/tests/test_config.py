"""Config loading and validation."""

import pytest
from research_service.config import (
    ConfigError,
    load_pricing,
    load_team,
)


def test_default_team_loads(team_config):
    assert team_config.coordinator.name == "research-coordinator"
    assert len(team_config.workers) == 5
    names = {w.name for w in team_config.workers}
    assert {
        "premise-checker",
        "search-worker",
        "fetch-extract-worker",
        "fact-checker",
        "citation-formatter",
    } == names
    assert team_config.betas == ("managed-agents-2026-04-01",)
    assert team_config.solo.tools == ("web_search", "web_fetch")


def test_coordinator_prompt_renders_parallelism(team_config):
    rendered = team_config.coordinator.rendered_system()
    assert "{max_parallel_workers}" not in rendered
    assert str(team_config.coordinator.max_parallel_workers) in rendered


def test_worker_lookup(team_config):
    assert team_config.worker("fact-checker").tools == ("web_search", "web_fetch")
    assert team_config.worker("citation-formatter").tools == ()
    with pytest.raises(KeyError):
        team_config.worker("nope")


def test_model_env_overrides(monkeypatch):
    monkeypatch.setenv("COOKBOOK_COORDINATOR_MODEL", "coordinator-override")
    monkeypatch.setenv("COOKBOOK_WORKER_MODEL", "worker-override")
    team = load_team()
    assert team.coordinator.model == "coordinator-override"
    assert all(w.model == "worker-override" for w in team.workers)
    assert team.solo.model == "coordinator-override"


def test_default_pricing_loads(pricing_config):
    rates = pricing_config.rates("claude-sonnet-5")
    assert rates.input == 2.0
    assert rates.output == 10.0
    assert pricing_config.cache_multipliers.read == 0.1
    assert pricing_config.version


def test_pricing_missing_model_is_config_error(pricing_config):
    with pytest.raises(ConfigError, match="no pricing configured"):
        pricing_config.rates("claude-nonexistent")


def test_malformed_team_yaml(tmp_path):
    bad = tmp_path / "team.yaml"
    bad.write_text("coordinator:\n  name: x\n")
    with pytest.raises(ConfigError):
        load_team(bad)


def test_malformed_pricing_yaml(tmp_path):
    bad = tmp_path / "pricing.yaml"
    bad.write_text("models:\n  m:\n    input: 1.0\n")
    with pytest.raises(ConfigError):
        load_pricing(bad)
