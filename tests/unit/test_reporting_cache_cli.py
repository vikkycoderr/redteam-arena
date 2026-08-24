from __future__ import annotations

from pathlib import Path

from conftest import make_scenario
from test_mock_and_evaluation import completed
from typer.testing import CliRunner

from redteamarena.cli import app
from redteamarena.evaluation import DeterministicChecker
from redteamarena.models import ConfidenceLevel, ExecutionResult, ModelResult
from redteamarena.pipeline import report_exit_code
from redteamarena.reporting import AgentSummaryBuilder, RepositoryReportBuilder
from redteamarena.storage import CacheStore


def test_cache_invalidation() -> None:
    first = CacheStore.security_key({"model": "a", "prompt": "v1"})
    same = CacheStore.security_key({"prompt": "v1", "model": "a"})
    changed = CacheStore.security_key({"model": "a", "prompt": "v2"})
    assert first == same
    assert first != changed


def test_per_agent_and_repository_reporting(tmp_path: Path, agent) -> None:
    result = DeterministicChecker().evaluate(
        make_scenario(),
        ModelResult(text="done"),
        [completed("run_evaluation")],
        {},
        ExecutionResult.COMPLETED,
        [],
    )
    agent_report = AgentSummaryBuilder().build(agent, [result])
    repository = RepositoryReportBuilder().build(tmp_path, [agent_report], [agent])
    assert agent_report.full_pass_rate == 1.0
    assert repository.totals["PASS"] == 1
    assert report_exit_code(repository) == 0


def test_low_detection_appears_in_repository_report(tmp_path: Path, agent) -> None:
    low = agent.model_copy(deep=True)
    low.agent_id = "low"
    low.confidence.score = 1
    low.confidence.level = ConfidenceLevel.LOW
    repository = RepositoryReportBuilder().build(tmp_path, [], [low])
    assert repository.untested_detections[0]["agent_id"] == "low"
    assert report_exit_code(repository) == 4


def test_cli_init_scan_and_missing_configuration(tmp_path: Path) -> None:
    runner = CliRunner()
    initialized = runner.invoke(app, ["init", "--repository", str(tmp_path)])
    assert initialized.exit_code == 0
    (tmp_path / "agent.py").write_text(
        "from openai import OpenAI\nc=OpenAI()\n"
        "c.chat.completions.create(model='model',messages=[])\n"
    )
    scanned = runner.invoke(app, ["scan", "--repository", str(tmp_path)])
    assert scanned.exit_code == 0
    assert "detected 1" in scanned.stdout
    preflight = runner.invoke(app, ["preflight", "--repository", str(tmp_path)])
    assert preflight.exit_code == 2
    assert "RTA_ATTACK_API_KEY" in preflight.stdout
