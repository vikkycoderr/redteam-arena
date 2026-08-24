from __future__ import annotations

from pathlib import Path

from ..models import AgentReport, RepositoryReport, ScenarioResult
from .json_store import JsonStore
from .path_manager import PathManager


class ResultStore:
    def __init__(self, paths: PathManager) -> None:
        self.paths = paths

    def save_scenario(self, agent_id: str, fingerprint: str, result: ScenarioResult) -> Path:
        path = (
            self.paths.agent_root(agent_id, fingerprint)
            / "results"
            / "scenarios"
            / f"{result.scenario_id}.json"
        )
        JsonStore.write(path, result)
        return path

    def save_agent_report(self, report: AgentReport) -> Path:
        root = self.paths.agent_root(report.agent_id, report.configuration_fingerprint)
        path = root / "results" / "report.json"
        JsonStore.write(path, report)
        JsonStore.write(
            root / "results" / "summary.json", report.model_dump(exclude={"scenario_results"})
        )
        return path

    def save_repository_report(self, report: RepositoryReport) -> Path:
        path = self.paths.reports / "repository_report.json"
        JsonStore.write(path, report)
        JsonStore.write(
            self.paths.reports / "repository_summary.json",
            report.model_dump(exclude={"agent_reports"}),
        )
        return path
