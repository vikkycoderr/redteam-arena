from __future__ import annotations

import json
from typing import Optional

from rich.console import Console
from rich.table import Table

from ..models import OverallResult, RepositoryReport


class ConsoleReporter:
    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console()

    def render(
        self,
        report: RepositoryReport,
        agent_id: Optional[str] = None,
        failures_only: bool = False,
        output_format: str = "console",
    ) -> None:
        filtered = report.model_copy(deep=True)
        if agent_id:
            filtered.agent_reports = [
                item for item in filtered.agent_reports if item.agent_id == agent_id
            ]
        if failures_only:
            for agent in filtered.agent_reports:
                agent.scenario_results = [
                    result
                    for result in agent.scenario_results
                    if result.overall_result != OverallResult.PASS
                ]
        if output_format == "json":
            self.console.print_json(json.dumps(filtered.model_dump(mode="json")))
            return
        self.console.print("[bold]RedTeamArena repository report[/bold]")
        self.console.print(f"Repository: {filtered.repository_root}")
        if filtered.highest_risk_agent:
            self.console.print(f"Highest-risk agent: [red]{filtered.highest_risk_agent}[/red]")
        for agent in filtered.agent_reports:
            table = Table(
                title=f"{agent.agent_id} ({agent.provider}/{agent.model or 'unresolved'})"
            )
            table.add_column("Scenario")
            table.add_column("Category")
            table.add_column("Difficulty")
            table.add_column("Result")
            table.add_column("Evidence")
            for result in agent.scenario_results:
                table.add_row(
                    result.scenario_id,
                    result.category,
                    result.difficulty.value,
                    result.overall_result.value,
                    result.evidence_paths[-1] if result.evidence_paths else "",
                )
            self.console.print(table)
        if filtered.untested_detections:
            self.console.print("[yellow]Untested LOW/UNRESOLVED detections:[/yellow]")
            for detection in filtered.untested_detections:
                self.console.print(
                    f"- {detection['agent_id']}: {detection['confidence_level']} "
                    f"({detection['confidence_score']})"
                )
