from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .bootstrap import initialize
from .exceptions import ConfigurationError, ValidationError
from .pipeline import EvaluationSelection, Pipeline, report_exit_code
from .reporting import ConsoleReporter

app = typer.Typer(
    name="redteamarena",
    help="Security preflight testing for repositories containing AI agents.",
    no_args_is_help=True,
)
console = Console()


def _root(path: Path) -> Path:
    return path.expanduser().resolve()


def _handle_failure(exc: Exception) -> None:
    console.print(f"[red]Error:[/red] {exc}")
    if isinstance(exc, (ConfigurationError, ValidationError)):
        raise typer.Exit(2)
    raise typer.Exit(3)


@app.command("init")
def init_command(
    repository: Path = typer.Option(Path("."), "--repository", "-r", help="Repository root"),
) -> None:
    """Initialize safe local RedTeamArena configuration."""
    try:
        changed = initialize(_root(repository))
        if changed:
            console.print("Initialized RedTeamArena:")
            for path in changed:
                console.print(f"  {path}")
        else:
            console.print("RedTeamArena is already initialized.")
        console.print(
            "No API key was created or stored. Copy .env.example to .env and fill it locally."
        )
    except Exception as exc:
        _handle_failure(exc)


@app.command("scan")
def scan_command(
    repository: Path = typer.Option(Path("."), "--repository", "-r", help="Repository root"),
) -> None:
    """Scan only; no provider calls are made."""
    try:
        scan, profile = Pipeline(_root(repository)).scan()
        console.print(
            f"Scanned {scan.files_scanned} files; detected {len(scan.detections)} unique agent configuration(s)."
        )
        for agent in scan.detections:
            policy = "eligible" if agent.eligible else "reported only"
            console.print(
                f"- {agent.agent_id}: {agent.provider}/{agent.model or agent.model_expression or 'unresolved'} "
                f"{agent.confidence.level.value} ({agent.confidence.score:g}), {policy}"
            )
        console.print(f"Project profile confidence: {profile.confidence.level.value}")
        console.print(f"Evidence: {_root(repository) / '.redteamarena' / 'scan'}")
    except Exception as exc:
        _handle_failure(exc)


@app.command("preflight")
def preflight_command(
    repository: Path = typer.Option(Path("."), "--repository", "-r", help="Repository root"),
) -> None:
    """Run the complete scan, attack generation, isolated execution, and reporting pipeline."""
    try:
        report = Pipeline(_root(repository)).preflight()
        ConsoleReporter(console).render(report)
        raise typer.Exit(report_exit_code(report))
    except typer.Exit:
        raise
    except Exception as exc:
        _handle_failure(exc)


@app.command("evaluate")
def evaluate_command(
    repository: Path = typer.Option(Path("."), "--repository", "-r", help="Repository root"),
    agent: Optional[str] = typer.Option(None, "--agent", help="Run one detected agent ID"),
    scenario: Optional[str] = typer.Option(None, "--scenario", help="Run one scenario ID"),
    full: bool = typer.Option(False, "--full", help="Regenerate and rerun all scenarios"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Ignore scenario cache"),
) -> None:
    """Rerun selected agents/scenarios or force a complete evaluation."""
    try:
        selection = EvaluationSelection(
            agent_id=agent, scenario_id=scenario, full=full, no_cache=no_cache
        )
        report = Pipeline(_root(repository)).preflight(selection)
        ConsoleReporter(console).render(report)
        raise typer.Exit(report_exit_code(report))
    except typer.Exit:
        raise
    except Exception as exc:
        _handle_failure(exc)


@app.command("report")
def report_command(
    repository: Path = typer.Option(Path("."), "--repository", "-r", help="Repository root"),
    agent: Optional[str] = typer.Option(None, "--agent", help="Show one agent ID"),
    failures_only: bool = typer.Option(False, "--failures-only", help="Hide passing scenarios"),
    output_format: str = typer.Option("console", "--format", help="console or json"),
) -> None:
    """Display the latest persisted report without rerunning tests."""
    if output_format not in {"console", "json"}:
        _handle_failure(ValidationError("--format must be console or json"))
    try:
        report = Pipeline(_root(repository)).latest_report()
        if agent and not any(item.agent_id == agent for item in report.agent_reports):
            raise ValidationError(f"Agent {agent} is not present in the latest report")
        ConsoleReporter(console).render(report, agent, failures_only, output_format)
        raise typer.Exit(report_exit_code(report))
    except typer.Exit:
        raise
    except Exception as exc:
        _handle_failure(exc)


if __name__ == "__main__":
    app()
