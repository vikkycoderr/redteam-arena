from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Dict, List, Optional

from ..adapters import AdapterRegistry, TextGenerator, UnavailableAdapter
from ..attacks import (
    AttackAgent,
    AttackPromptBuilder,
    ScenarioGenerator,
    ScenarioLoader,
    ScenarioRepairer,
    ScenarioValidator,
)
from ..config import Settings, load_settings
from ..evaluation import DeterministicChecker
from ..exceptions import ConfigurationError, ValidationError
from ..execution import AgentRunner, ExecutionLimits, ScenarioRunner
from ..mock_environment import EnvironmentGenerator, FallbackToolPlanner
from ..models import (
    AgentDetection,
    AgentReport,
    OverallResult,
    ProjectProfile,
    RepositoryReport,
    RepositoryScan,
    ScenarioSet,
)
from ..profiling import (
    ApplicationProfiler,
    CapabilityDetector,
    FallbackApplicationProfiler,
    ProtectedResourceDetector,
)
from ..reporting import AgentSummaryBuilder, RepositoryReportBuilder
from ..scanner import RepositoryScanner
from ..storage import CacheStore, JsonStore, PathManager, ResultStore
from ..utils.redaction import SecretRedactor


@dataclass
class EvaluationSelection:
    agent_id: Optional[str] = None
    scenario_id: Optional[str] = None
    full: bool = False
    no_cache: bool = False


class Pipeline:
    def __init__(
        self,
        root: Path,
        settings: Optional[Settings] = None,
        adapter_registry: Optional[AdapterRegistry] = None,
    ) -> None:
        self.root = root.resolve()
        self.paths = PathManager(self.root)
        self.paths.ensure_base()
        self.settings = settings or load_settings(self.root)
        self.adapters = adapter_registry or AdapterRegistry()
        self.result_store = ResultStore(self.paths)

    def scan(self) -> tuple[RepositoryScan, ProjectProfile]:
        scan = RepositoryScanner().scan(self.root)
        files = RepositoryScanner().files(self.root)
        protected, protected_confidence = ProtectedResourceDetector().detect(files, self.root)
        capability_detector = CapabilityDetector()
        for agent in scan.detections:
            agent.protected_resources = sorted(set(agent.protected_resources) | set(protected))
            agent.protected_resource_confidence = protected_confidence.model_copy(deep=True)
            agent.capabilities = capability_detector.refine(agent)
            agent.compute_fingerprint()
        profile = ApplicationProfiler().profile(self.root, scan)
        profile.protected_resources = protected
        JsonStore.write(self.paths.scan / "repository_scan.json", scan)
        JsonStore.write(self.paths.scan / "agents.json", scan.detections)
        JsonStore.write(self.paths.scan / "project_profile.json", profile)
        return scan, profile

    def preflight(self, selection: Optional[EvaluationSelection] = None) -> RepositoryReport:
        selection = selection or EvaluationSelection()
        scan, profile = self.scan()
        eligible = [agent for agent in scan.detections if agent.eligible]
        if selection.agent_id:
            eligible = [agent for agent in eligible if agent.agent_id == selection.agent_id]
            if not eligible:
                raise ValidationError(f"No eligible detected agent matches {selection.agent_id}")
        if not eligible:
            empty_report = RepositoryReportBuilder().build(self.root, [], scan.detections)
            self.result_store.save_repository_report(empty_report)
            return empty_report

        self.settings.attack_agent.validate_for_generation(self.settings.environment)
        attack_adapter = self.adapters.for_attack(
            self.settings.attack_agent, self.settings.environment
        )
        attack_agent = AttackAgent(attack_adapter, self.settings.attack_agent)
        fallback_generator = TextGenerator(
            attack_adapter,
            self.settings.attack_agent.model,
            self.settings.attack_agent.timeout_seconds,
            self.settings.attack_agent.temperature,
        )
        profiler = ApplicationProfiler()
        if profiler.needs_fallback(
            profile, self.settings.pipeline.application_confidence_threshold
        ):
            fallback_prompt = (
                files("redteamarena.prompts")
                .joinpath("application_fallback_v1.txt")
                .read_text(encoding="utf-8")
            )
            try:
                profile = FallbackApplicationProfiler(fallback_generator, fallback_prompt).profile(
                    profiler.safe_fallback_package(profile, scan, self.root)
                )
                JsonStore.write(self.paths.scan / "project_profile.json", profile)
            except ValidationError as exc:
                scan.warnings.append(str(exc))
        prompt_builder = AttackPromptBuilder(self.settings.attack_agent.prompt_version)
        validator = ScenarioValidator(max_scenarios=self.settings.pipeline.max_scenarios)
        repair_generator = TextGenerator(
            attack_adapter,
            self.settings.attack_agent.model,
            self.settings.attack_agent.timeout_seconds,
            self.settings.attack_agent.temperature,
        )
        repairer = ScenarioRepairer(
            repair_generator, validator, self.settings.pipeline.repair_attempts
        )
        generator = ScenarioGenerator(attack_agent, prompt_builder, validator, repairer)
        scenario_cache = CacheStore(self.paths.cache / "scenario_cache.json")
        fingerprint_cache: Dict[str, list[str]] = {}
        reports: List[AgentReport] = []
        for original in eligible:
            agent = self._resolve_runtime_model(original)
            if FallbackToolPlanner.needs_fallback(agent.tools):
                tool_prompt = (
                    files("redteamarena.prompts")
                    .joinpath("mock_tool_fallback_v1.txt")
                    .read_text(encoding="utf-8")
                )
                try:
                    agent.tools = FallbackToolPlanner(fallback_generator, tool_prompt).plan(
                        {
                            "project_profile": profile.model_dump(mode="json"),
                            "agent_purpose": profile.agent_purposes.get(agent.agent_id, ""),
                            "important_operations": profile.important_operations,
                            "protected_resources": agent.protected_resources,
                            "target_capabilities": agent.capabilities.model_dump(mode="json"),
                        }
                    )
                    agent.compute_fingerprint()
                except ValidationError as exc:
                    scan.warnings.append(str(exc))
            EnvironmentGenerator(self.paths).generate(agent)
            scenario_set = self._scenario_set(
                agent,
                profile,
                generator,
                scenario_cache,
                read_cache=not (selection.no_cache or selection.full),
                write_cache=not selection.no_cache,
            )
            if selection.scenario_id:
                scenario_set = scenario_set.model_copy(deep=True)
                scenario_set.scenarios = [
                    item
                    for item in scenario_set.scenarios
                    if item.scenario_id == selection.scenario_id
                ]
                if not scenario_set.scenarios:
                    continue
            try:
                target_adapter = self.adapters.for_target(agent, self.settings.environment)
            except ConfigurationError as exc:
                target_adapter = UnavailableAdapter(str(exc), provider=agent.provider)
            scenario_runner = ScenarioRunner(
                self.paths,
                target_adapter,
                ExecutionLimits(
                    max_tool_steps=self.settings.pipeline.max_tool_steps,
                    timeout_seconds=self.settings.attack_agent.timeout_seconds,
                ),
                SecretRedactor(
                    self.settings.environment,
                    protected_names=[
                        self.settings.attack_agent.api_key_env,
                        *agent.credential_env_vars,
                    ],
                ),
            )
            runner = AgentRunner(scenario_runner, DeterministicChecker(), self.result_store)
            results = runner.run(agent, scenario_set)
            agent_report = AgentSummaryBuilder().build(agent, results)
            self.result_store.save_agent_report(agent_report)
            reports.append(agent_report)
            fingerprint_cache.setdefault(agent.configuration_fingerprint, []).extend(
                agent.call_sites + agent.equivalent_call_sites
            )
        if selection.scenario_id and not reports:
            raise ValidationError(
                f"Scenario {selection.scenario_id} was not found for any eligible agent"
            )
        JsonStore.write(self.paths.cache / "configuration_fingerprints.json", fingerprint_cache)
        repository_report = RepositoryReportBuilder().build(self.root, reports, scan.detections)
        self.result_store.save_repository_report(repository_report)
        return repository_report

    def latest_report(self) -> RepositoryReport:
        path = self.paths.reports / "repository_report.json"
        if not path.exists():
            raise ValidationError("No report exists. Run redteamarena preflight first.")
        return RepositoryReport.model_validate(JsonStore.read(path))

    def _scenario_set(
        self,
        agent: AgentDetection,
        profile: ProjectProfile,
        generator: ScenarioGenerator,
        cache: CacheStore,
        read_cache: bool,
        write_cache: bool,
    ) -> ScenarioSet:
        output = (
            self.paths.agent_root(agent.agent_id, agent.configuration_fingerprint)
            / "generated"
            / "scenarios.json"
        )
        key_payload = {
            "configuration_fingerprint": agent.configuration_fingerprint,
            "attack_prompt_version": self.settings.attack_agent.prompt_version,
            "scenario_schema_version": "2.6.0",
            "protected_resources": agent.protected_resources,
            "capabilities": agent.capabilities.model_dump(mode="json"),
        }
        key = CacheStore.security_key(key_payload)
        cached = cache.get(key) if read_cache else None
        if isinstance(cached, str):
            unresolved = self.root / cached if not Path(cached).is_absolute() else Path(cached)
            candidate = unresolved.resolve()
            state = self.paths.state.resolve()
            if (
                candidate.exists()
                and not unresolved.is_symlink()
                and state in candidate.parents
                and candidate.name == "scenarios.json"
            ):
                loaded = ScenarioLoader.load(candidate)
                report = ScenarioValidator(
                    max_scenarios=self.settings.pipeline.max_scenarios
                ).validate(loaded.model_dump(mode="json"), agent)
                if report.valid:
                    return loaded
        generated = generator.generate(profile, agent, output)
        if write_cache:
            cache.put(key, output.relative_to(self.root).as_posix())
        return generated

    def _resolve_runtime_model(self, agent: AgentDetection) -> AgentDetection:
        result = agent.model_copy(deep=True)
        if not result.model:
            for name in result.configuration_env_vars:
                value = self.settings.environment.get(name)
                if value:
                    result.model = value
                    break
        result.compute_fingerprint()
        return result


def report_exit_code(report: RepositoryReport) -> int:
    results = [result for agent in report.agent_reports for result in agent.scenario_results]
    if any(result.overall_result == OverallResult.SECURITY_FAIL for result in results):
        return 1
    if not results or all(
        result.overall_result == OverallResult.INDETERMINATE for result in results
    ):
        return 4
    return 0
