from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

from ..adapters import AdapterRequest, BaseAdapter
from ..artifacts import ArtifactBuilder
from ..exceptions import ConfigurationError, ProviderError, ProviderTimeout
from ..mock_environment import (
    ActionLogger,
    EnvironmentCloner,
    MockToolExecutor,
    ToolRegistry,
)
from ..models import (
    ActionLogEntry,
    AgentDetection,
    ExecutionResult,
    ModelResult,
    ProposedAction,
    Scenario,
)
from ..storage import JsonStore, PathManager
from ..utils.redaction import SecretRedactor
from .conversation_manager import ConversationManager
from .execution_limits import ExecutionLimits
from .input_builder import InputBuilder
from .response_normalizer import ResponseNormalizer


class ScenarioRunner:
    def __init__(
        self,
        paths: PathManager,
        adapter: BaseAdapter,
        limits: ExecutionLimits,
        redactor: SecretRedactor,
    ) -> None:
        self.paths = paths
        self.adapter = adapter
        self.limits = limits
        self.redactor = redactor
        self.cloner = EnvironmentCloner(paths)
        self.artifact_builder = ArtifactBuilder()
        self.input_builder = InputBuilder()
        self.normalizer = ResponseNormalizer()

    def run(
        self, agent: AgentDetection, scenario: Scenario
    ) -> Tuple[
        ModelResult,
        List[ActionLogEntry],
        dict[str, Any],
        ExecutionResult,
        List[Path],
    ]:
        environment = self.cloner.clone(agent, scenario.scenario_id)
        runtime = self.paths.scenario_runtime(
            agent.agent_id, agent.configuration_fingerprint, scenario.scenario_id
        )
        built = [
            self.artifact_builder.build(spec, runtime / "artifacts")
            for spec in scenario.target_visible.artifacts
        ]
        package = self.input_builder.build(scenario, agent, built)
        conversation = ConversationManager(package["messages"])
        registry = ToolRegistry(agent.tools)
        logger = ActionLogger(runtime / "action_log.json")
        executor = MockToolExecutor(environment, registry, logger)
        raw_dir = runtime / "raw_responses"
        raw_dir.mkdir(parents=True, exist_ok=True)
        evidence: List[Path] = [runtime / "action_log.json"] + [item.path for item in built]
        final = ModelResult(text="", format_status="invalid")
        all_proposed_actions: List[ProposedAction] = []
        execution = ExecutionResult.COMPLETED
        native_tools_available = bool(package["tools"])
        for step in range(1, self.limits.max_tool_steps + 1):
            try:
                raw = self.adapter.complete(
                    AdapterRequest(
                        model=agent.model or "",
                        messages=conversation.messages,
                        tools=package["tools"] if native_tools_available else [],
                        artifacts=package["artifacts"] if step == 1 else [],
                        timeout_seconds=self.limits.timeout_seconds,
                    )
                )
            except ProviderTimeout:
                execution = ExecutionResult.TIMEOUT
                break
            except ConfigurationError as exc:
                final.finish_reason = str(exc)
                execution = ExecutionResult.CONFIGURATION_ERROR
                break
            except ProviderError as exc:
                if native_tools_available and self._native_tool_use_unsupported(exc):
                    native_tools_available = False
                    conversation.enable_structured_tool_fallback(package["tools"])
                    try:
                        raw = self.adapter.complete(
                            AdapterRequest(
                                model=agent.model or "",
                                messages=conversation.messages,
                                tools=[],
                                artifacts=package["artifacts"] if step == 1 else [],
                                timeout_seconds=self.limits.timeout_seconds,
                            )
                        )
                    except ProviderTimeout:
                        execution = ExecutionResult.TIMEOUT
                        break
                    except ProviderError as fallback_exc:
                        final.finish_reason = str(fallback_exc)
                        execution = ExecutionResult.PROVIDER_ERROR
                        break
                else:
                    final.finish_reason = str(exc)
                    execution = ExecutionResult.PROVIDER_ERROR
                    break
            safe_raw = self.redactor.value(raw)
            raw_path = raw_dir / f"step_{step:02d}.json"
            JsonStore.write(raw_path, safe_raw)
            evidence.append(raw_path)
            final = self.normalizer.normalize(safe_raw, self.adapter.provider)
            final.raw_response_reference = raw_path.as_posix()
            if not final.proposed_actions:
                break
            for index, action in enumerate(final.proposed_actions, start=1):
                if not action.call_id:
                    action.call_id = f"rta_step_{step}_{index}"
            all_proposed_actions.extend(
                action.model_copy(deep=True) for action in final.proposed_actions
            )
            results = [executor.execute(action, step) for action in final.proposed_actions]
            if native_tools_available:
                conversation.add_result(final, results)
            else:
                conversation.add_structured_result(final, results)
        else:
            execution = ExecutionResult.STEP_LIMIT
        final.proposed_actions = all_proposed_actions
        final.text = self.redactor.text(final.text)
        final_state_path = runtime / "final_state.json"
        JsonStore.write(final_state_path, environment.state)
        evidence.append(final_state_path)
        return final, logger.entries, environment.state, execution, evidence

    @staticmethod
    def _native_tool_use_unsupported(error: ProviderError) -> bool:
        message = str(error).lower()
        return "no endpoints found that support tool use" in message or (
            "tool use" in message and "unsupported" in message
        )
