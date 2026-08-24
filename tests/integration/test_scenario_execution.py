from __future__ import annotations

from pathlib import Path

from conftest import make_scenario

from redteamarena.adapters import BaseAdapter
from redteamarena.evaluation import DeterministicChecker
from redteamarena.exceptions import ProviderError
from redteamarena.execution import ExecutionLimits, ScenarioRunner
from redteamarena.mock_environment import EnvironmentGenerator
from redteamarena.models import OverallResult
from redteamarena.storage import PathManager
from redteamarena.utils.redaction import SecretRedactor


class ToolThenFinalAdapter(BaseAdapter):
    provider = "openai_compatible"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        if self.calls == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "1",
                                    "function": {"name": "run_evaluation", "arguments": "{}"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        return {
            "choices": [{"message": {"content": "Evaluation complete"}, "finish_reason": "stop"}]
        }


class StructuredFallbackAdapter(BaseAdapter):
    provider = "openai_compatible"

    def __init__(self) -> None:
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            raise ProviderError(
                "OpenAI-compatible provider returned HTTP 404: "
                "No endpoints found that support tool use"
            )
        if len(self.requests) == 2:
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"action":"run_evaluation","arguments":{}}'
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        return {
            "choices": [{"message": {"content": "Evaluation complete"}, "finish_reason": "stop"}]
        }


class TwoActionAdapter(BaseAdapter):
    provider = "openai_compatible"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        if self.calls <= 2:
            name = "run_evaluation" if self.calls == 1 else "read_file"
            arguments = "{}" if self.calls == 1 else '{"path":"README.txt"}'
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": str(self.calls),
                                    "function": {"name": name, "arguments": arguments},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        return {"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]}


def test_scenario_runner_isolated_multistep(tmp_path: Path, agent) -> None:
    paths = PathManager(tmp_path)
    paths.ensure_base()
    EnvironmentGenerator(paths).generate(agent)
    runner = ScenarioRunner(
        paths,
        ToolThenFinalAdapter(),
        ExecutionLimits(max_tool_steps=3),
        SecretRedactor({}),
    )
    scenario = make_scenario()
    output, actions, state, execution, evidence = runner.run(agent, scenario)
    result = DeterministicChecker().evaluate(scenario, output, actions, state, execution, evidence)
    assert result.overall_result == OverallResult.PASS
    assert actions[0].tool == "run_evaluation"
    assert (
        paths.scenario_runtime(
            agent.agent_id, agent.configuration_fingerprint, scenario.scenario_id
        )
        / "action_log.json"
    ).is_file()


def test_scenario_runner_falls_back_when_native_tools_are_unsupported(
    tmp_path: Path, agent
) -> None:
    paths = PathManager(tmp_path)
    paths.ensure_base()
    EnvironmentGenerator(paths).generate(agent)
    adapter = StructuredFallbackAdapter()
    runner = ScenarioRunner(
        paths,
        adapter,
        ExecutionLimits(max_tool_steps=3),
        SecretRedactor({}),
    )

    scenario = make_scenario()
    output, actions, state, execution, evidence = runner.run(agent, scenario)
    result = DeterministicChecker().evaluate(scenario, output, actions, state, execution, evidence)

    assert result.overall_result == OverallResult.PASS
    assert adapter.requests[0].tools
    assert adapter.requests[1].tools == []
    assert all(message["role"] != "tool" for message in adapter.requests[2].messages)
    assert actions[0].tool == "run_evaluation"
    assert output.proposed_actions[0].tool == "run_evaluation"
    assert len(output.proposed_actions) == len(actions) == 1


def test_normalized_result_preserves_all_proposals_across_steps(tmp_path: Path, agent) -> None:
    paths = PathManager(tmp_path)
    paths.ensure_base()
    EnvironmentGenerator(paths).generate(agent)
    runner = ScenarioRunner(
        paths,
        TwoActionAdapter(),
        ExecutionLimits(max_tool_steps=3),
        SecretRedactor({}),
    )

    output, actions, _, execution, _ = runner.run(agent, make_scenario())

    assert execution.value == "COMPLETED"
    assert [action.tool for action in output.proposed_actions] == [
        "run_evaluation",
        "read_file",
    ]
    assert [action.tool for action in actions] == ["run_evaluation", "read_file"]
