from __future__ import annotations

from typing import List

from ..evaluation import DeterministicChecker
from ..models import AgentDetection, ScenarioResult, ScenarioSet
from ..storage import ResultStore
from .scenario_runner import ScenarioRunner


class AgentRunner:
    def __init__(
        self, runner: ScenarioRunner, checker: DeterministicChecker, store: ResultStore
    ) -> None:
        self.runner = runner
        self.checker = checker
        self.store = store

    def run(self, agent: AgentDetection, scenario_set: ScenarioSet) -> List[ScenarioResult]:
        results: List[ScenarioResult] = []
        for scenario in scenario_set.scenarios:
            model_result, actions, state, execution, evidence = self.runner.run(agent, scenario)
            result = self.checker.evaluate(
                scenario, model_result, actions, state, execution, evidence
            )
            path = self.store.save_scenario(agent.agent_id, agent.configuration_fingerprint, result)
            result.evidence_paths.append(path.as_posix())
            self.store.save_scenario(agent.agent_id, agent.configuration_fingerprint, result)
            results.append(result)
        return results
