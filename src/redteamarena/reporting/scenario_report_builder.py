from __future__ import annotations

from typing import Any, Dict

from ..models import Scenario, ScenarioResult


class ScenarioReportBuilder:
    def build(self, scenario: Scenario, result: ScenarioResult) -> Dict[str, Any]:
        return {
            **result.model_dump(mode="json"),
            "what_was_tested": scenario.scenario_description,
            "attack_description": scenario.attack_metadata.attack_description,
            "expected_safe_behavior": scenario.attack_metadata.expected_safe_behavior,
            "actual_response": result.normalized_response.text,
            "actual_attempted_actions": [
                item.model_dump(mode="json") for item in result.action_log
            ],
            "why_it_failed": self._why(result),
            "security_impact": scenario.attack_metadata.risk_description,
        }

    @staticmethod
    def _why(result: ScenarioResult) -> str:
        reasons = []
        if result.violated_rules:
            reasons.append("Violated deterministic rules: " + ", ".join(result.violated_rules))
        if result.missing_required_actions:
            reasons.append(
                "Missing required actions: " + ", ".join(result.missing_required_actions)
            )
        if result.overall_result.value == "INDETERMINATE":
            reasons.append(f"Execution was {result.execution_result.value}")
        return "; ".join(reasons) or "No failure"
