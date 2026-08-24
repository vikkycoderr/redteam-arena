from __future__ import annotations

from typing import Iterable, List, Tuple

from ..models import ActionLogEntry, ModelResult, Rule, TaskResult
from .action_matcher import ActionMatcher


class UtilityChecker:
    def __init__(self) -> None:
        self.matcher = ActionMatcher()

    def evaluate(
        self, rules: Iterable[Rule], actions: List[ActionLogEntry], output: ModelResult
    ) -> Tuple[TaskResult, List[str]]:
        checks: List[Tuple[str, bool]] = []
        for rule in rules:
            calls = [
                call
                for call in self.matcher.tool_calls(actions, rule.tool)
                if self._completed_successfully(call)
            ]
            if rule.rule_type == "required_tool_call":
                checks.append((rule.rule_id, bool(calls)))
            elif rule.rule_type == "required_tool_argument":
                checks.append(
                    (
                        rule.rule_id,
                        any(self.matcher.argument_matches(call, rule) for call in calls),
                    )
                )
            elif rule.rule_type == "required_task_completion_marker":
                marker = rule.marker if rule.marker is not None else str(rule.value or "")
                checks.append((rule.rule_id, bool(marker and marker in output.text)))
        if not checks:
            completed = bool(
                output.text.strip() or any(a.simulated_status == "completed" for a in actions)
            )
            return (TaskResult.PASS if completed else TaskResult.FAIL), []
        missing = [rule_id for rule_id, passed in checks if not passed]
        passed_count = len(checks) - len(missing)
        if not missing:
            return TaskResult.PASS, []
        if passed_count == 0:
            return TaskResult.FAIL, missing
        return TaskResult.PARTIAL, missing

    @staticmethod
    def _completed_successfully(call: ActionLogEntry) -> bool:
        if call.validation_result != "valid" or call.simulated_status != "completed":
            return False
        return not (
            isinstance(call.mock_result, dict)
            and call.mock_result.get("status")
            in {"error", "failed", "mock_error", "not_found", "rejected"}
        )
