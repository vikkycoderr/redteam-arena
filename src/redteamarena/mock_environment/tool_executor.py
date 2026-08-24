from __future__ import annotations

from typing import Any, Dict

from ..models import ActionLogEntry, ProposedAction
from .action_logger import ActionLogger
from .environment import MockEnvironment
from .tool_registry import ToolRegistry


class MockToolExecutor:
    def __init__(
        self, environment: MockEnvironment, registry: ToolRegistry, logger: ActionLogger
    ) -> None:
        self.environment = environment
        self.registry = registry
        self.logger = logger

    def execute(self, action: ProposedAction, step: int) -> Dict[str, Any]:
        validation = self._validate(action)
        handler = self.registry.handlers.get(action.tool)
        entry = ActionLogEntry(
            step=step,
            tool=action.tool,
            arguments=action.arguments,
            validation_result=validation,
            mock_handler=getattr(handler, "__name__", None),
            simulated_status="attempted",
            real_project_touched=False,
        )
        # Persist the attempted action before any handler runs.
        self.logger.append(entry)
        if validation != "valid" or handler is None:
            result = {"status": "rejected", "reason": validation}
            entry.simulated_status = "rejected"
            entry.mock_result = result
            self.logger.replace_last(entry)
            return result
        try:
            result = handler(self.environment, action.arguments)
            entry.simulated_status = "completed"
            entry.mock_result = result
        except Exception as exc:
            result = {"status": "mock_error", "error": type(exc).__name__}
            entry.simulated_status = "mock_error"
            entry.mock_result = result
        self.logger.replace_last(entry)
        return result

    def _validate(self, action: ProposedAction) -> str:
        spec = self.registry.specs.get(action.tool)
        if spec is None:
            return "unknown_tool"
        expected = spec.parameters
        missing = [
            name
            for name, item in expected.items()
            if item.required and name not in action.arguments
        ]
        if missing:
            return "missing_arguments:" + ",".join(missing)
        unknown = set(action.arguments) - set(expected)
        if unknown:
            return "unknown_arguments:" + ",".join(sorted(unknown))
        return "valid"
