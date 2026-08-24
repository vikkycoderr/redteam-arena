from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Any, Iterable

from ..models import ActionLogEntry, ProposedAction, Rule


class ActionMatcher:
    @staticmethod
    def tool_calls(actions: Iterable[ActionLogEntry], tool: str | None) -> list[ActionLogEntry]:
        return [action for action in actions if action.tool == tool]

    @staticmethod
    def proposed_tool_calls(
        actions: Iterable[ProposedAction], tool: str | None
    ) -> list[ProposedAction]:
        return [action for action in actions if action.tool == tool]

    def argument_matches(self, action: ActionLogEntry | ProposedAction, rule: Rule) -> bool:
        expected = dict(rule.arguments)
        if rule.argument:
            expected[rule.argument] = rule.value
        return bool(expected) and all(
            key in action.arguments and self._equivalent(action.arguments[key], value)
            for key, value in expected.items()
        )

    def path_matches(self, action: ActionLogEntry | ProposedAction, prefix: str | None) -> bool:
        if not prefix:
            return False
        expected = self._normalize(prefix)
        for key in ("path", "file", "filename", "target", "source"):
            value = action.arguments.get(key)
            if isinstance(value, str):
                candidate = self._normalize(value)
                if candidate == expected or candidate.startswith(expected.rstrip("/") + "/"):
                    return True
        return False

    @staticmethod
    def _normalize(value: str) -> str:
        return str(PurePosixPath(value.lstrip("/")))

    @staticmethod
    def _equivalent(actual: Any, expected: Any) -> bool:
        if type(actual) is type(expected):
            if isinstance(actual, str) and isinstance(expected, str):
                return actual.casefold() == expected.casefold()
            return bool(actual == expected)
        # OpenAI-compatible providers sometimes serialize JSON-compatible tool
        # arguments as strings. Match their scalar meaning without treating
        # arbitrary prose as a structured value (and without Python's True == 1).
        if isinstance(actual, str) != isinstance(expected, str):
            encoded = actual if isinstance(actual, str) else expected
            other = expected if isinstance(actual, str) else actual
            try:
                decoded = json.loads(encoded)
            except json.JSONDecodeError:
                return False
            if isinstance(decoded, (dict, list)):
                return False
            return ActionMatcher._equivalent(decoded, other)
        return False
