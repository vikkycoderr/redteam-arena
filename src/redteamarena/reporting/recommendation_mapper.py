from __future__ import annotations

from typing import Iterable, List


class RecommendationMapper:
    MAPPINGS = {
        "forbidden_tool_call": [
            "Add authorization checks before sensitive tool calls.",
            "Require explicit user confirmation for sensitive operations.",
        ],
        "forbidden_tool_argument": [
            "Validate sensitive tool arguments against an explicit allowlist.",
        ],
        "forbidden_path_read": [
            "Add path-level read access control.",
            "Deny secret and protected paths by default.",
        ],
        "forbidden_path_write": [
            "Restrict writes to an explicit workspace allowlist.",
            "Require confirmation before changing protected configuration.",
        ],
        "required_tool_call": [
            "Add a deterministic completion check for the legitimate task.",
        ],
        "required_tool_argument": [
            "Validate that required task parameters are preserved through execution.",
        ],
        "forbidden_output_string": [
            "Filter prohibited output deterministically before returning it.",
        ],
        "fake_secret_leakage": [
            "Remove secrets from model context.",
            "Add deterministic output redaction.",
        ],
        "maximum_tool_calls": [
            "Enforce a strict per-request tool-call budget.",
        ],
        "required_task_completion_marker": [
            "Require a machine-checkable task completion marker.",
        ],
        "prompt_injection": [
            "Treat document instructions as untrusted data.",
            "Prioritize explicit user instructions over artifact contents.",
        ],
        "configuration_manipulation": [
            "Separate configuration mutation from routine task execution.",
        ],
    }

    def for_keys(self, keys: Iterable[str]) -> List[str]:
        result: List[str] = []
        for key in keys:
            result.extend(self.MAPPINGS.get(key, []))
        return list(dict.fromkeys(result))
