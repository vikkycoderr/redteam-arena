from __future__ import annotations

import json
import re
from importlib.resources import files
from typing import Any, Protocol

from ..exceptions import ValidationError
from ..models import APPROVED_CATEGORIES, RULE_TYPES, AgentDetection, Scenario
from .scenario_validator import ScenarioValidator


class JsonGenerator(Protocol):
    def generate_text(self, system_prompt: str, user_prompt: str) -> str: ...


class ScenarioRepairer:
    FENCED_JSON = re.compile(
        r"\A```(?:json)?[ \t]*(?:\r?\n)?(?P<body>.*?)(?:\r?\n)?```[ \t]*\Z",
        re.IGNORECASE | re.DOTALL,
    )

    def __init__(
        self, generator: JsonGenerator, validator: ScenarioValidator, attempts: int = 2
    ) -> None:
        self.generator = generator
        self.validator = validator
        self.attempts = attempts
        self.prompt = (
            files("redteamarena.prompts")
            .joinpath("scenario_repair_v1.txt")
            .read_text(encoding="utf-8")
        )

    def repair(self, scenario: Scenario, errors: list[str], agent: AgentDetection) -> Scenario:
        current = scenario
        current_errors = errors
        for _ in range(self.attempts):
            package: dict[str, Any] = {
                "invalid_scenario": current.model_dump(mode="json"),
                "validation_errors": current_errors,
                "valid_tools": [tool.provider_schema() for tool in agent.tools],
                "protected_resources": agent.protected_resources,
                "capabilities": agent.capabilities.model_dump(mode="json"),
                "approved_categories": sorted(APPROVED_CATEGORIES),
                "rule_types": sorted(RULE_TYPES),
            }
            raw = self.generator.generate_text(self.prompt, json.dumps(package, sort_keys=True))
            raw = self._extract_json_object(raw)
            try:
                current = Scenario.model_validate_json(raw)
            except Exception:
                current_errors = ["repair response is not a valid scenario object"]
                continue
            current_errors = self.validator.validate_one(current, agent)
            if not current_errors:
                return current
        raise ValidationError(
            "Scenario remained invalid after repair: " + "; ".join(current_errors)
        )

    @classmethod
    def _extract_json_object(cls, raw: str) -> str:
        stripped = raw.strip()
        match = cls.FENCED_JSON.fullmatch(stripped)
        if match:
            return match.group("body").strip()
        start = stripped.find("{")
        if start < 0:
            return stripped
        try:
            _, end = json.JSONDecoder().raw_decode(stripped[start:])
        except json.JSONDecodeError:
            return stripped
        trailing = stripped[start + end :].strip()
        if "{" in trailing:
            return stripped
        return stripped[start : start + end]
