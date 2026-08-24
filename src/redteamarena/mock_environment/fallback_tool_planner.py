from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import TypeAdapter

from ..exceptions import ValidationError
from ..models import ToolSpec


class JsonGenerator(Protocol):
    def generate_text(self, system_prompt: str, user_prompt: str) -> str: ...


class FallbackToolPlanner:
    """Validates suggestions and emits data-only specs; it never accepts executable code."""

    def __init__(self, generator: JsonGenerator, prompt: str) -> None:
        self.generator = generator
        self.prompt = prompt

    @staticmethod
    def needs_fallback(tools: list[ToolSpec]) -> bool:
        return not tools

    def plan(self, safe_package: dict[str, Any]) -> list[ToolSpec]:
        raw = self.generator.generate_text(self.prompt, json.dumps(safe_package, sort_keys=True))
        try:
            data = json.loads(raw)
            tools = TypeAdapter(list[ToolSpec]).validate_python(data.get("tools", data))
        except Exception as exc:
            raise ValidationError(
                "Mock-tool fallback returned invalid tool specifications"
            ) from exc
        rejected = {"shell", "exec", "eval", "subprocess", "real_tool", "network_request"}
        safe: list[ToolSpec] = []
        for tool in tools:
            if any(token in tool.name.lower() for token in rejected):
                continue
            tool.source = "llm_fallback"
            tool.description_confidence = min(tool.description_confidence, 0.5)
            safe.append(tool)
        return safe
