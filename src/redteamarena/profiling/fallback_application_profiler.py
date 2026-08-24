from __future__ import annotations

import json
from typing import Any, Protocol

from ..exceptions import ValidationError
from ..models import ProjectProfile


class JsonGenerator(Protocol):
    def generate_text(self, system_prompt: str, user_prompt: str) -> str: ...


class FallbackApplicationProfiler:
    """Narrow JSON-only fallback. The caller selects and redacts safe context."""

    def __init__(self, generator: JsonGenerator, prompt: str) -> None:
        self.generator = generator
        self.prompt = prompt

    def profile(self, safe_package: dict[str, Any]) -> ProjectProfile:
        raw = self.generator.generate_text(self.prompt, json.dumps(safe_package, sort_keys=True))
        try:
            profile = ProjectProfile.model_validate_json(raw)
        except Exception as exc:
            raise ValidationError("Application fallback returned invalid profile JSON") from exc
        profile.source = "llm_fallback"
        return profile
