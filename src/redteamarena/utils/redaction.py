from __future__ import annotations

import re
from typing import Any, Dict, Iterable

SECRET_NAME = re.compile(r"(?i)(api[_-]?key|secret|token|password|credential)")


class SecretRedactor:
    def __init__(self, environment: Dict[str, str], protected_names: Iterable[str] = ()) -> None:
        explicit = set(protected_names)
        self.values = sorted(
            {
                value
                for name, value in environment.items()
                if (SECRET_NAME.search(name) or name in explicit) and value and len(value) >= 6
            },
            key=len,
            reverse=True,
        )

    def text(self, value: str) -> str:
        for secret in self.values:
            value = value.replace(secret, "[REDACTED]")
        return value

    def value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {str(key): self.value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.value(item) for item in value]
        return value
