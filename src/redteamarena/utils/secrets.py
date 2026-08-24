import re
from typing import Any

ASSIGNMENT = re.compile(
    r"(?i)\b(api[ _-]?key|secret|access[ _-]?token|password|credential)\s*[:=]\s*([^\s,;]+)"
)
TOKEN_SHAPES = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{12,}\b"),
)


def redact_potential_secrets(value: str) -> str:
    """Conservatively remove likely literal credentials from scanned prose."""
    value = ASSIGNMENT.sub(lambda match: f"{match.group(1)}=[REDACTED]", value)
    for pattern in TOKEN_SHAPES:
        value = pattern.sub("[REDACTED]", value)
    return value


def sanitize_scanned_value(value: Any) -> Any:
    """Redact secret-shaped literals before any static-analysis value is persisted."""
    if isinstance(value, str):
        return redact_potential_secrets(value)
    if isinstance(value, list):
        return [sanitize_scanned_value(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_scanned_value(item) for item in value]
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if re.search(r"(?i)(api[_-]?key|secret|token|password|credential)", name):
                sanitized[name] = "[REDACTED]"
            else:
                sanitized[name] = sanitize_scanned_value(item)
        return sanitized
    return value
