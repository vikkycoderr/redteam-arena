from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionLimits:
    max_tool_steps: int = 8
    timeout_seconds: float = 120.0
