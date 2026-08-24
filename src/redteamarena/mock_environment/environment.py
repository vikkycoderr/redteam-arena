from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from ..models import ActionLogEntry, ToolSpec


@dataclass
class MockEnvironment:
    workspace: Path
    tools: Dict[str, ToolSpec]
    protected_resources: List[str]
    state: Dict[str, Any] = field(default_factory=dict)
    action_log: List[ActionLogEntry] = field(default_factory=list)
