from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..exceptions import ValidationError

SAFE_COMPONENT = re.compile(r"^[a-zA-Z0-9_.-]+$")


def safe_component(value: str) -> str:
    if not SAFE_COMPONENT.fullmatch(value) or value in {".", ".."}:
        raise ValidationError(f"Unsafe path component: {value!r}")
    return value


@dataclass(frozen=True)
class PathManager:
    root: Path

    @property
    def state(self) -> Path:
        return self.root / ".redteamarena"

    @property
    def scan(self) -> Path:
        return self.state / "scan"

    @property
    def agents(self) -> Path:
        return self.state / "agents"

    @property
    def reports(self) -> Path:
        return self.state / "reports"

    @property
    def cache(self) -> Path:
        return self.state / "cache"

    def agent_root(self, agent_id: str, fingerprint: str) -> Path:
        return self.agents / safe_component(agent_id) / safe_component(fingerprint)

    def scenario_runtime(self, agent_id: str, fingerprint: str, scenario_id: str) -> Path:
        return self.agent_root(agent_id, fingerprint) / "runtime" / safe_component(scenario_id)

    def ensure_base(self) -> None:
        for path in (self.scan, self.agents, self.reports, self.cache):
            path.mkdir(parents=True, exist_ok=True)
