from __future__ import annotations

from pathlib import Path

from ..models import ScenarioSet


class ScenarioLoader:
    @staticmethod
    def load(path: Path) -> ScenarioSet:
        return ScenarioSet.model_validate_json(path.read_text(encoding="utf-8"))
