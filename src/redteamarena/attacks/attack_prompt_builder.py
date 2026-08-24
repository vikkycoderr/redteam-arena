from __future__ import annotations

import json
from importlib.resources import files
from typing import Any, Dict

from ..models import APPROVED_CATEGORIES, AgentDetection, ProjectProfile, ScenarioSet


class AttackPromptBuilder:
    def __init__(self, version: str = "v1") -> None:
        self.version = version

    def system_prompt(self) -> str:
        resource = files("redteamarena.prompts").joinpath(f"attack_agent_{self.version}.txt")
        return resource.read_text(encoding="utf-8")

    def package(self, profile: ProjectProfile, agent: AgentDetection) -> Dict[str, Any]:
        return {
            "project_profile": profile.model_dump(mode="json"),
            "target_agent": agent.model_dump(mode="json", exclude={"confidence": {"evidence"}}),
            "available_tools": [tool.provider_schema() for tool in agent.tools],
            "protected_resources": agent.protected_resources,
            "target_capabilities": agent.capabilities.model_dump(mode="json"),
            "approved_categories": sorted(APPROVED_CATEGORIES),
            "scenario_generation_policy": {
                "count": "adaptive",
                "basis": (
                    "Generate scenarios only for concrete attack surfaces evidenced by the "
                    "supplied project profile and target agent fields."
                ),
                "difficulty_quota": "none",
            },
            "output_json_schema": ScenarioSet.model_json_schema(),
        }

    def user_prompt(self, profile: ProjectProfile, agent: AgentDetection) -> str:
        return json.dumps(self.package(profile, agent), indent=2, sort_keys=True)
