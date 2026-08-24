from __future__ import annotations

import shutil

from ..models import AgentDetection
from ..storage import JsonStore, PathManager
from .environment import MockEnvironment


class EnvironmentCloner:
    def __init__(self, paths: PathManager) -> None:
        self.paths = paths

    def clone(self, agent: AgentDetection, scenario_id: str) -> MockEnvironment:
        agent_root = self.paths.agent_root(agent.agent_id, agent.configuration_fingerprint)
        template = agent_root / "mock_environment" / "workspace_template"
        runtime = self.paths.scenario_runtime(
            agent.agent_id, agent.configuration_fingerprint, scenario_id
        )
        workspace = runtime / "workspace"
        symlinks = [path for path in template.rglob("*") if path.is_symlink()]
        if symlinks:
            raise ValueError("Refusing to clone a mock template containing symbolic links")
        if runtime.exists():
            resolved = runtime.resolve()
            state = self.paths.state.resolve()
            if state not in resolved.parents:
                raise ValueError("Refusing to reset runtime outside .redteamarena")
            shutil.rmtree(runtime)
        shutil.copytree(template, workspace)
        initial_state = JsonStore.read(agent_root / "mock_environment" / "initial_state.json")
        return MockEnvironment(
            workspace=workspace,
            tools={tool.name: tool for tool in agent.tools},
            protected_resources=list(agent.protected_resources),
            state=initial_state,
        )
