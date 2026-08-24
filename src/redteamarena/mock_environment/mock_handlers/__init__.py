from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict

from ...exceptions import ValidationError
from ..environment import MockEnvironment

Handler = Callable[[MockEnvironment, Dict[str, Any]], Any]


def _workspace_path(environment: MockEnvironment, raw: str) -> Path:
    candidate = (environment.workspace / raw.lstrip("/")).resolve()
    workspace = environment.workspace.resolve()
    if candidate != workspace and workspace not in candidate.parents:
        raise ValidationError("Mock path escapes the isolated workspace")
    return candidate


def mock_read_file(environment: MockEnvironment, arguments: Dict[str, Any]) -> Dict[str, Any]:
    path = _workspace_path(environment, str(arguments.get("path", "")))
    if not path.is_file():
        return {"status": "not_found", "path": path.name}
    return {"status": "ok", "content": path.read_text(encoding="utf-8", errors="replace")}


def mock_write_file(environment: MockEnvironment, arguments: Dict[str, Any]) -> Dict[str, Any]:
    path = _workspace_path(environment, str(arguments.get("path", "")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(arguments.get("content", "")), encoding="utf-8")
    return {"status": "simulated", "path": path.relative_to(environment.workspace).as_posix()}


def mock_operation(environment: MockEnvironment, arguments: Dict[str, Any]) -> Dict[str, Any]:
    history = environment.state.setdefault("operations", [])
    history.append(arguments)
    return {"status": "simulated", "accepted_arguments": arguments}


def handler_for(tool_name: str) -> Handler:
    lower = tool_name.lower()
    if lower in {"read_file", "get_file", "load_file"}:
        return mock_read_file
    if lower in {"write_file", "save_file", "update_file"}:
        return mock_write_file
    return mock_operation
