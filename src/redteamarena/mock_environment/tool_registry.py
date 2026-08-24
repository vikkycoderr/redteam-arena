from __future__ import annotations

from typing import Any, Dict

from ..models import ToolSpec
from .mock_handlers import Handler, handler_for


class ToolRegistry:
    def __init__(self, tools: list[ToolSpec]) -> None:
        self.specs: Dict[str, ToolSpec] = {tool.name: tool for tool in tools}
        self.handlers: Dict[str, Handler] = {name: handler_for(name) for name in self.specs}

    def schema_list(self) -> list[dict[str, Any]]:
        return [tool.provider_schema() for tool in self.specs.values()]
