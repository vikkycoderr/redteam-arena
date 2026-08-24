from __future__ import annotations

import json
from typing import Any, Dict, List

from ..models import ModelResult


class ConversationManager:
    def __init__(self, initial_messages: List[Dict[str, Any]]) -> None:
        self.messages = list(initial_messages)

    def add_result(self, result: ModelResult, tool_results: List[Dict[str, Any]]) -> None:
        self.messages.append(
            {
                "role": "assistant",
                "content": result.text,
                "tool_calls": [
                    {
                        "id": action.call_id,
                        "name": action.tool,
                        "arguments": action.arguments,
                    }
                    for action in result.proposed_actions
                ],
            }
        )
        for action, tool_result in zip(result.proposed_actions, tool_results):
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": action.call_id,
                    "name": action.tool,
                    "content": json.dumps(tool_result, sort_keys=True),
                }
            )

    def enable_structured_tool_fallback(self, tools: List[Dict[str, Any]]) -> None:
        self.messages.append(
            {
                "role": "system",
                "content": (
                    "Native tool calls are unavailable on this model route. To call a tool, "
                    "reply with exactly one JSON object and no prose: "
                    '{"action":"tool_name","arguments":{}}. '
                    "Use only these synthetic tool schemas: "
                    + json.dumps(tools, sort_keys=True)
                    + " When no tool is needed, return the final answer as plain text."
                ),
            }
        )

    def add_structured_result(
        self, result: ModelResult, tool_results: List[Dict[str, Any]]
    ) -> None:
        content = result.text or json.dumps(
            {
                "action": result.proposed_actions[0].tool,
                "arguments": result.proposed_actions[0].arguments,
            },
            sort_keys=True,
        )
        self.messages.append({"role": "assistant", "content": content})
        for action, tool_result in zip(result.proposed_actions, tool_results):
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Synthetic tool result for {action.tool}: "
                        + json.dumps(tool_result, sort_keys=True)
                    ),
                }
            )
