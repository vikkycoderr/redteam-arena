from __future__ import annotations

import copy
from typing import Any, Optional

import httpx

from ..exceptions import ProviderError, ProviderTimeout
from .base import AdapterRequest, BaseAdapter


class AnthropicAdapter(BaseAdapter):
    provider = "anthropic"

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self._api_key = api_key
        self.base_url = (base_url or "https://api.anthropic.com/v1").rstrip("/")
        self.client = client or httpx.Client()

    def complete(self, request: AdapterRequest) -> Any:
        system_parts = [
            message.get("content", "")
            for message in request.messages
            if message.get("role") == "system"
        ]
        messages = copy.deepcopy(
            [message for message in request.messages if message.get("role") != "system"]
        )
        messages = self._normalize_tool_messages(messages)
        if request.artifacts:
            self._attach_artifacts(messages, request.artifacts)
        payload: dict[str, Any] = {
            "model": request.model,
            "max_tokens": 4096,
            "messages": messages,
            "temperature": request.temperature,
        }
        if system_parts:
            payload["system"] = "\n\n".join(str(part) for part in system_parts)
        if request.tools:
            payload["tools"] = [
                {
                    "name": item["function"]["name"],
                    "description": item["function"]["description"],
                    "input_schema": item["function"]["parameters"],
                }
                for item in request.tools
            ]
        try:
            response = self.client.post(
                f"{self.base_url}/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                },
                json=payload,
                timeout=request.timeout_seconds,
            )
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as exc:
            raise ProviderTimeout("Anthropic provider timed out") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError("Anthropic provider request failed") from exc

    @staticmethod
    def _attach_artifacts(messages: list[dict[str, Any]], artifacts: list[dict[str, Any]]) -> None:
        user = next(
            (message for message in reversed(messages) if message.get("role") == "user"),
            None,
        )
        if user is None:
            return
        content = user.get("content", "")
        blocks: list[dict[str, Any]] = (
            [{"type": "text", "text": content}] if isinstance(content, str) else list(content)
        )
        for artifact in artifacts:
            source = {
                "type": "base64",
                "media_type": artifact["mime_type"],
                "data": artifact["data_base64"],
            }
            if artifact.get("type") == "image":
                blocks.append({"type": "image", "source": source})
            elif artifact.get("type") == "pdf":
                blocks.append({"type": "document", "source": source})
        user["content"] = blocks

    @staticmethod
    def _normalize_tool_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for message in messages:
            if message.get("role") == "assistant" and isinstance(message.get("tool_calls"), list):
                blocks: list[dict[str, Any]] = []
                if message.get("content"):
                    blocks.append({"type": "text", "text": message["content"]})
                blocks.extend(
                    {
                        "type": "tool_use",
                        "id": call.get("id"),
                        "name": call.get("name"),
                        "input": call.get("arguments", {}),
                    }
                    for call in message["tool_calls"]
                )
                normalized.append({"role": "assistant", "content": blocks})
            elif message.get("role") == "tool":
                normalized.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.get("tool_call_id"),
                                "content": message.get("content", ""),
                            }
                        ],
                    }
                )
            else:
                normalized.append(message)
        return normalized
