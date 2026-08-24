from __future__ import annotations

import copy
from typing import Any, Optional

import httpx

from ..exceptions import ProviderError, ProviderTimeout
from .base import AdapterRequest, BaseAdapter


class OllamaAdapter(BaseAdapter):
    provider = "ollama"

    def __init__(
        self, base_url: Optional[str] = None, client: Optional[httpx.Client] = None
    ) -> None:
        self.base_url = (base_url or "http://localhost:11434").rstrip("/")
        self.client = client or httpx.Client()

    def complete(self, request: AdapterRequest) -> Any:
        messages = copy.deepcopy(request.messages)
        self._normalize_tool_messages(messages)
        images = [
            artifact["data_base64"]
            for artifact in request.artifacts
            if artifact.get("type") == "image"
        ]
        if images:
            user = next(
                (message for message in reversed(messages) if message.get("role") == "user"),
                None,
            )
            if user is not None:
                user["images"] = images
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": request.temperature},
        }
        if request.tools:
            payload["tools"] = request.tools
        try:
            response = self.client.post(
                f"{self.base_url}/api/chat", json=payload, timeout=request.timeout_seconds
            )
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as exc:
            raise ProviderTimeout("Ollama provider timed out") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError("Ollama provider request failed") from exc

    @staticmethod
    def _normalize_tool_messages(messages: list[dict[str, Any]]) -> None:
        for message in messages:
            calls = message.get("tool_calls")
            if message.get("role") == "assistant" and isinstance(calls, list):
                message["tool_calls"] = [
                    {
                        "function": {
                            "name": call.get("name"),
                            "arguments": call.get("arguments", {}),
                        }
                    }
                    for call in calls
                ]
            elif message.get("role") == "tool":
                message["tool_name"] = message.pop("name", None)
