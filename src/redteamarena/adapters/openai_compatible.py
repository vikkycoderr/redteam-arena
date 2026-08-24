from __future__ import annotations

import copy
import json
from typing import Any, Optional

import httpx

from ..exceptions import ProviderError, ProviderTimeout
from ..utils.secrets import redact_potential_secrets
from .base import AdapterRequest, BaseAdapter


class OpenAICompatibleAdapter(BaseAdapter):
    provider = "openai_compatible"

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self._api_key = api_key
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.client = client or httpx.Client()

    def complete(self, request: AdapterRequest) -> Any:
        messages = copy.deepcopy(request.messages)
        self._normalize_tool_messages(messages)
        if request.artifacts:
            self._attach_artifacts(messages, request.artifacts)
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
        }
        if request.tools:
            payload["tools"] = request.tools
        try:
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
                timeout=request.timeout_seconds,
            )
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as exc:
            raise ProviderTimeout("OpenAI-compatible provider timed out") from exc
        except httpx.HTTPStatusError as exc:
            detail = self._safe_error_detail(exc.response)
            suffix = f": {detail}" if detail else ""
            raise ProviderError(
                f"OpenAI-compatible provider returned HTTP {exc.response.status_code}{suffix}"
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError("OpenAI-compatible provider request failed") from exc

    @staticmethod
    def _safe_error_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return ""
        error = payload.get("error") if isinstance(payload, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        if not isinstance(message, str):
            return ""
        return redact_potential_secrets(message).replace("\n", " ")[:500]

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
            data_url = f"data:{artifact['mime_type']};base64,{artifact['data_base64']}"
            if artifact.get("type") == "image":
                blocks.append({"type": "image_url", "image_url": {"url": data_url}})
            elif artifact.get("type") == "pdf":
                blocks.append(
                    {
                        "type": "file",
                        "file": {
                            "filename": artifact["filename"],
                            "file_data": data_url,
                        },
                    }
                )
        user["content"] = blocks

    @staticmethod
    def _normalize_tool_messages(messages: list[dict[str, Any]]) -> None:
        for message in messages:
            calls = message.get("tool_calls")
            if message.get("role") == "assistant" and isinstance(calls, list):
                message["tool_calls"] = [
                    {
                        "id": call.get("id"),
                        "type": "function",
                        "function": {
                            "name": call.get("name"),
                            "arguments": json.dumps(call.get("arguments", {}), sort_keys=True),
                        },
                    }
                    for call in calls
                ]
            if message.get("role") == "tool":
                message.pop("name", None)
