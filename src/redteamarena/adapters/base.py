from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AdapterRequest:
    model: str
    messages: List[Dict[str, Any]]
    tools: List[Dict[str, Any]] = field(default_factory=list)
    temperature: float = 0.0
    timeout_seconds: float = 120.0
    artifacts: List[Dict[str, Any]] = field(default_factory=list)


class BaseAdapter(ABC):
    provider: str

    @abstractmethod
    def complete(self, request: AdapterRequest) -> Any:
        """Return the provider response without logging credentials or request headers."""


class UnavailableAdapter(BaseAdapter):
    def __init__(self, reason: str, provider: str = "unavailable") -> None:
        self.reason = reason
        self.provider = provider

    def complete(self, request: AdapterRequest) -> Any:
        from ..exceptions import ConfigurationError

        raise ConfigurationError(self.reason)


class TextGenerator:
    def __init__(
        self, adapter: BaseAdapter, model: str, timeout_seconds: float, temperature: float
    ) -> None:
        self.adapter = adapter
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        response = self.adapter.complete(
            AdapterRequest(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                timeout_seconds=self.timeout_seconds,
            )
        )
        return self._text(response)

    @staticmethod
    def _text(response: Any) -> str:
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            response_text = response.get("response")
            if isinstance(response_text, str):
                return response_text
            choices = response.get("choices") or []
            if choices:
                message = choices[0].get("message", {})
                message_text = message.get("content")
                if isinstance(message_text, str):
                    return message_text
            content = response.get("content") or []
            return "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return ""
