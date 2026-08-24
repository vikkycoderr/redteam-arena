from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx

from ..exceptions import ConfigurationError, ProviderError, ProviderTimeout
from .base import AdapterRequest, BaseAdapter


@dataclass(frozen=True)
class StaticCallRecipe:
    url: str
    model_field: str = "model"
    messages_field: str = "messages"
    tools_field: str = "tools"
    authorization_env: Optional[str] = None

    def validate(self) -> None:
        parsed = urlparse(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ConfigurationError("Restricted generic adapter requires a static HTTP(S) URL")
        if parsed.username or parsed.password:
            raise ConfigurationError("Restricted generic adapter URL must not contain credentials")
        if any(
            token in parsed.query.lower() for token in ("key=", "token=", "secret=", "password=")
        ):
            raise ConfigurationError("Restricted generic adapter URL must not contain credentials")
        for field in (self.model_field, self.messages_field, self.tools_field):
            if not field.replace("_", "").isalnum():
                raise ConfigurationError("Restricted generic adapter fields must be simple names")


class RestrictedGenericAdapter(BaseAdapter):
    provider = "restricted_generic"

    def __init__(
        self,
        recipe: StaticCallRecipe,
        environment: Dict[str, str],
        client: Optional[httpx.Client] = None,
    ) -> None:
        recipe.validate()
        self.recipe = recipe
        self._environment = environment
        self.client = client or httpx.Client()

    def complete(self, request: AdapterRequest) -> Any:
        payload = {
            self.recipe.model_field: request.model,
            self.recipe.messages_field: request.messages,
        }
        if request.tools:
            payload[self.recipe.tools_field] = request.tools
        headers: Dict[str, str] = {}
        if self.recipe.authorization_env:
            value = self._environment.get(self.recipe.authorization_env)
            if not value:
                raise ConfigurationError(
                    f"Missing target credential environment variable: {self.recipe.authorization_env}"
                )
            headers["Authorization"] = f"Bearer {value}"
        try:
            response = self.client.post(
                self.recipe.url, headers=headers, json=payload, timeout=request.timeout_seconds
            )
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as exc:
            raise ProviderTimeout("Restricted generic provider timed out") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError("Restricted generic provider request failed") from exc
