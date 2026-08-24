from __future__ import annotations

from typing import Any, Callable, Optional

from ..exceptions import ConfigurationError, ProviderError
from .base import AdapterRequest, BaseAdapter


class TransformersAdapter(BaseAdapter):
    provider = "transformers"

    def __init__(self, pipeline_factory: Optional[Callable[..., Any]] = None) -> None:
        self._pipeline_factory = pipeline_factory
        self._pipelines: dict[str, Any] = {}

    def complete(self, request: AdapterRequest) -> Any:
        if request.tools:
            raise ProviderError("Transformers adapter does not support native tools")
        factory = self._pipeline_factory
        if factory is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise ConfigurationError(
                    "Install RedTeamArena with the 'transformers' extra for local targets"
                ) from exc
            factory = pipeline
        try:
            generator = self._pipelines.setdefault(
                request.model, factory("text-generation", model=request.model)
            )
            prompt = "\n".join(
                f"{message.get('role', 'user')}: {message.get('content', '')}"
                for message in request.messages
            )
            output = generator(prompt, max_new_tokens=1024)
            return {"generated_text": output[0]["generated_text"]}
        except ConfigurationError:
            raise
        except Exception as exc:
            raise ProviderError("Transformers generation failed") from exc
