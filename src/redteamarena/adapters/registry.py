from __future__ import annotations

from typing import Dict, Optional

from ..config import AttackAgentConfig
from ..exceptions import ConfigurationError
from ..models import AgentDetection
from .anthropic import AnthropicAdapter
from .base import BaseAdapter
from .ollama import OllamaAdapter
from .openai_compatible import OpenAICompatibleAdapter
from .restricted_generic import RestrictedGenericAdapter, StaticCallRecipe
from .transformers import TransformersAdapter


class AdapterRegistry:
    def for_attack(self, config: AttackAgentConfig, environment: Dict[str, str]) -> BaseAdapter:
        config.validate_for_generation(environment)
        if config.provider == "openai_compatible":
            return OpenAICompatibleAdapter(
                api_key=config.api_key(environment) or "",
                base_url=config.base_url(environment),
            )
        if config.provider == "anthropic":
            return AnthropicAdapter(
                api_key=config.api_key(environment) or "",
                base_url=config.base_url(environment),
            )
        if config.provider == "ollama":
            return OllamaAdapter(base_url=config.base_url(environment))
        raise ConfigurationError(f"Unsupported Attack Agent provider: {config.provider}")

    def for_target(self, agent: AgentDetection, environment: Dict[str, str]) -> BaseAdapter:
        missing = [name for name in agent.credential_env_vars if not environment.get(name)]
        if missing and agent.provider not in {"ollama", "transformers"}:
            raise ConfigurationError(
                "Missing target credential environment variable(s): " + ", ".join(missing)
            )
        if not agent.model:
            required = agent.configuration_env_vars or ["the target model variable"]
            raise ConfigurationError("Target model is unresolved; configure " + ", ".join(required))
        if agent.provider == "openai_compatible" or agent.protocol == "openai_compatible":
            key_name = next(iter(agent.credential_env_vars), "OPENAI_API_KEY")
            base_url = self._first(
                environment,
                ("OPENAI_BASE_URL", "AZURE_OPENAI_ENDPOINT", "OPENROUTER_BASE_URL"),
            )
            return OpenAICompatibleAdapter(environment.get(key_name, ""), base_url)
        if agent.provider == "anthropic":
            key_name = next(iter(agent.credential_env_vars), "ANTHROPIC_API_KEY")
            return AnthropicAdapter(
                environment.get(key_name, ""), environment.get("ANTHROPIC_BASE_URL")
            )
        if agent.provider == "ollama":
            return OllamaAdapter(environment.get("OLLAMA_HOST"))
        if agent.provider == "transformers":
            return TransformersAdapter()
        if agent.static_call_recipe:
            try:
                recipe = StaticCallRecipe(**agent.static_call_recipe)
            except TypeError as exc:
                raise ConfigurationError("Invalid restricted static call recipe") from exc
            return RestrictedGenericAdapter(recipe, environment)
        raise ConfigurationError(
            f"Unsupported target provider/protocol for {agent.agent_id}: {agent.provider}"
        )

    @staticmethod
    def _first(environment: Dict[str, str], names: tuple[str, ...]) -> Optional[str]:
        return next((environment[name] for name in names if environment.get(name)), None)
