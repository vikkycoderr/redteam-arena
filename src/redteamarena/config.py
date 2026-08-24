from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 development fallback
    import tomli as tomllib  # type: ignore[no-redef]

from dotenv import dotenv_values
from pydantic import Field
from pydantic import ValidationError as PydanticValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from .exceptions import ConfigurationError

DEFAULT_CONFIG = """# RedTeamArena stores variable names here, never secret values.
[attack_agent]
provider = "openai_compatible"
model = "qwen/qwen3-30b-a3b-instruct-2507"
api_key_env = "RTA_ATTACK_API_KEY"
base_url_env = "RTA_ATTACK_BASE_URL"
timeout_seconds = 120
temperature = 0.6
max_retries = 2
prompt_version = "v1"

[pipeline]
application_confidence_threshold = 0.55
max_scenarios = 60
max_tool_steps = 8
repair_attempts = 1
inline_document_fallback = true
"""

ENV_EXAMPLE = """# Copy this file to .env. Never commit .env.
# OpenRouter-hosted Attack Agent
RTA_ATTACK_PROVIDER=openai_compatible
# Enter your OpenRouter API key below.
RTA_ATTACK_API_KEY=
RTA_ATTACK_MODEL=qwen/qwen3-30b-a3b-instruct-2507
RTA_ATTACK_BASE_URL=https://openrouter.ai/api/v1
RTA_ATTACK_TIMEOUT_SECONDS=120
"""


class RuntimeOverrides(BaseSettings):
    """Standard non-secret overrides; credentials stay out of model representations."""

    model_config = SettingsConfigDict(extra="ignore", case_sensitive=True)

    attack_provider: Optional[str] = Field(None, validation_alias="RTA_ATTACK_PROVIDER")
    attack_model: Optional[str] = Field(None, validation_alias="RTA_ATTACK_MODEL")
    attack_base_url: Optional[str] = Field(None, validation_alias="RTA_ATTACK_BASE_URL")
    attack_timeout_seconds: Optional[str] = Field(
        None, validation_alias="RTA_ATTACK_TIMEOUT_SECONDS"
    )


@dataclass(frozen=True)
class AttackAgentConfig:
    provider: str = "openai_compatible"
    model: str = ""
    api_key_env: str = "RTA_ATTACK_API_KEY"
    base_url_env: str = "RTA_ATTACK_BASE_URL"
    timeout_seconds: float = 120.0
    temperature: float = 0.6
    max_retries: int = 2
    prompt_version: str = "v1"

    def api_key(self, environment: Dict[str, str]) -> Optional[str]:
        value = environment.get(self.api_key_env)
        return value or None

    def base_url(self, environment: Dict[str, str]) -> Optional[str]:
        value = environment.get(self.base_url_env)
        return value or None

    def missing_fields(self, environment: Dict[str, str]) -> list[str]:
        missing: list[str] = []
        if not self.model:
            missing.append("RTA_ATTACK_MODEL")
        if self.provider not in {"ollama"} and not self.api_key(environment):
            missing.append(self.api_key_env)
        return missing

    def validate_for_generation(self, environment: Dict[str, str]) -> None:
        missing = self.missing_fields(environment)
        if missing:
            joined = ", ".join(sorted(set(missing)))
            raise ConfigurationError(f"Attack Agent configuration is incomplete. Set: {joined}")
        if self.provider not in {"openai_compatible", "anthropic", "ollama"}:
            raise ConfigurationError(f"Unsupported Attack Agent provider: {self.provider}")


@dataclass(frozen=True)
class PipelineConfig:
    application_confidence_threshold: float = 0.55
    max_scenarios: int = 60
    max_tool_steps: int = 8
    repair_attempts: int = 1
    inline_document_fallback: bool = True


@dataclass(frozen=True)
class Settings:
    root: Path
    attack_agent: AttackAgentConfig
    pipeline: PipelineConfig
    environment: Dict[str, str]


def _clean_environment(root: Path) -> Dict[str, str]:
    # dotenv_values reads values only for runtime use. Values are never returned by reports/logging.
    file_values = {
        str(key): str(value)
        for key, value in dotenv_values(root / ".env").items()
        if value is not None
    }
    return {**file_values, **dict(os.environ)}


def load_settings(root: Path) -> Settings:
    root = root.resolve()
    config_path = root / ".redteamarena" / "config.toml"
    raw: Dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    attack = raw.get("attack_agent", {})
    pipeline = raw.get("pipeline", {})
    environment = _clean_environment(root)
    try:
        overrides = RuntimeOverrides.model_validate(environment)
    except PydanticValidationError as exc:
        raise ConfigurationError("Attack Agent environment overrides are invalid") from exc
    provider = overrides.attack_provider or attack.get("provider", "openai_compatible")
    model = overrides.attack_model or attack.get("model", "")
    timeout = overrides.attack_timeout_seconds or attack.get("timeout_seconds", 120)
    try:
        timeout_seconds = float(timeout)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("RTA_ATTACK_TIMEOUT_SECONDS must be numeric") from exc
    return Settings(
        root=root,
        attack_agent=AttackAgentConfig(
            provider=str(provider),
            model=str(model),
            api_key_env=str(attack.get("api_key_env", "RTA_ATTACK_API_KEY")),
            base_url_env=str(attack.get("base_url_env", "RTA_ATTACK_BASE_URL")),
            timeout_seconds=timeout_seconds,
            temperature=float(attack.get("temperature", 0.6)),
            max_retries=int(attack.get("max_retries", 2)),
            prompt_version=str(attack.get("prompt_version", "v1")),
        ),
        pipeline=PipelineConfig(
            application_confidence_threshold=float(
                pipeline.get("application_confidence_threshold", 0.55)
            ),
            max_scenarios=int(pipeline.get("max_scenarios", 60)),
            max_tool_steps=int(pipeline.get("max_tool_steps", 8)),
            repair_attempts=int(pipeline.get("repair_attempts", 1)),
            inline_document_fallback=bool(pipeline.get("inline_document_fallback", True)),
        ),
        environment=environment,
    )
