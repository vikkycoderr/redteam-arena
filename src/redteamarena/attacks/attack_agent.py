from __future__ import annotations

from ..adapters import BaseAdapter, TextGenerator
from ..config import AttackAgentConfig
from ..exceptions import ProviderError


class AttackAgent:
    def __init__(self, adapter: BaseAdapter, config: AttackAgentConfig) -> None:
        self.generator = TextGenerator(
            adapter,
            model=config.model,
            timeout_seconds=config.timeout_seconds,
            temperature=config.temperature,
        )
        self.max_retries = config.max_retries

    def generate(self, system_prompt: str, package: str) -> str:
        last_error: ProviderError | None = None
        for _ in range(self.max_retries + 1):
            try:
                return self.generator.generate_text(system_prompt, package)
            except ProviderError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error
