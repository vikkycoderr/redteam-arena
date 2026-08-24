from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from ..models import ModelResult, ProposedAction


class ResponseNormalizer:
    FENCED_OBJECT = re.compile(
        r"\A```(?:json)?[ \t]*(?:\r?\n)?(?P<body>.*?)(?:\r?\n)?```[ \t]*\Z",
        re.IGNORECASE | re.DOTALL,
    )

    def normalize(self, raw: Any, provider: str = "unknown") -> ModelResult:
        if isinstance(raw, ModelResult):
            return raw
        if isinstance(raw, str):
            return self._from_text(raw)
        if not isinstance(raw, dict):
            return ModelResult(text="", format_status="unsupported", raw=raw)
        if "choices" in raw:
            return self._openai(raw)
        if isinstance(raw.get("content"), list):
            return self._anthropic(raw)
        if "message" in raw:
            return self._ollama(raw)
        if isinstance(raw.get("generated_text"), str):
            return self._from_text(raw["generated_text"])
        if isinstance(raw.get("response"), str):
            return self._from_text(raw["response"])
        return ModelResult(text="", format_status="unsupported", raw=raw)

    def _openai(self, raw: Dict[str, Any]) -> ModelResult:
        choices = raw.get("choices") or []
        if not choices:
            return ModelResult(text="", format_status="invalid", raw=raw)
        choice = choices[0]
        message = choice.get("message") or {}
        actions: List[ProposedAction] = []
        status = "valid"
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            try:
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                actions.append(
                    ProposedAction(
                        tool=str(function["name"]), arguments=arguments, call_id=call.get("id")
                    )
                )
            except (KeyError, TypeError, json.JSONDecodeError):
                status = "partially_parsed"
        text = message.get("content") or ""
        if not actions:
            fallback, structured_status = self._structured_action(text)
            if fallback:
                actions.append(fallback)
            if structured_status == "invalid":
                status = "invalid"
        return ModelResult(
            text=str(text),
            proposed_actions=actions,
            format_status=status,
            finish_reason=choice.get("finish_reason"),
            raw=raw,
        )

    def _anthropic(self, raw: Dict[str, Any]) -> ModelResult:
        texts: List[str] = []
        actions: List[ProposedAction] = []
        status = "valid"
        for block in raw.get("content") or []:
            if not isinstance(block, dict):
                status = "partially_parsed"
                continue
            if block.get("type") == "text":
                texts.append(str(block.get("text", "")))
            elif block.get("type") == "tool_use":
                try:
                    actions.append(
                        ProposedAction(
                            tool=str(block["name"]),
                            arguments=block.get("input") or {},
                            call_id=block.get("id"),
                        )
                    )
                except (KeyError, TypeError):
                    status = "partially_parsed"
        text = "\n".join(texts)
        if not actions:
            fallback, structured_status = self._structured_action(text)
            if fallback:
                actions.append(fallback)
            if structured_status == "invalid":
                status = "invalid"
        return ModelResult(
            text=text,
            proposed_actions=actions,
            format_status=status,
            finish_reason=raw.get("stop_reason"),
            raw=raw,
        )

    def _ollama(self, raw: Dict[str, Any]) -> ModelResult:
        message = raw.get("message") or {}
        actions: List[ProposedAction] = []
        status = "valid"
        for call in message.get("tool_calls") or []:
            function = call.get("function") or call
            try:
                arguments = function.get("arguments") or {}
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                actions.append(ProposedAction(tool=str(function["name"]), arguments=arguments))
            except (KeyError, TypeError, json.JSONDecodeError):
                status = "partially_parsed"
        text = str(message.get("content") or "")
        if not actions:
            fallback, structured_status = self._structured_action(text)
            if fallback:
                actions.append(fallback)
            if structured_status == "invalid":
                status = "invalid"
        return ModelResult(
            text=text,
            proposed_actions=actions,
            format_status=status,
            finish_reason=raw.get("done_reason"),
            raw=raw,
        )

    def _from_text(self, text: str) -> ModelResult:
        action, structured_status = self._structured_action(text)
        status = (
            "invalid"
            if structured_status == "invalid" or not text.strip()
            else "valid"
        )
        return ModelResult(
            text=text,
            proposed_actions=[action] if action else [],
            format_status=status,
            raw=text,
        )

    @classmethod
    def _structured_action(
        cls, text: str
    ) -> tuple[ProposedAction | None, str]:
        """Parse one whole bare/fenced action object; never infer an action from prose."""
        stripped = text.strip()
        if not stripped:
            return None, "not_structured"
        candidate = stripped
        if stripped.startswith("```"):
            match = cls.FENCED_OBJECT.fullmatch(stripped)
            if not match:
                return None, "invalid"
            candidate = match.group("body").strip()
        elif not stripped.startswith("{"):
            # A JSON action embedded in prose is an action attempt, but it is
            # deliberately rejected rather than inferred and executed.
            if "```" in stripped or (
                re.search(r'"action"\s*:', stripped)
                and re.search(r'"arguments"\s*:', stripped)
            ):
                return None, "invalid"
            return None, "not_structured"
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, AttributeError):
            return None, "invalid"
        if not isinstance(parsed, dict) or set(parsed) != {"action", "arguments"}:
            return None, "invalid"
        action = parsed.get("action")
        arguments = parsed.get("arguments", {})
        if not isinstance(action, str) or not action.strip() or not isinstance(arguments, dict):
            return None, "invalid"
        if action.casefold() == "none":
            return None, "valid"
        return ProposedAction(tool=action, arguments=arguments), "valid"
