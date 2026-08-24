from __future__ import annotations

from ..models import FormatResult, ModelResult


class FormatChecker:
    MAP = {
        "valid": FormatResult.VALID,
        "invalid": FormatResult.INVALID,
        "partially_parsed": FormatResult.PARTIALLY_PARSED,
        "unsupported": FormatResult.UNSUPPORTED,
    }

    def evaluate(self, result: ModelResult) -> FormatResult:
        return self.MAP.get(result.format_status, FormatResult.INVALID)
