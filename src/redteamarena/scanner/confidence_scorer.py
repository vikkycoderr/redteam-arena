from __future__ import annotations

from typing import Iterable, List

from ..models import Confidence, ConfidenceLevel, Evidence


class ConfidenceScorer:
    """Evidence scores are additive and retain their provenance for reports."""

    @staticmethod
    def level(score: float) -> ConfidenceLevel:
        if score >= 8:
            return ConfidenceLevel.HIGH
        if score >= 4:
            return ConfidenceLevel.MEDIUM
        if score >= 1:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.UNRESOLVED

    def score(
        self, evidence: Iterable[Evidence], missing_evidence: Iterable[str] = ()
    ) -> Confidence:
        items: List[Evidence] = list(evidence)
        total = sum(item.points for item in items)
        return Confidence(
            score=total,
            level=self.level(total),
            evidence=items,
            missing_evidence=list(dict.fromkeys(missing_evidence)),
        )
