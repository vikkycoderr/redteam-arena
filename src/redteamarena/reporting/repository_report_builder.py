from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, List

from ..models import AgentDetection, AgentReport, OverallResult, RepositoryReport


class RepositoryReportBuilder:
    def build(
        self,
        root: Path,
        reports: List[AgentReport],
        detections: Iterable[AgentDetection],
    ) -> RepositoryReport:
        counts: Counter[str] = Counter()
        patterns: Counter[str] = Counter()
        risks: dict[str, int] = {}
        for report in reports:
            counts.update(result.overall_result.value for result in report.scenario_results)
            patterns.update(report.repeated_failure_patterns)
            risks[report.agent_id] = sum(
                result.overall_result == OverallResult.SECURITY_FAIL
                for result in report.scenario_results
            )
        untested = []
        for detection in detections:
            if detection.eligible:
                continue
            untested.append(
                {
                    "agent_id": detection.agent_id,
                    "provider": detection.provider,
                    "model": detection.model,
                    "confidence_score": detection.confidence.score,
                    "confidence_level": getattr(
                        detection.confidence.level, "value", str(detection.confidence.level)
                    ),
                    "detection_evidence": [
                        item.model_dump(mode="json") for item in detection.confidence.evidence
                    ],
                    "missing_evidence": detection.confidence.missing_evidence,
                    "reason_not_tested": "LOW/UNRESOLVED detections are reported but not automatically tested",
                    "call_sites": detection.call_sites,
                }
            )
        highest = (
            max(risks, key=lambda agent_id: risks[agent_id])
            if risks and max(risks.values()) > 0
            else None
        )
        total_scenarios = sum(len(report.scenario_results) for report in reports)
        return RepositoryReport(
            repository_root=str(root),
            totals={"agents_tested": len(reports), "scenarios": total_scenarios, **dict(counts)},
            highest_risk_agent=highest,
            repeated_failure_patterns=sorted(patterns),
            agent_reports=reports,
            untested_detections=untested,
        )
