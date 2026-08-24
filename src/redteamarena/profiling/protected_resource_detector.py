from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

from ..models import Confidence, Evidence
from ..scanner.confidence_scorer import ConfidenceScorer


class ProtectedResourceDetector:
    SENSITIVE = re.compile(
        r"(?i)(\.env|secret|credential|api[_-]?key|token|private[_-]?key|password|auth|config|\.git)"
    )

    def detect(self, files: Iterable[Path], root: Path) -> tuple[List[str], Confidence]:
        resources = {".env", ".git/", "credentials/", "secrets/"}
        evidence: List[Evidence] = []
        for path in files:
            relative = path.relative_to(root).as_posix()
            if self.SENSITIVE.search(relative):
                resources.add(relative)
                evidence.append(
                    Evidence(
                        kind="sensitive_filename",
                        path=relative,
                        detail="Filename indicates a protected resource",
                        points=1,
                    )
                )
        confidence = ConfidenceScorer().score(
            evidence
            or [
                Evidence(
                    kind="default_protection_policy",
                    path=".",
                    detail="Default secret and VCS resources are protected",
                    points=4,
                )
            ]
        )
        return sorted(resources), confidence
