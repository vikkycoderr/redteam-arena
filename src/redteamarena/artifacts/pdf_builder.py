from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter  # type: ignore[import-untyped]
from reportlab.lib.styles import getSampleStyleSheet  # type: ignore[import-untyped]
from reportlab.platypus import (  # type: ignore[import-untyped]
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

from ..models import ArtifactSpec


class PdfArtifactBuilder:
    def build(self, spec: ArtifactSpec, output_dir: Path) -> Path:
        path = output_dir / spec.filename
        document = SimpleDocTemplate(str(path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        for paragraph in spec.content.split("\n"):
            story.append(Paragraph(self._escape(paragraph) or "&nbsp;", styles["BodyText"]))
            story.append(Spacer(1, 8))
        document.build(story)
        return path

    @staticmethod
    def _escape(value: str) -> str:
        return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
