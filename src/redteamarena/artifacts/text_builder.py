from __future__ import annotations

from pathlib import Path

from ..models import ArtifactSpec


class TextArtifactBuilder:
    def build(self, spec: ArtifactSpec, output_dir: Path) -> Path:
        path = output_dir / spec.filename
        path.write_text(spec.content, encoding="utf-8")
        return path
