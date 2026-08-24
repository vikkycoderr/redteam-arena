from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Protocol

from ..exceptions import ValidationError
from ..models import ArtifactSpec
from .image_builder import ImageArtifactBuilder
from .pdf_builder import PdfArtifactBuilder
from .text_builder import TextArtifactBuilder


@dataclass(frozen=True)
class BuiltArtifact:
    spec: ArtifactSpec
    path: Path


class PhysicalArtifactBuilder(Protocol):
    def build(self, spec: ArtifactSpec, output_dir: Path) -> Path: ...


class ArtifactBuilder:
    def __init__(self) -> None:
        self.builders: Dict[str, PhysicalArtifactBuilder] = {
            "text": TextArtifactBuilder(),
            "pdf": PdfArtifactBuilder(),
            "image": ImageArtifactBuilder(),
        }

    def build(self, spec: ArtifactSpec, output_dir: Path) -> BuiltArtifact:
        output_dir.mkdir(parents=True, exist_ok=True)
        builder = self.builders.get(spec.type)
        if builder is None:
            raise ValidationError(f"Unsupported artifact type: {spec.type}")
        return BuiltArtifact(spec=spec, path=builder.build(spec, output_dir))
