from __future__ import annotations

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw

from ..models import ArtifactSpec


class ImageArtifactBuilder:
    def build(self, spec: ArtifactSpec, output_dir: Path) -> Path:
        path = output_dir / spec.filename
        image = Image.new("RGB", (1200, 800), "white")
        draw = ImageDraw.Draw(image)
        lines = []
        for paragraph in spec.content.splitlines() or [spec.content]:
            lines.extend(textwrap.wrap(paragraph, width=90) or [""])
        draw.multiline_text((50, 50), "\n".join(lines[:35]), fill="black", spacing=10)
        image.save(path)
        return path
