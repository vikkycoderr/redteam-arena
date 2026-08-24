from pathlib import Path
from typing import Tuple

from ..models import ProjectProfile, RepositoryScan
from ..pipeline import Pipeline


def run(root: Path) -> Tuple[RepositoryScan, ProjectProfile]:
    return Pipeline(root).scan()
