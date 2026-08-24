from pathlib import Path

from ..models import RepositoryReport
from ..pipeline import EvaluationSelection, Pipeline


def run(root: Path, selection: EvaluationSelection | None = None) -> RepositoryReport:
    return Pipeline(root).preflight(selection)
