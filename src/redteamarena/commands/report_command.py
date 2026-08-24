from pathlib import Path

from ..models import RepositoryReport
from ..pipeline import Pipeline


def run(root: Path) -> RepositoryReport:
    return Pipeline(root).latest_report()
