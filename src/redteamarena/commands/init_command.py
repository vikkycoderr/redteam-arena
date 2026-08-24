from pathlib import Path
from typing import List

from ..bootstrap import initialize


def run(root: Path) -> List[Path]:
    return initialize(root)
