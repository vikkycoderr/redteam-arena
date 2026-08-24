from __future__ import annotations

from pathlib import Path
from typing import List

from .config import DEFAULT_CONFIG, ENV_EXAMPLE
from .storage import PathManager

GITIGNORE_ENTRIES = [
    ".env",
    ".redteamarena/scan/",
    ".redteamarena/agents/",
    ".redteamarena/reports/",
    ".redteamarena/cache/",
    "*.pyc",
    "__pycache__/",
    ".venv/",
]


def _write_if_missing(path: Path, content: str) -> bool:
    if path.exists():
        return False
    path.write_text(content, encoding="utf-8")
    return True


def _update_gitignore(path: Path) -> bool:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    present = set(existing.splitlines())
    additions = [entry for entry in GITIGNORE_ENTRIES if entry not in present]
    if not additions:
        return False
    separator = "" if not existing or existing.endswith("\n") else "\n"
    block = separator + "\n# RedTeamArena\n" + "\n".join(additions) + "\n"
    path.write_text(existing + block, encoding="utf-8")
    return True


def initialize(root: Path) -> List[Path]:
    root = root.resolve()
    paths = PathManager(root)
    paths.ensure_base()
    changed: List[Path] = []
    config_path = paths.state / "config.toml"
    if _write_if_missing(config_path, DEFAULT_CONFIG):
        changed.append(config_path)
    env_example = root / ".env.example"
    if _write_if_missing(env_example, ENV_EXAMPLE):
        changed.append(env_example)
    gitignore = root / ".gitignore"
    if _update_gitignore(gitignore):
        changed.append(gitignore)
    return changed
