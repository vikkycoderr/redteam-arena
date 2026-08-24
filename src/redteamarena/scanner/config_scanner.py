from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Set

ENV_PATTERNS = (
    re.compile(r"(?:os\.)?getenv\(\s*['\"]([A-Z][A-Z0-9_]+)['\"]"),
    re.compile(r"environ(?:\.get)?\s*(?:\[|\()\s*['\"]([A-Z][A-Z0-9_]+)['\"]"),
    re.compile(r"\$\{([A-Z][A-Z0-9_]+)\}"),
    re.compile(r"(?m)^([A-Z][A-Z0-9_]+)\s*="),
)
SECRETISH = re.compile(r"(?i)(api[_-]?key|secret|token|password|credential)")
SAFE_CONFIG_NAMES = {
    "pyproject.toml",
    "setup.cfg",
    "package.json",
    ".env.example",
    "config.toml",
    "config.yaml",
    "config.yml",
    "config.json",
}


class ConfigScanner:
    def environment_names(self, files: List[Path]) -> List[str]:
        names: Set[str] = set()
        for path in files:
            if path.name == ".env":
                continue
            if (
                path.suffix not in {".py", ".toml", ".yaml", ".yml", ".json", ".ini", ".cfg"}
                and path.name not in SAFE_CONFIG_NAMES
            ):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in ENV_PATTERNS:
                names.update(pattern.findall(text))
        return sorted(names)

    def protected_resource_candidates(self, files: List[Path], root: Path) -> Dict[str, List[str]]:
        resources: Set[str] = {".env", ".git", "credentials", "secrets"}
        evidence: List[str] = []
        for path in files:
            relative = path.relative_to(root).as_posix()
            if SECRETISH.search(path.name):
                resources.add(relative)
                evidence.append(relative)
        return {"resources": sorted(resources), "evidence": sorted(evidence)}
