from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Set

DEPENDENCY_FILES = ("pyproject.toml", "requirements.txt", "setup.py", "setup.cfg", "package.json")
KNOWN_PACKAGES = {
    "openai",
    "anthropic",
    "ollama",
    "transformers",
    "langchain",
    "langgraph",
    "crewai",
    "autogen",
    "google-generativeai",
    "mistralai",
    "cohere",
    "litellm",
}


class DependencyScanner:
    def scan(self, root: Path) -> List[str]:
        found: Set[str] = set()
        for name in DEPENDENCY_FILES:
            path = root / name
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if name == "package.json":
                try:
                    data = json.loads(text)
                    names = set(data.get("dependencies", {})) | set(data.get("devDependencies", {}))
                    found.update(names)
                except (json.JSONDecodeError, AttributeError):
                    pass
            else:
                for package in KNOWN_PACKAGES:
                    if re.search(rf"(?im)(?<![\w-]){re.escape(package)}(?![\w-])", text):
                        found.add(package)
        return sorted(found)
