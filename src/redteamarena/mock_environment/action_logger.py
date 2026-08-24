from __future__ import annotations

from pathlib import Path
from typing import List

from ..models import ActionLogEntry
from ..storage import JsonStore


class ActionLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: List[ActionLogEntry] = []
        JsonStore.write(self.path, self.entries)

    def append(self, entry: ActionLogEntry) -> None:
        self.entries.append(entry)
        JsonStore.write(self.path, self.entries)

    def replace_last(self, entry: ActionLogEntry) -> None:
        if not self.entries:
            self.append(entry)
            return
        self.entries[-1] = entry
        JsonStore.write(self.path, self.entries)
