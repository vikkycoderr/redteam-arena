from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .json_store import JsonStore


class CacheStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: Dict[str, Any] = JsonStore.read(path) if path.exists() else {}

    @staticmethod
    def security_key(payload: Dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        return self.data.get(key)

    def put(self, key: str, value: Any) -> None:
        self.data[key] = value
        JsonStore.write(self.path, self.data)
