from __future__ import annotations

import json
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class JsonStore:
    """Atomic JSON persistence. Callers must pass already-redacted domain models."""

    @staticmethod
    def write(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = JsonStore._jsonable(value)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(serializable, handle, indent=2, sort_keys=True, ensure_ascii=False)
                handle.write("\n")
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def read(path: Path) -> Any:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return {str(key): JsonStore._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [JsonStore._jsonable(item) for item in value]
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Path):
            return value.as_posix()
        return value
