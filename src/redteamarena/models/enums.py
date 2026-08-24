from __future__ import annotations

from enum import Enum


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNRESOLVED = "UNRESOLVED"


class SecurityResult(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class TaskResult(str, Enum):
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class FormatResult(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    PARTIALLY_PARSED = "PARTIALLY_PARSED"
    UNSUPPORTED = "UNSUPPORTED"


class ExecutionResult(str, Enum):
    COMPLETED = "COMPLETED"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    TIMEOUT = "TIMEOUT"
    STEP_LIMIT = "STEP_LIMIT"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"


class OverallResult(str, Enum):
    PASS = "PASS"
    SECURITY_FAIL = "SECURITY_FAIL"
    UTILITY_FAIL = "UTILITY_FAIL"
    PARTIAL = "PARTIAL"
    INDETERMINATE = "INDETERMINATE"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
