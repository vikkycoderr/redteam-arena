from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .enums import (
    ConfidenceLevel,
    Difficulty,
    ExecutionResult,
    FormatResult,
    OverallResult,
    SecurityResult,
    TaskResult,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Evidence(StrictModel):
    kind: str
    path: str
    line: Optional[int] = None
    detail: str
    points: float = 0.0


class Confidence(StrictModel):
    score: float = 0.0
    level: ConfidenceLevel = ConfidenceLevel.UNRESOLVED
    evidence: List[Evidence] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)


class ToolParameter(StrictModel):
    type: str = "string"
    description: str = ""
    required: bool = True
    default: Any = None


class ToolSpec(StrictModel):
    name: str
    description: str
    parameters: Dict[str, ToolParameter] = Field(default_factory=dict)
    source: str = "deterministic"
    description_confidence: float = 0.5
    sensitive: bool = False
    evidence: List[Evidence] = Field(default_factory=list)

    def provider_schema(self) -> Dict[str, Any]:
        properties = {
            name: {
                "type": parameter.type,
                "description": parameter.description,
                **({"default": parameter.default} if parameter.default is not None else {}),
            }
            for name, parameter in self.parameters.items()
        }
        required = [name for name, parameter in self.parameters.items() if parameter.required]
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }


class Capabilities(StrictModel):
    text_input: bool = True
    pdf_input: bool = False
    inline_document_fallback: bool = True
    image_input: bool = False
    native_tool_calling: bool = False
    structured_output: bool = False
    system_messages: bool = True
    multiple_messages: bool = True
    file_upload: bool = False


class AgentDetection(StrictModel):
    agent_id: str
    name: str
    provider: str = "unknown"
    protocol: Optional[str] = None
    static_call_recipe: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    model_expression: Optional[str] = None
    system_prompt: str = ""
    prompt_source: Optional[str] = None
    call_sites: List[str] = Field(default_factory=list)
    tools: List[ToolSpec] = Field(default_factory=list)
    permissions: Dict[str, Any] = Field(default_factory=dict)
    readable_paths: List[str] = Field(default_factory=list)
    writable_paths: List[str] = Field(default_factory=list)
    protected_resources: List[str] = Field(default_factory=list)
    capabilities: Capabilities = Field(default_factory=Capabilities)
    configuration_env_vars: List[str] = Field(default_factory=list)
    credential_env_vars: List[str] = Field(default_factory=list)
    confidence: Confidence = Field(default_factory=Confidence)
    capability_confidence: Confidence = Field(default_factory=Confidence)
    tool_confidence: Confidence = Field(default_factory=Confidence)
    protected_resource_confidence: Confidence = Field(default_factory=Confidence)
    equivalent_call_sites: List[str] = Field(default_factory=list)
    configuration_fingerprint: str = ""

    @property
    def eligible(self) -> bool:
        return self.confidence.level in {ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM}

    def compute_fingerprint(self) -> str:
        security_config = {
            "provider": self.provider,
            "protocol": self.protocol,
            "static_call_recipe": self.static_call_recipe,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "tools": [tool.model_dump(mode="json", exclude={"evidence"}) for tool in self.tools],
            "permissions": self.permissions,
            "capabilities": self.capabilities.model_dump(mode="json"),
            "protected_resources": sorted(self.protected_resources),
        }
        digest = hashlib.sha256(
            json.dumps(security_config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        self.configuration_fingerprint = f"agent_cfg_{digest}"
        return self.configuration_fingerprint


class ProjectProfile(StrictModel):
    project_type: str = "unknown"
    main_purpose: str = ""
    agent_purposes: Dict[str, str] = Field(default_factory=dict)
    important_operations: List[str] = Field(default_factory=list)
    sensitive_operations: List[str] = Field(default_factory=list)
    protected_resources: List[str] = Field(default_factory=list)
    important_files: List[str] = Field(default_factory=list)
    input_types: List[str] = Field(default_factory=lambda: ["text"])
    output_types: List[str] = Field(default_factory=lambda: ["text"])
    detected_tools: List[str] = Field(default_factory=list)
    source: str = "deterministic"
    confidence: Confidence = Field(default_factory=Confidence)


class RepositoryScan(StrictModel):
    schema_version: str = "1.0.0"
    repository_root: str
    scanned_at: str = Field(default_factory=utc_now)
    files_scanned: int = 0
    ignored_directories: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    environment_variable_names: List[str] = Field(default_factory=list)
    detections: List[AgentDetection] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


APPROVED_CATEGORIES = {
    "direct_instruction_override",
    "fake_authority",
    "instruction_contradiction",
    "goal_hijacking",
    "scope_expansion",
    "unauthorized_tool_use",
    "protected_resource_access",
    "test_manipulation",
    "silent_code_sabotage",
    "validation_removal",
    "configuration_manipulation",
    "research_result_manipulation",
    "false_technical_guidance",
    "conflicting_information",
    "multi_document_attack",
    "secret_leakage",
    "obfuscated_instruction",
    "resource_wasting",
    "workflow_derailment",
    "output_manipulation",
}

RULE_TYPES = {
    "forbidden_tool_call",
    "forbidden_tool_argument",
    "forbidden_path_read",
    "forbidden_path_write",
    "required_tool_call",
    "required_tool_argument",
    "forbidden_output_string",
    "fake_secret_leakage",
    "maximum_tool_calls",
    "required_task_completion_marker",
}


class ArtifactSpec(StrictModel):
    artifact_id: str
    type: str
    filename: str
    content: str


class Rule(StrictModel):
    rule_id: str
    rule_type: str
    tool: Optional[str] = None
    arguments: Dict[str, Any] = Field(default_factory=dict)
    argument: Optional[str] = None
    value: Any = None
    path_prefix: Optional[str] = None
    string: Optional[str] = None
    maximum: Optional[int] = None
    marker: Optional[str] = None


class TargetVisible(StrictModel):
    user_task: str
    task_relevant_tools: List[str] = Field(default_factory=list)
    artifacts: List[ArtifactSpec] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)


class AttackMetadata(StrictModel):
    attack_description: str
    attack_goal: Dict[str, Any] = Field(default_factory=dict)
    forbidden_actions: List[Rule] = Field(default_factory=list)
    required_actions: List[Rule] = Field(default_factory=list)
    expected_safe_behavior: str
    risk_description: str
    failure_tags: List[str] = Field(default_factory=list)


class Scenario(StrictModel):
    scenario_id: str
    title: str
    category: str
    difficulty: Difficulty
    scenario_description: str
    target_visible: TargetVisible
    attack_metadata: AttackMetadata

    @field_validator("scenario_id")
    @classmethod
    def stable_id(cls, value: str) -> str:
        if not value or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789_" for ch in value):
            raise ValueError("scenario_id must be lowercase snake_case")
        return value


class ScenarioSet(StrictModel):
    schema_version: str = "1.0.0"
    scenario_set_id: str
    agent_id: str
    configuration_fingerprint: str
    scenarios: List[Scenario]


class ProposedAction(StrictModel):
    tool: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    call_id: Optional[str] = None


class ModelResult(StrictModel):
    text: str = ""
    proposed_actions: List[ProposedAction] = Field(default_factory=list)
    format_status: str = "valid"
    finish_reason: Optional[str] = None
    raw_response_reference: Optional[str] = None
    raw: Any = Field(default=None, exclude=True)


class ActionLogEntry(StrictModel):
    step: int
    tool: str
    arguments: Dict[str, Any]
    validation_result: str
    mock_handler: Optional[str] = None
    simulated_status: str
    real_project_touched: bool = False
    timestamp: str = Field(default_factory=utc_now)
    mock_result: Any = None


class ScenarioResult(StrictModel):
    scenario_id: str
    scenario_title: str
    scenario_description: str
    attack_description: str = ""
    expected_safe_behavior: str = ""
    category: str
    difficulty: Difficulty
    legitimate_task: str
    task_relevant_tools: List[str] = Field(default_factory=list)
    artifact_summary: List[Dict[str, str]] = Field(default_factory=list)
    raw_response_reference: Optional[str] = None
    normalized_response: ModelResult
    action_log: List[ActionLogEntry] = Field(default_factory=list)
    security_result: SecurityResult
    task_result: TaskResult
    format_result: FormatResult
    execution_result: ExecutionResult
    overall_result: OverallResult
    violated_rules: List[str] = Field(default_factory=list)
    proposed_violations: List[str] = Field(default_factory=list)
    executed_violations: List[str] = Field(default_factory=list)
    missing_required_actions: List[str] = Field(default_factory=list)
    failure_tags: List[str] = Field(default_factory=list)
    failure_patterns: List[str] = Field(default_factory=list)
    risk_explanation: str = ""
    why_failed: str = ""
    recommendations: List[str] = Field(default_factory=list)
    evidence_paths: List[str] = Field(default_factory=list)


class AgentReport(StrictModel):
    agent_id: str
    configuration_fingerprint: str
    provider: str
    model: Optional[str]
    totals: Dict[str, int]
    security_pass_rate: float
    task_completion_rate: float
    full_pass_rate: float
    per_category: Dict[str, Dict[str, int]]
    per_difficulty: Dict[str, Dict[str, int]]
    critical_failures: List[str]
    repeated_failure_patterns: List[str]
    scenario_results: List[ScenarioResult]


class RepositoryReport(StrictModel):
    generated_at: str = Field(default_factory=utc_now)
    repository_root: str
    totals: Dict[str, int]
    highest_risk_agent: Optional[str] = None
    repeated_failure_patterns: List[str] = Field(default_factory=list)
    agent_reports: List[AgentReport] = Field(default_factory=list)
    untested_detections: List[Dict[str, Any]] = Field(default_factory=list)


def safe_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name
