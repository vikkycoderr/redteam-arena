from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, Iterable, List

from ..models import AgentDetection, Evidence, RepositoryScan, ToolSpec
from .agent_deduplicator import AgentDeduplicator
from .confidence_scorer import ConfidenceScorer
from .config_scanner import ConfigScanner
from .dependency_scanner import DependencyScanner
from .python_ast_scanner import PythonAstScanner

IGNORED = {
    ".git",
    ".nox",
    ".redteamarena",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}
VIRTUAL_ENV_DIRECTORY = re.compile(r"^\.?venv(?:[-_.]?(?:py)?\d[\w.-]*)?$", re.IGNORECASE)
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".toml",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
}


def is_ignored_directory(name: str) -> bool:
    return name in IGNORED or VIRTUAL_ENV_DIRECTORY.fullmatch(name) is not None


class RepositoryScanner:
    def __init__(self, max_file_bytes: int = 2_000_000) -> None:
        self.max_file_bytes = max_file_bytes
        self.ast_scanner = PythonAstScanner()
        self.dependency_scanner = DependencyScanner()
        self.config_scanner = ConfigScanner()
        self.deduplicator = AgentDeduplicator()
        self.scorer = ConfidenceScorer()

    def files(self, root: Path) -> List[Path]:
        files: List[Path] = []
        for path in root.rglob("*"):
            try:
                relative_parts = path.relative_to(root).parts
            except ValueError:
                continue
            if any(is_ignored_directory(part) for part in relative_parts):
                continue
            if path.is_symlink() or not path.is_file():
                continue
            if path.name == ".env":
                continue
            try:
                if path.stat().st_size > self.max_file_bytes:
                    continue
            except OSError:
                continue
            if path.suffix.lower() in TEXT_SUFFIXES or path.name in {
                "Dockerfile",
                ".env.example",
                "requirements.txt",
            }:
                files.append(path)
        return sorted(files)

    def scan(self, root: Path) -> RepositoryScan:
        root = root.resolve()
        files = self.files(root)
        environment_names = self.config_scanner.environment_names(files)
        detections: List[AgentDetection] = []
        warnings: List[str] = []
        for path in files:
            if path.suffix == ".py":
                found, parse_warnings = self.ast_scanner.scan(path, root)
                detections.extend(found)
                warnings.extend(parse_warnings)
        dependencies = self.dependency_scanner.scan(root)
        fallback = self._fallback_detections(files, root, dependencies)
        existing = {(item.provider, item.model) for item in detections}
        detections.extend(item for item in fallback if (item.provider, item.model) not in existing)
        self._link_environment_configuration(detections, environment_names)
        self._link_repository_tool_schemas(detections, files, root)
        for detection in detections:
            detection.compute_fingerprint()
        detections = self.deduplicator.deduplicate(detections)
        return RepositoryScan(
            repository_root=str(root),
            files_scanned=len(files),
            ignored_directories=sorted(IGNORED),
            dependencies=dependencies,
            environment_variable_names=environment_names,
            detections=detections,
            warnings=warnings,
        )

    @staticmethod
    def _link_environment_configuration(
        detections: List[AgentDetection], environment_names: List[str]
    ) -> None:
        target_names = [name for name in environment_names if not name.startswith("RTA_")]
        model_names = [
            name
            for name in target_names
            if name == "MODEL" or name.endswith(("_MODEL", "_MODEL_NAME", "_MODEL_ID"))
        ]
        credential_prefixes = {
            "openai_compatible": ("OPENAI", "AZURE_OPENAI", "OPENROUTER"),
            "anthropic": ("ANTHROPIC",),
            "transformers": ("HF_", "HUGGINGFACE"),
        }
        for detection in detections:
            if not detection.model and not detection.configuration_env_vars and len(model_names) == 1:
                detection.configuration_env_vars = model_names.copy()
            prefixes = credential_prefixes.get(detection.provider, ())
            credentials = [
                name
                for name in target_names
                if name.endswith(("API_KEY", "TOKEN", "SECRET"))
                and name.startswith(prefixes)
            ]
            if len(credentials) == 1:
                detection.credential_env_vars = credentials

    def _link_repository_tool_schemas(
        self, detections: List[AgentDetection], files: List[Path], root: Path
    ) -> None:
        collections: Dict[str, List[ToolSpec]] = {}
        for path in files:
            if path.suffix != ".py":
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except (OSError, SyntaxError):
                continue
            assignments: Dict[str, ast.AST] = {}
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            assignments[target.id] = node.value
                elif (
                    isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name)
                    and node.value is not None
                ):
                    assignments[node.target.id] = node.value
            resolved: Dict[str, object] = {}

            relative = path.relative_to(root).as_posix()
            for name, assignment_node in assignments.items():
                value = self._resolve_assignment(
                    assignment_node, assignments, resolved, {name}
                )
                candidates = value if isinstance(value, list) else [value]
                tools = [
                    tool
                    for candidate in candidates
                    if (
                        tool := self.ast_scanner._schema_tool(
                            candidate, relative, getattr(assignment_node, "lineno", 1)
                        )
                    )
                    is not None
                ]
                if tools:
                    collections[name] = tools
        if not collections:
            return
        source_cache: Dict[str, str] = {}
        for detection in detections:
            if detection.tools:
                continue
            source_name = detection.call_sites[0].rsplit(":", 1)[0] if detection.call_sites else ""
            source = source_cache.get(source_name)
            if source is None:
                path = root / source_name
                source = (
                    path.read_text(encoding="utf-8", errors="replace")
                    if path.is_file() and not path.is_symlink()
                    else ""
                )
                source_cache[source_name] = source
            matched = [
                tools
                for name, tools in collections.items()
                if re.search(rf"\b{re.escape(name)}\b", source)
            ]
            if len(matched) == 1:
                detection.tools = [tool.model_copy(deep=True) for tool in matched[0]]
                detection.tool_confidence = self.scorer.score(
                    [evidence for tool in detection.tools for evidence in tool.evidence], []
                )

    @classmethod
    def _resolve_assignment(
        cls,
        node: ast.AST | None,
        assignments: Dict[str, ast.AST],
        resolved: Dict[str, object],
        active: set[str],
    ) -> object:
        if node is None:
            return None
        if isinstance(node, ast.Name) and node.id in assignments:
            if node.id in resolved:
                return resolved[node.id]
            if node.id in active:
                return None
            value = cls._resolve_assignment(
                assignments[node.id], assignments, resolved, active | {node.id}
            )
            resolved[node.id] = value
            return value
        if isinstance(node, (ast.List, ast.Tuple)):
            return [cls._resolve_assignment(item, assignments, resolved, active) for item in node.elts]
        if isinstance(node, ast.Dict):
            return {
                cls._resolve_assignment(key, assignments, resolved, active): cls._resolve_assignment(
                    value, assignments, resolved, active
                )
                for key, value in zip(node.keys, node.values)
                if key is not None
            }
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError):
            return None

    def _fallback_detections(
        self, files: Iterable[Path], root: Path, dependencies: List[str]
    ) -> List[AgentDetection]:
        dependency_evidence: List[Evidence] = []
        providers: List[str] = []
        provider_map = {
            "openai": "openai_compatible",
            "anthropic": "anthropic",
            "ollama": "ollama",
            "transformers": "transformers",
        }
        for dependency in dependencies:
            if dependency in provider_map:
                providers.append(provider_map[dependency])
                dependency_evidence.append(
                    Evidence(
                        kind="dependency_mention",
                        path="pyproject.toml/requirements.txt",
                        detail=f"Dependency {dependency} suggests an LLM integration",
                        points=1,
                    )
                )
        if len(set(providers)) > 1:
            dependency_evidence.append(
                Evidence(
                    kind="conflicting_provider_evidence",
                    path="pyproject.toml/requirements.txt",
                    detail="Multiple provider dependencies prevent a unique configuration link",
                    points=-2,
                )
            )
        configured: Dict[tuple[str, str], List[Evidence]] = {}
        for path in files:
            if path.suffix not in {".toml", ".yaml", ".yml", ".json", ".md"}:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            provider_match = re.search(r"(?im)^\s*provider\s*[:=]\s*['\"]?([\w.-]+)", text)
            file_provider = (
                provider_map.get(provider_match.group(1), provider_match.group(1))
                if provider_match
                else providers[0]
                if len(set(providers)) == 1
                else "unknown"
            )
            for match in re.finditer(
                r"(?im)^\s*(?:model|model_name)\s*[:=]\s*['\"]?([\w./:-]+)", text
            ):
                model = match.group(1)
                kind = (
                    "readme_only"
                    if path.name.lower().startswith("readme")
                    else "config_model_mention"
                )
                points = 0.5 if kind == "readme_only" else 1
                configured.setdefault((file_provider, model), []).append(
                    Evidence(
                        kind=kind,
                        path=path.relative_to(root).as_posix(),
                        line=text[: match.start()].count("\n") + 1,
                        detail="Unlinked model configuration mention",
                        points=points,
                    )
                )
        detections: List[AgentDetection] = []
        if configured:
            for index, ((provider, model), config_evidence) in enumerate(
                sorted(configured.items()), start=1
            ):
                evidence = dependency_evidence + config_evidence
                detection = AgentDetection(
                    agent_id=f"unlinked_llm_configuration_{index}",
                    name=f"Unlinked LLM configuration {index}",
                    provider=provider,
                    model=model,
                    confidence=self.scorer.score(
                        evidence, ["LLM call site", "linked runtime configuration"]
                    ),
                )
                detection.compute_fingerprint()
                detections.append(detection)
            return detections
        if not dependency_evidence:
            return []
        provider = providers[0] if len(set(providers)) == 1 else "unknown"
        detection = AgentDetection(
            agent_id="unresolved_llm_configuration",
            name="Unresolved LLM configuration",
            provider=provider,
            confidence=self.scorer.score(
                dependency_evidence, ["LLM call site", "linked runtime configuration"]
            ),
        )
        detection.compute_fingerprint()
        return [detection]
