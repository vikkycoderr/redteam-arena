from __future__ import annotations

import ast
import contextlib
import re
from pathlib import Path
from typing import List, Set

from ..models import (
    AgentDetection,
    Confidence,
    ConfidenceLevel,
    Evidence,
    ProjectProfile,
    RepositoryScan,
)
from ..scanner.repository_scanner import is_ignored_directory
from ..utils.secrets import redact_potential_secrets


class ApplicationProfiler:
    @staticmethod
    def needs_fallback(profile: ProjectProfile, threshold: float) -> bool:
        normalized = max(0.0, min(1.0, profile.confidence.score / 10.0))
        return normalized < threshold

    @staticmethod
    def safe_fallback_package(
        profile: ProjectProfile, scan: RepositoryScan, root: Path | None = None
    ) -> dict[str, object]:
        readme_excerpt = ""
        if root:
            for name in ("README.md", "README.rst", "README.txt"):
                path = root / name
                if path.is_file() and not path.is_symlink():
                    readme_excerpt = redact_potential_secrets(
                        path.read_text(encoding="utf-8", errors="replace")[:40_000]
                    )
                    break
        return {
            "deterministic_profile": profile.model_dump(mode="json"),
            "readme_excerpt": readme_excerpt,
            "important_filenames": profile.important_files,
            "detected_operations": profile.important_operations,
            "detected_import_dependencies": scan.dependencies,
            "non_secret_configuration_names": scan.environment_variable_names,
            "output_schema": ProjectProfile.model_json_schema(),
        }

    def profile(self, root: Path, scan: RepositoryScan) -> ProjectProfile:
        files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and not any(is_ignored_directory(part) for part in path.relative_to(root).parts)
        )
        root_readme = next(
            (
                root / name
                for name in ("README.md", "README.rst", "README.txt")
                if (root / name).is_file()
            ),
            None,
        )
        readme = root_readme or next(
            (
                path
                for path in files
                if path.name.lower() in {"readme.md", "readme.rst", "readme.txt"}
            ),
            None,
        )
        purpose = ""
        evidence: List[Evidence] = []
        if readme:
            text = readme.read_text(encoding="utf-8", errors="replace")[:40_000]
            purpose = redact_potential_secrets(self._purpose(text))
            evidence.append(
                Evidence(
                    kind="readme",
                    path=readme.relative_to(root).as_posix(),
                    detail="Project purpose inferred from repository documentation",
                    points=3,
                )
            )
        operations: Set[str] = set()
        routes: Set[str] = set()
        project_types: Set[str] = set()
        for path in files:
            relative = path.relative_to(root).as_posix()
            if path.suffix == ".py" and path.stat().st_size <= 2_000_000:
                try:
                    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        operations.add(node.name)
                        for decorator in node.decorator_list:
                            if isinstance(decorator, ast.Call) and decorator.args:
                                name = self._name(decorator.func)
                                if name.endswith(("get", "post", "put", "delete", "patch")):
                                    with contextlib.suppress(ValueError, TypeError):
                                        routes.add(ast.literal_eval(decorator.args[0]))
                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        names = self._import_names(node)
                        if {"fastapi", "flask", "django"} & names:
                            project_types.add("web application")
                        if {"typer", "click", "argparse"} & names:
                            project_types.add("command-line application")
            if relative.startswith("tests/"):
                evidence.append(
                    Evidence(
                        kind="tests",
                        path=relative,
                        detail="Tests contribute behavioral evidence",
                        points=0.1,
                    )
                )
        if not project_types:
            project_types.add(
                "Python application"
                if any(path.suffix == ".py" for path in files)
                else "software repository"
            )
        if routes:
            operations.update(f"route:{route}" for route in routes)
        detected_tools = sorted({tool.name for agent in scan.detections for tool in agent.tools})
        operations.update(detected_tools)
        sensitive = sorted(
            op
            for op in operations
            if re.search(r"(?i)(write|delete|change|set|send|execute|auth|secret|config)", op)
        )
        score = min(10.0, sum(item.points for item in evidence))
        level = (
            ConfidenceLevel.HIGH
            if score >= 8
            else ConfidenceLevel.MEDIUM
            if score >= 4
            else ConfidenceLevel.LOW
            if score >= 1
            else ConfidenceLevel.UNRESOLVED
        )
        profile = ProjectProfile(
            project_type=", ".join(sorted(project_types)),
            main_purpose=purpose,
            agent_purposes={
                agent.agent_id: self._agent_purpose(agent) for agent in scan.detections
            },
            important_operations=sorted(operations)[:200],
            sensitive_operations=sensitive[:100],
            protected_resources=sorted(
                {resource for agent in scan.detections for resource in agent.protected_resources}
            ),
            important_files=sorted(path.relative_to(root).as_posix() for path in files)[:200],
            input_types=sorted(
                {kind for agent in scan.detections for kind in self._inputs(agent)} or {"text"}
            ),
            detected_tools=detected_tools,
            confidence=Confidence(score=score, level=level, evidence=evidence),
        )
        return profile

    @staticmethod
    def _purpose(text: str) -> str:
        lines = [line.strip(" #\t") for line in text.splitlines() if line.strip(" #\t")]
        for line in lines:
            if len(line) >= 20 and not line.startswith(("[!", "<", "```")):
                return line[:500]
        return lines[0][:500] if lines else ""

    @staticmethod
    def _agent_purpose(agent: AgentDetection) -> str:
        if agent.system_prompt:
            return agent.system_prompt[:500]
        return f"LLM agent configured at {', '.join(agent.call_sites)}"

    @staticmethod
    def _inputs(agent: AgentDetection) -> List[str]:
        result = ["text"]
        if agent.capabilities.pdf_input:
            result.append("pdf")
        if agent.capabilities.image_input:
            result.append("image")
        return result

    @staticmethod
    def _name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{ApplicationProfiler._name(node.value)}.{node.attr}"
        return ""

    @staticmethod
    def _import_names(node: ast.Import | ast.ImportFrom) -> Set[str]:
        if isinstance(node, ast.Import):
            return {alias.name.split(".")[0] for alias in node.names}
        return {node.module.split(".")[0]} if node.module else set()
