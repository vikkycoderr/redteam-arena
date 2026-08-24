from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..models import AgentDetection, Capabilities, Evidence, ToolParameter, ToolSpec
from ..utils.secrets import redact_potential_secrets, sanitize_scanned_value
from .confidence_scorer import ConfidenceScorer
from .tool_scanner import ToolScanner

PROVIDER_IMPORTS = {
    "openai": "openai",
    "anthropic": "anthropic",
    "ollama": "ollama",
    "transformers": "transformers",
    "langchain": "openai_compatible",
    "litellm": "openai_compatible",
    "mistralai": "openai_compatible",
    "cohere": "unknown",
    "google.generativeai": "unknown",
}

CALL_PATTERNS = (
    (re.compile(r"(?:^|\.)chat\.completions\.create$"), "openai_compatible"),
    (re.compile(r"(?:^|\.)responses\.create$"), "openai_compatible"),
    (re.compile(r"(?:^|\.)messages\.create$"), "anthropic"),
    (re.compile(r"(?:^|\.)(?:chat|generate)$"), "ollama"),
    (re.compile(r"(?:^|\.)(?:pipeline|from_pretrained)$"), "transformers"),
    (re.compile(r"(?:^|\.)(?:completion|acompletion)$"), "openai_compatible"),
    (re.compile(r"(?:^|\.)(?:invoke|ainvoke|predict)$"), "unknown"),
)


class PythonAstScanner:
    def __init__(self) -> None:
        self.scorer = ConfidenceScorer()
        self.tool_scanner = ToolScanner()

    def scan(self, path: Path, root: Path) -> Tuple[List[AgentDetection], List[str]]:
        relative = path.relative_to(root).as_posix()
        warnings: List[str] = []
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=relative)
        except (OSError, SyntaxError) as exc:
            return [], [f"Could not parse {relative}: {type(exc).__name__}"]

        constants, env_expressions = self._constants(tree)
        imports = self._imports(tree)
        clients = self._client_bindings(tree)
        factories = self._factory_providers(tree)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                names = self._assigned_names(node)
                if isinstance(value, ast.Call):
                    called = self._name(value.func).split(".")[-1]
                    if called in factories:
                        for name in names:
                            clients[name] = factories[called]
        tools = self.tool_scanner.discover(tree, relative)
        detections: List[AgentDetection] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = self._name(node.func)
            provider = self._provider_for_call(call_name, clients, imports)
            if provider is None:
                continue
            model, expression, model_env = self._extract_model(node, constants, env_expressions)
            prompt, prompt_source = self._extract_prompt(node, constants)
            call_tools = self._call_tools(node, tools, constants, relative)
            permissions, readable, writable, protected = self._permissions(node, constants)
            evidence: List[Evidence] = []
            if model:
                evidence.append(
                    Evidence(
                        kind="sdk_call_exact_model",
                        path=relative,
                        line=node.lineno,
                        detail=f"{call_name} uses a statically resolved model",
                        points=6,
                    )
                )
            else:
                evidence.append(
                    Evidence(
                        kind="known_sdk_pattern",
                        path=relative,
                        line=node.lineno,
                        detail=f"Known LLM call pattern: {call_name}",
                        points=5,
                    )
                )
                evidence.append(
                    Evidence(
                        kind="unresolved_model",
                        path=relative,
                        line=node.lineno,
                        detail=f"Model expression could not be statically resolved: {expression or 'missing'}",
                        points=-2,
                    )
                )
            imported_providers = {value for key, value in imports.items() if key}
            if provider in imported_providers or (
                provider == "openai_compatible" and "openai" in imported_providers
            ):
                evidence.append(
                    Evidence(
                        kind="provider_import",
                        path=relative,
                        line=1,
                        detail=f"Provider import supports {provider}",
                        points=2,
                    )
                )
            if model_env:
                evidence.append(
                    Evidence(
                        kind="environment_linked_to_call",
                        path=relative,
                        line=node.lineno,
                        detail=f"Environment variable {model_env} supplies the model name",
                        points=3,
                    )
                )
            call_site = f"{relative}:{node.lineno}"
            suffix = hashlib.sha1(call_site.encode()).hexdigest()[:8]
            base_name = self._enclosing_name(tree, node) or path.stem
            agent_id = self._slug(f"{base_name}_{suffix}")
            capabilities = self._capabilities(provider, call_name, node)
            missing = [] if model else ["exact model name"]
            detection = AgentDetection(
                agent_id=agent_id,
                name=base_name,
                provider=provider,
                protocol="openai_compatible" if provider == "openai_compatible" else provider,
                model=model,
                model_expression=expression,
                system_prompt=prompt,
                prompt_source=prompt_source,
                call_sites=[call_site],
                tools=call_tools,
                permissions=permissions,
                readable_paths=readable,
                writable_paths=writable,
                protected_resources=protected,
                capabilities=capabilities,
                configuration_env_vars=[model_env] if model_env else [],
                credential_env_vars=self._credential_envs(provider, source),
                confidence=self.scorer.score(evidence, missing),
                capability_confidence=self.scorer.score(
                    [
                        Evidence(
                            kind="call_shape",
                            path=relative,
                            line=node.lineno,
                            detail="Capabilities inferred from provider call arguments",
                            points=5,
                        )
                    ]
                ),
                tool_confidence=self.scorer.score(
                    [item for tool in call_tools for item in tool.evidence],
                    [] if call_tools else ["explicit tool registration"],
                ),
            )
            detection.compute_fingerprint()
            detections.append(detection)
        return detections, warnings

    @staticmethod
    def _slug(value: str) -> str:
        value = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
        return value or "agent"

    @staticmethod
    def _imports(tree: ast.AST) -> Dict[str, str]:
        imports: Dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in PROVIDER_IMPORTS:
                        imports[alias.asname or root] = PROVIDER_IMPORTS[root]
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                if root in PROVIDER_IMPORTS:
                    for alias in node.names:
                        imports[alias.asname or alias.name] = PROVIDER_IMPORTS[root]
        return imports

    def _client_bindings(self, tree: ast.AST) -> Dict[str, str]:
        clients: Dict[str, str] = {}
        constructors = {
            "OpenAI": "openai_compatible",
            "AsyncOpenAI": "openai_compatible",
            "Anthropic": "anthropic",
            "AsyncAnthropic": "anthropic",
            "Client": "ollama",
        }
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not isinstance(
                node.value, ast.Call
            ):
                continue
            constructor = self._name(node.value.func).split(".")[-1]
            provider = constructors.get(constructor)
            if provider:
                for name in self._assigned_names(node):
                    clients[name] = provider
        return clients

    def _factory_providers(self, tree: ast.AST) -> Dict[str, str]:
        factories: Dict[str, str] = {}
        constructors = {
            "OpenAI": "openai_compatible",
            "AsyncOpenAI": "openai_compatible",
            "Anthropic": "anthropic",
            "AsyncAnthropic": "anthropic",
            "Client": "ollama",
        }
        for node in getattr(tree, "body", []):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Return) and isinstance(child.value, ast.Call):
                    constructor = self._name(child.value.func).split(".")[-1]
                    if constructor in constructors:
                        factories[node.name] = constructors[constructor]
        return factories

    @staticmethod
    def _assigned_names(node: ast.Assign | ast.AnnAssign) -> List[str]:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return [target.id for target in targets if isinstance(target, ast.Name)]

    def _provider_for_call(
        self, call_name: str, clients: Dict[str, str], imports: Dict[str, str]
    ) -> Optional[str]:
        imported = {
            "openai_compatible" if provider == "openai" else provider
            for provider in imports.values()
        }
        context = set(clients.values()) | imported
        for pattern, provider in CALL_PATTERNS:
            if pattern.search(call_name):
                root = call_name.split(".")[0]
                if root in clients:
                    return clients[root]
                if provider == "unknown":
                    return next(iter(context)) if len(context) == 1 else None
                return provider if provider in context else None
        return None

    @staticmethod
    def _constants(tree: ast.AST) -> Tuple[Dict[str, Any], Dict[str, str]]:
        constants: Dict[str, Any] = {}
        envs: Dict[str, str] = {}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [target.id for target in targets if isinstance(target, ast.Name)]
            if not names or node.value is None:
                continue
            try:
                value = ast.literal_eval(node.value)
                for name in names:
                    constants[name] = value
            except (ValueError, TypeError):
                if isinstance(node.value, ast.Call):
                    called = PythonAstScanner._name(node.value.func)
                    if called.endswith(("getenv", "environ.get")) and node.value.args:
                        try:
                            env_name = ast.literal_eval(node.value.args[0])
                        except (ValueError, TypeError):
                            continue
                        if isinstance(env_name, str):
                            for name in names:
                                envs[name] = env_name
        return constants, envs

    @staticmethod
    def _resolve(node: ast.AST, constants: Dict[str, Any]) -> Any:
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError):
            if isinstance(node, ast.Name):
                return constants.get(node.id)
        return None

    def _extract_model(
        self, call: ast.Call, constants: Dict[str, Any], envs: Dict[str, str]
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        node: Optional[ast.AST] = None
        for keyword in call.keywords:
            if keyword.arg in {"model", "model_name", "model_id"}:
                node = keyword.value
                break
        if node is None and call.args and self._name(call.func).endswith("from_pretrained"):
            node = call.args[0]
        if node is None:
            return None, None, None
        resolved = self._resolve(node, constants)
        expression = ast.unparse(node) if hasattr(ast, "unparse") else type(node).__name__
        env_name = envs.get(node.id) if isinstance(node, ast.Name) else None
        if isinstance(node, ast.Call) and self._name(node.func).endswith(("getenv", "environ.get")):
            if node.args:
                candidate = self._resolve(node.args[0], constants)
                env_name = candidate if isinstance(candidate, str) else None
        elif isinstance(node, ast.Subscript) and self._name(node.value).endswith("environ"):
            candidate = self._resolve(node.slice, constants)
            env_name = candidate if isinstance(candidate, str) else None
        if isinstance(resolved, str) and not re.search(
            r"(?i)(secret|key|token|password)", resolved
        ):
            sanitized = redact_potential_secrets(resolved)
            if "[REDACTED]" not in sanitized:
                return sanitized, expression, env_name
        return None, expression, env_name

    def _extract_prompt(
        self, call: ast.Call, constants: Dict[str, Any]
    ) -> Tuple[str, Optional[str]]:
        for keyword in call.keywords:
            if keyword.arg in {"system", "system_prompt", "instructions"}:
                value = self._resolve(keyword.value, constants)
                if isinstance(value, str):
                    return redact_potential_secrets(value), "inline_or_constant"
            if keyword.arg == "messages":
                value = self._resolve(keyword.value, constants)
                if isinstance(value, list):
                    for message in value:
                        if isinstance(message, dict) and message.get("role") == "system":
                            content = message.get("content")
                            if isinstance(content, str):
                                return redact_potential_secrets(content), "messages"
        return "", None

    def _call_tools(
        self,
        call: ast.Call,
        tools: Dict[str, ToolSpec],
        constants: Dict[str, Any],
        relative_path: str,
    ) -> List[ToolSpec]:
        selected: List[ToolSpec] = []
        for keyword in call.keywords:
            if keyword.arg not in {"tools", "functions"}:
                continue
            nodes = (
                keyword.value.elts
                if isinstance(keyword.value, (ast.List, ast.Tuple))
                else [keyword.value]
            )
            for item in nodes:
                if isinstance(item, ast.Name) and item.id in tools:
                    selected.append(tools[item.id])
                    continue
                value = self._resolve(item, constants)
                candidates = value if isinstance(value, list) else [value]
                for candidate in candidates:
                    parsed = self._schema_tool(candidate, relative_path, call.lineno)
                    if parsed:
                        selected.append(parsed)
        return selected or list(tools.values())

    @staticmethod
    def _schema_tool(value: Any, path: str, line: int) -> Optional[ToolSpec]:
        if not isinstance(value, dict):
            return None
        function = value.get("function", value)
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            return None
        schema = function.get("parameters", {})
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        required = set(schema.get("required", [])) if isinstance(schema, dict) else set()
        parameters: Dict[str, ToolParameter] = {}
        if isinstance(properties, dict):
            for name, definition in properties.items():
                if not isinstance(name, str) or not isinstance(definition, dict):
                    continue
                parameters[name] = ToolParameter(
                    type=str(definition.get("type", "string")),
                    description=str(definition.get("description", "")),
                    required=name in required,
                    default=sanitize_scanned_value(definition.get("default")),
                )
        return ToolSpec(
            name=function["name"],
            description=str(
                sanitize_scanned_value(function.get("description"))
                or f"Simulated project operation: {function['name'].replace('_', ' ')}."
            ),
            parameters=parameters,
            source="existing_schema",
            description_confidence=0.95,
            evidence=[
                Evidence(
                    kind="existing_tool_schema",
                    path=path,
                    line=line,
                    detail=f"Existing JSON schema for {function['name']}",
                    points=4,
                )
            ],
        )

    def _permissions(
        self, call: ast.Call, constants: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[str], List[str], List[str]]:
        permissions: Dict[str, Any] = {}
        readable: List[str] = []
        writable: List[str] = []
        protected: List[str] = []
        for keyword in call.keywords:
            value = self._resolve(keyword.value, constants)
            if keyword.arg == "permissions" and isinstance(value, dict):
                permissions = sanitize_scanned_value(value)
            elif keyword.arg in {"readable_paths", "allowed_read_paths"} and isinstance(
                value, list
            ):
                readable = [str(item) for item in value]
            elif keyword.arg in {"writable_paths", "allowed_write_paths"} and isinstance(
                value, list
            ):
                writable = [str(item) for item in value]
            elif keyword.arg == "protected_resources" and isinstance(value, list):
                protected = [str(item) for item in value]
        return permissions, readable, writable, protected

    @staticmethod
    def _capabilities(provider: str, call_name: str, call: ast.Call) -> Capabilities:
        keywords = {keyword.arg for keyword in call.keywords}
        native = "tools" in keywords or "functions" in keywords
        structured = "response_format" in keywords or "format" in keywords
        return Capabilities(
            native_tool_calling=native or provider in {"anthropic", "openai_compatible"},
            structured_output=structured,
            pdf_input="responses.create" in call_name,
            image_input="responses.create" in call_name,
            file_upload="responses.create" in call_name,
        )

    @staticmethod
    def _credential_envs(provider: str, source: str) -> List[str]:
        defaults = {
            "openai_compatible": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "ollama": "",
            "transformers": "HF_TOKEN",
        }
        pattern = re.compile(r"['\"]([A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET)[A-Z0-9_]*)['\"]")
        found = set(pattern.findall(source))
        default = defaults.get(provider, "")
        provider_prefixes = {
            "openai_compatible": ("OPENAI", "AZURE_OPENAI"),
            "anthropic": ("ANTHROPIC",),
            "ollama": ("OLLAMA",),
            "transformers": ("HF_", "HUGGINGFACE"),
        }
        prefixes = provider_prefixes.get(provider, ())
        linked = {name for name in found if name.startswith(prefixes) or "API_KEY" in name}
        if default and not linked:
            linked.add(default)
        return sorted(linked)

    @staticmethod
    def _name(node: ast.AST) -> str:
        parts: List[str] = []
        current: Optional[ast.AST] = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))

    @staticmethod
    def _enclosing_name(tree: ast.AST, target: ast.AST) -> Optional[str]:
        # AST nodes do not carry parents; locate the narrowest named node containing target.
        candidates: List[Tuple[int, str]] = []
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ) and target in ast.walk(node):
                candidates.append((getattr(node, "lineno", 0), node.name))
        return max(candidates, default=(0, ""))[1] or None
