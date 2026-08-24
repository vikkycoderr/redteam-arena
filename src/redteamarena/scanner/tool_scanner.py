from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional, Set

from ..models import Evidence, ToolParameter, ToolSpec
from ..utils.secrets import sanitize_scanned_value

TYPE_MAP = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "dict": "object",
    "Dict": "object",
    "list": "array",
    "List": "array",
}


def _annotation_name(annotation: Optional[ast.expr]) -> str:
    if annotation is None:
        return "string"
    if isinstance(annotation, ast.Name):
        return TYPE_MAP.get(annotation.id, "string")
    if isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name):
        return TYPE_MAP.get(annotation.value.id, "string")
    return "string"


def _literal(node: Optional[ast.AST]) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None


class ToolSchemaBuilder:
    def from_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, path: str) -> ToolSpec:
        docstring = ast.get_docstring(node) or ""
        first_line = docstring.strip().splitlines()[0] if docstring.strip() else ""
        if first_line:
            description = str(sanitize_scanned_value(first_line))
            description_confidence = 0.9
        else:
            label = node.name.replace("_", " ")
            description = f"Simulated project operation: {label}."
            description_confidence = 0.45
        parameters: Dict[str, ToolParameter] = {}
        positional = list(node.args.posonlyargs) + list(node.args.args)
        defaults = [None] * (len(positional) - len(node.args.defaults)) + list(node.args.defaults)
        for argument, default_node in zip(positional, defaults):
            if argument.arg in {"self", "cls"}:
                continue
            default = sanitize_scanned_value(_literal(default_node))
            parameters[argument.arg] = ToolParameter(
                type=_annotation_name(argument.annotation),
                description=f"Value for {argument.arg.replace('_', ' ')}.",
                required=default_node is None,
                default=default,
            )
        for argument, default_node in zip(node.args.kwonlyargs, node.args.kw_defaults):
            parameters[argument.arg] = ToolParameter(
                type=_annotation_name(argument.annotation),
                description=f"Value for {argument.arg.replace('_', ' ')}.",
                required=default_node is None,
                default=sanitize_scanned_value(_literal(default_node)),
            )
        sensitive = any(
            token in node.name.lower()
            for token in (
                "write",
                "delete",
                "change",
                "set",
                "send",
                "execute",
                "secret",
                "credential",
            )
        )
        return ToolSpec(
            name=node.name,
            description=description,
            parameters=parameters,
            description_confidence=description_confidence,
            sensitive=sensitive,
            evidence=[
                Evidence(
                    kind="function_definition",
                    path=path,
                    line=node.lineno,
                    detail=f"Function {node.name} used as an agent tool",
                    points=2,
                )
            ],
        )


class ToolScanner:
    DECORATORS = {"tool", "function_tool", "register_tool", "ai_function", "kernel_function"}

    def discover(self, tree: ast.AST, relative_path: str) -> Dict[str, ToolSpec]:
        functions: Dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        explicit: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions[node.name] = node
                for decorator in node.decorator_list:
                    name = self._name(
                        decorator.func if isinstance(decorator, ast.Call) else decorator
                    )
                    if name.split(".")[-1] in self.DECORATORS:
                        explicit.add(node.name)
            elif isinstance(node, ast.Call):
                call_name = self._name(node.func)
                if call_name.endswith(("register_tool", "add_tool")):
                    for arg in node.args:
                        if isinstance(arg, ast.Name):
                            explicit.add(arg.id)
                for keyword in node.keywords:
                    if keyword.arg in {"tools", "functions"} and isinstance(
                        keyword.value, (ast.List, ast.Tuple)
                    ):
                        for item in keyword.value.elts:
                            if isinstance(item, ast.Name):
                                explicit.add(item.id)
        builder = ToolSchemaBuilder()
        return {
            name: builder.from_function(functions[name], relative_path)
            for name in sorted(explicit)
            if name in functions
        }

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
