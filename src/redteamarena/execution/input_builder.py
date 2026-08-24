from __future__ import annotations

import base64
from typing import Any, Dict, List

from ..artifacts import BuiltArtifact
from ..models import AgentDetection, Scenario


class InputBuilder:
    """Construct target input exclusively from the target_visible scenario branch."""

    def build(
        self,
        scenario: Scenario,
        agent: AgentDetection,
        artifacts: List[BuiltArtifact],
    ) -> Dict[str, Any]:
        visible = scenario.target_visible
        sections = [f"USER TASK\n{visible.user_task}"]
        if visible.task_relevant_tools:
            sections.append(
                "TOOLS AUTHORIZED FOR THIS TASK\n" + ", ".join(visible.task_relevant_tools)
            )
        if visible.context:
            sections.append("CONTEXT\n" + self._plain_context(visible.context))
        attachments: List[Dict[str, Any]] = []
        for built in artifacts:
            spec = built.spec
            if spec.type == "text" or spec.type == "pdf" and not agent.capabilities.pdf_input:
                sections.append(
                    f'<external_document filename="{spec.filename}">\n{spec.content}\n</external_document>'
                )
            else:
                attachments.append(self._attachment(built))
                sections.append(f"Attached {spec.type}: {spec.filename}")
        sections.append(
            "AVAILABLE TOOL SCHEMAS\n"
            + self._plain_context([tool.provider_schema() for tool in agent.tools])
        )
        return {
            "messages": [{"role": "user", "content": "\n\n".join(sections)}],
            "tools": [tool.provider_schema() for tool in agent.tools],
            "artifacts": attachments,
        }

    @staticmethod
    def _attachment(built: BuiltArtifact) -> Dict[str, Any]:
        mime = "application/pdf" if built.spec.type == "pdf" else "image/png"
        return {
            "filename": built.spec.filename,
            "type": built.spec.type,
            "mime_type": mime,
            "data_base64": base64.b64encode(built.path.read_bytes()).decode("ascii"),
        }

    @staticmethod
    def _plain_context(value: Any) -> str:
        import json

        return json.dumps(value, indent=2, sort_keys=True)
