from __future__ import annotations

from typing import Dict, List

from ..models import AgentDetection


class AgentDeduplicator:
    def deduplicate(self, detections: List[AgentDetection]) -> List[AgentDetection]:
        logical: Dict[tuple[str, str, str, str, str], AgentDetection] = {}
        for detection in detections:
            source = detection.call_sites[0].rsplit(":", 1)[0] if detection.call_sites else ""
            key = (
                source,
                detection.name,
                detection.provider,
                detection.model or detection.model_expression or "",
                detection.system_prompt,
            )
            if source and key in logical:
                self._merge(logical[key], detection)
            else:
                logical[key] = detection
        unique: Dict[str, AgentDetection] = {}
        for detection in logical.values():
            fingerprint = detection.compute_fingerprint()
            if fingerprint not in unique:
                unique[fingerprint] = detection
                continue
            primary = unique[fingerprint]
            self._merge(primary, detection)
        return list(unique.values())

    @staticmethod
    def _merge(primary: AgentDetection, alternate: AgentDetection) -> None:
        sites = primary.call_sites + primary.equivalent_call_sites + alternate.call_sites
        primary.equivalent_call_sites = sorted(set(sites) - set(primary.call_sites))
        primary.confidence.evidence.extend(alternate.confidence.evidence)
        primary.confidence.score = max(primary.confidence.score, alternate.confidence.score)
        capabilities = primary.capabilities.model_dump()
        alternate_capabilities = alternate.capabilities.model_dump()
        primary.capabilities = primary.capabilities.model_copy(
            update={
                name: bool(value or alternate_capabilities[name])
                for name, value in capabilities.items()
            }
        )
        tools = {tool.name: tool for tool in primary.tools}
        tools.update({tool.name: tool for tool in alternate.tools})
        primary.tools = list(tools.values())
        primary.configuration_env_vars = sorted(
            set(primary.configuration_env_vars) | set(alternate.configuration_env_vars)
        )
        primary.credential_env_vars = sorted(
            set(primary.credential_env_vars) | set(alternate.credential_env_vars)
        )
        primary.compute_fingerprint()
