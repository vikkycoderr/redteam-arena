from __future__ import annotations

from ..models import AgentDetection, Capabilities


class CapabilityDetector:
    MODEL_HINTS = {
        "vision": (False, True),
        "vlm": (False, True),
        "gpt-4o": (True, True),
        "claude-3": (True, True),
        "gemini": (True, True),
    }

    def refine(self, agent: AgentDetection) -> Capabilities:
        capabilities = agent.capabilities.model_copy(deep=True)
        model = (agent.model or "").lower()
        for hint, (pdf_input, image_input) in self.MODEL_HINTS.items():
            if hint in model:
                capabilities.pdf_input = capabilities.pdf_input or pdf_input
                capabilities.image_input = capabilities.image_input or image_input
        capabilities.text_input = True
        return capabilities
