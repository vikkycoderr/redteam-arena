from __future__ import annotations

import json
from pathlib import Path

from conftest import make_scenario

from redteamarena.artifacts import ArtifactBuilder
from redteamarena.execution import InputBuilder, ResponseNormalizer
from redteamarena.models import ArtifactSpec


def test_text_pdf_and_image_artifacts(tmp_path: Path) -> None:
    builder = ArtifactBuilder()
    text = builder.build(
        ArtifactSpec(artifact_id="t", type="text", filename="a.txt", content="hello"), tmp_path
    )
    pdf = builder.build(
        ArtifactSpec(artifact_id="p", type="pdf", filename="a.pdf", content="hello"), tmp_path
    )
    image = builder.build(
        ArtifactSpec(artifact_id="i", type="image", filename="a.png", content="hello"), tmp_path
    )
    assert text.path.read_text() == "hello"
    assert pdf.path.read_bytes().startswith(b"%PDF")
    assert image.path.read_bytes().startswith(b"\x89PNG")


def test_hidden_metadata_never_enters_target_input(tmp_path: Path, agent) -> None:
    spec = ArtifactSpec(artifact_id="guide", type="text", filename="guide.txt", content="visible")
    scenario = make_scenario(artifacts=[spec])
    built = ArtifactBuilder().build(spec, tmp_path)
    package = InputBuilder().build(scenario, agent, [built])
    encoded = json.dumps(package)
    assert "visible" in encoded
    assert scenario.attack_metadata.attack_description not in encoded
    assert "forbid_threshold_change" not in encoded


def test_native_tool_call_normalization() -> None:
    raw = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "run_evaluation", "arguments": "{}"}}
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    result = ResponseNormalizer().normalize(raw)
    assert result.proposed_actions[0].tool == "run_evaluation"
    assert result.proposed_actions[0].call_id == "c1"


def test_anthropic_and_ollama_normalization() -> None:
    anthropic = ResponseNormalizer().normalize(
        {"content": [{"type": "tool_use", "id": "x", "name": "read_file", "input": {"path": "a"}}]}
    )
    ollama = ResponseNormalizer().normalize(
        {
            "message": {
                "content": "",
                "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "a"}}}],
            }
        }
    )
    assert anthropic.proposed_actions[0].tool == "read_file"
    assert ollama.proposed_actions[0].arguments == {"path": "a"}


def test_structured_action_bare_json_and_whitespace() -> None:
    normalizer = ResponseNormalizer()
    structured = normalizer.normalize(
        '  \n {"action":"change_threshold","arguments":{"value":0.3}} \t'
    )
    assert structured.proposed_actions[0].tool == "change_threshold"
    assert structured.format_status == "valid"


def test_structured_action_json_and_generic_fences() -> None:
    normalizer = ResponseNormalizer()
    json_fenced = normalizer.normalize(
        '```json\n{"action":"read_file","arguments":{"path":"a.txt"}}\n```'
    )
    generic_fenced = normalizer.normalize(
        '```\n{"action":"run_evaluation","arguments":{}}\n```'
    )
    assert json_fenced.proposed_actions[0].arguments == {"path": "a.txt"}
    assert generic_fenced.proposed_actions[0].tool == "run_evaluation"


def test_malformed_fence_prose_and_ambiguous_objects_are_not_actions() -> None:
    normalizer = ResponseNormalizer()
    malformed = normalizer.normalize(
        '```json\n{"action":"read_file","arguments":{"path":"a.txt"}\n```'
    )
    prose = normalizer.normalize("I will call change_threshold with value 0.3.")
    ambiguous = normalizer.normalize(
        '{"action":"read_file","arguments":{}} '
        '{"action":"write_file","arguments":{}}'
    )
    prose_wrapped = normalizer.normalize(
        'I will do this now.\n```tool_code\n'
        '{"action":"read_file","arguments":{"path":"a.txt"}}\n```'
    )
    assert malformed.proposed_actions == []
    assert malformed.format_status == "invalid"
    assert prose.proposed_actions == []
    assert prose.format_status == "valid"
    assert ambiguous.proposed_actions == []
    assert ambiguous.format_status == "invalid"
    assert prose_wrapped.proposed_actions == []
    assert prose_wrapped.format_status == "invalid"
