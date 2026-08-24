from __future__ import annotations

import httpx
import pytest

from redteamarena.adapters import AdapterRequest, AnthropicAdapter, OllamaAdapter
from redteamarena.adapters.openai_compatible import OpenAICompatibleAdapter
from redteamarena.exceptions import ProviderError


def artifact(kind: str, mime: str) -> dict:
    return {
        "filename": f"sample.{kind}",
        "type": kind,
        "mime_type": mime,
        "data_base64": "ZmFrZQ==",
    }


def test_openai_compatible_serializes_image_and_pdf() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(__import__("json").loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = OpenAICompatibleAdapter("fake", client=client)
    adapter.complete(
        AdapterRequest(
            model="model",
            messages=[{"role": "user", "content": "inspect"}],
            artifacts=[artifact("image", "image/png"), artifact("pdf", "application/pdf")],
        )
    )
    content = captured["messages"][0]["content"]
    assert {block["type"] for block in content} == {"text", "image_url", "file"}


def test_openrouter_attack_model_request_structure() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["payload"] = __import__("json").loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    adapter = OpenAICompatibleAdapter(
        "user-enters-key-here",
        base_url="https://openrouter.ai/api/v1",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    adapter.complete(
        AdapterRequest(
            model="qwen/qwen3-30b-a3b-instruct-2507",
            messages=[{"role": "user", "content": "generate scenarios"}],
        )
    )

    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["authorization"] == "Bearer user-enters-key-here"
    assert captured["payload"]["model"] == "qwen/qwen3-30b-a3b-instruct-2507"


def test_openai_compatible_error_includes_safe_provider_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": "Unsupported request for sk-secret-value"}},
            request=request,
        )

    adapter = OpenAICompatibleAdapter(
        "fake", client=httpx.Client(transport=httpx.MockTransport(handler))
    )

    with pytest.raises(ProviderError, match=r"HTTP 400: Unsupported request for \[REDACTED\]"):
        adapter.complete(AdapterRequest(model="model", messages=[]))


def test_anthropic_and_ollama_serialize_supported_artifacts() -> None:
    anthropic_payload = {}
    ollama_payload = {}

    def anthropic_handler(request: httpx.Request) -> httpx.Response:
        anthropic_payload.update(__import__("json").loads(request.content))
        return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]})

    def ollama_handler(request: httpx.Request) -> httpx.Response:
        ollama_payload.update(__import__("json").loads(request.content))
        return httpx.Response(200, json={"message": {"content": "ok"}})

    request = AdapterRequest(
        model="model",
        messages=[{"role": "user", "content": "inspect"}],
        artifacts=[artifact("image", "image/png"), artifact("pdf", "application/pdf")],
    )
    AnthropicAdapter(
        "fake", client=httpx.Client(transport=httpx.MockTransport(anthropic_handler))
    ).complete(request)
    OllamaAdapter(client=httpx.Client(transport=httpx.MockTransport(ollama_handler))).complete(
        request
    )
    assert {block["type"] for block in anthropic_payload["messages"][0]["content"]} == {
        "text",
        "image",
        "document",
    }
    assert ollama_payload["messages"][0]["images"] == ["ZmFrZQ=="]
