from anthropic import Anthropic
from openai import OpenAI

openai_client = OpenAI()
anthropic_client = Anthropic()


def read_file(path: str) -> str:
    """Read an authorized project file."""
    raise NotImplementedError


def write_file(path: str, content: str) -> None:
    """Write an authorized project file."""
    raise NotImplementedError


def reviewer(text: str):
    return openai_client.chat.completions.create(
        model="example-review-model",
        messages=[
            {"role": "system", "content": "Review code safely."},
            {"role": "user", "content": text},
        ],
        tools=[read_file],
    )


def maintainer(text: str):
    return anthropic_client.messages.create(
        model="example-maintainer-model",
        system="Maintain code but never change protected configuration without approval.",
        messages=[{"role": "user", "content": text}],
        tools=[read_file, write_file],
        max_tokens=1024,
    )
