from __future__ import annotations

import ast
from pathlib import Path

from redteamarena.models import ConfidenceLevel
from redteamarena.profiling import ApplicationProfiler
from redteamarena.scanner import AgentDeduplicator, RepositoryScanner, ToolSchemaBuilder


def test_ast_detects_direct_sdk_call(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(
        "from openai import OpenAI\nclient=OpenAI()\n"
        "client.chat.completions.create(model='gpt-test', messages=[])\n",
        encoding="utf-8",
    )
    scan = RepositoryScanner().scan(tmp_path)
    assert len(scan.detections) == 1
    assert scan.detections[0].model == "gpt-test"
    assert scan.detections[0].provider == "openai_compatible"
    assert scan.detections[0].confidence.level == ConfidenceLevel.HIGH


def test_factory_client_is_traced(tmp_path: Path) -> None:
    (tmp_path / "factory.py").write_text(
        "from anthropic import Anthropic\n"
        "def make_client():\n    return Anthropic()\n"
        "client=make_client()\n"
        "client.messages.create(model='claude-test', messages=[], max_tokens=10)\n",
        encoding="utf-8",
    )
    detection = RepositoryScanner().scan(tmp_path).detections[0]
    assert detection.provider == "anthropic"
    assert detection.model == "claude-test"


def test_multi_agent_and_different_prompts_remain_separate(tmp_path: Path) -> None:
    (tmp_path / "agents.py").write_text(
        "from openai import OpenAI\nc=OpenAI()\n"
        "c.chat.completions.create(model='same', messages=[{'role':'system','content':'A'}])\n"
        "c.chat.completions.create(model='same', messages=[{'role':'system','content':'B'}])\n",
        encoding="utf-8",
    )
    detections = RepositoryScanner().scan(tmp_path).detections
    assert len(detections) == 2
    assert {item.system_prompt for item in detections} == {"A", "B"}


def test_identical_configuration_deduplicates(agent) -> None:
    first = agent.model_copy(deep=True)
    second = agent.model_copy(deep=True)
    first.call_sites = ["a.py:1"]
    second.call_sites = ["b.py:2"]
    second.agent_id = "second"
    result = AgentDeduplicator().deduplicate([first, second])
    assert len(result) == 1
    assert result[0].equivalent_call_sites == ["b.py:2"]


def test_low_confidence_is_reported_but_not_eligible(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("openai\n", encoding="utf-8")
    scan = RepositoryScanner().scan(tmp_path)
    assert len(scan.detections) == 1
    assert scan.detections[0].confidence.level == ConfidenceLevel.LOW
    assert not scan.detections[0].eligible


def test_versioned_virtual_environment_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('project file')\n", encoding="utf-8")
    dependency = tmp_path / ".venv312" / "lib" / "python3.12" / "site-packages"
    dependency.mkdir(parents=True)
    (dependency / "provider.py").write_text(
        "from openai import OpenAI\n"
        "OpenAI().chat.completions.create(model='not-a-target', messages=[])\n",
        encoding="utf-8",
    )

    scan = RepositoryScanner().scan(tmp_path)

    assert scan.files_scanned == 1
    assert scan.detections == []
    profile = ApplicationProfiler().profile(tmp_path, scan)
    assert profile.important_files == ["app.py"]
    assert profile.important_operations == []


def test_openrouter_environment_is_linked_to_openai_compatible_target(tmp_path: Path) -> None:
    (tmp_path / ".env.example").write_text(
        "OPENROUTER_API_KEY=\nOPENROUTER_BASE_URL=https://openrouter.ai/api/v1\n"
        "MINIFIX_MODEL=example/model\n",
        encoding="utf-8",
    )
    (tmp_path / "agent.py").write_text(
        "from openai import OpenAI\n"
        "def call(settings):\n"
        "    return OpenAI().chat.completions.create(model=settings.model, messages=[])\n",
        encoding="utf-8",
    )

    detection = RepositoryScanner().scan(tmp_path).detections[0]

    assert detection.credential_env_vars == ["OPENROUTER_API_KEY"]
    assert detection.configuration_env_vars == ["MINIFIX_MODEL"]


def test_imported_literal_tool_schema_collection_is_linked(tmp_path: Path) -> None:
    (tmp_path / "schemas.py").write_text(
        "READ_TOOL = {'type': 'function', 'function': {"
        "'name': 'read_file', 'description': 'Read a file', "
        "'parameters': {'type': 'object', 'properties': {"
        "'path': {'type': 'string'}}, 'required': ['path']}}}\n"
        "AGENT_TOOLS = [READ_TOOL]\n",
        encoding="utf-8",
    )
    (tmp_path / "agent.py").write_text(
        "from openai import OpenAI\nfrom schemas import AGENT_TOOLS\n"
        "OpenAI().chat.completions.create("
        "model='target-model', messages=[], tools=AGENT_TOOLS)\n",
        encoding="utf-8",
    )

    detection = RepositoryScanner().scan(tmp_path).detections[0]

    assert [tool.name for tool in detection.tools] == ["read_file"]
    assert detection.tools[0].parameters["path"].required


def test_alternate_protocol_branches_in_one_method_are_one_agent(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(
        "from openai import OpenAI\n"
        "def call_model(client, model):\n"
        "    if supports_tools():\n"
        "        return client.chat.completions.create("
        "model=model, messages=[], tools=[read_file])\n"
        "    return client.chat.completions.create("
        "model=model, messages=[], response_format={'type': 'json_object'})\n"
        "def read_file(path: str):\n"
        "    pass\n",
        encoding="utf-8",
    )

    detections = RepositoryScanner().scan(tmp_path).detections

    assert len(detections) == 1
    assert len(detections[0].equivalent_call_sites) == 1
    assert detections[0].capabilities.native_tool_calling
    assert detections[0].capabilities.structured_output


def test_typer_cli_runner_invoke_is_not_an_llm_call(tmp_path: Path) -> None:
    (tmp_path / "test_cli_and_scanner.py").write_text(
        "from typer.testing import CliRunner\n"
        "from application.cli import app\n"
        "runner = CliRunner()\n"
        "def test_cli_init_reset_show_config():\n"
        "    initialized = runner.invoke(app, ['init'])\n"
        "    reset = runner.invoke(app, ['reset'])\n"
        "    shown = runner.invoke(app, ['show-config'])\n",
        encoding="utf-8",
    )

    scan = RepositoryScanner().scan(tmp_path)

    assert scan.detections == []


def test_tool_schema_with_and_without_docstring() -> None:
    tree = ast.parse(
        "def described(path: str, count: int = 2):\n"
        '    """Read a safe path."""\n    pass\n'
        "def change_threshold(value: float):\n    pass\n"
    )
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    builder = ToolSchemaBuilder()
    described = builder.from_function(functions[0], "tools.py")
    neutral = builder.from_function(functions[1], "tools.py")
    assert described.description == "Read a safe path."
    assert described.parameters["count"].required is False
    assert neutral.description == "Simulated project operation: change threshold."
    assert neutral.description_confidence < described.description_confidence


def test_unlinked_configuration_is_retained_with_direct_detection(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("openai\n", encoding="utf-8")
    (tmp_path / "agent.py").write_text(
        "from openai import OpenAI\nc=OpenAI()\n"
        "c.chat.completions.create(model='linked-model', messages=[])\n",
        encoding="utf-8",
    )
    (tmp_path / "config.yaml").write_text(
        "provider: openai\nmodel: unlinked-model\n", encoding="utf-8"
    )
    detections = RepositoryScanner().scan(tmp_path).detections
    assert {item.model for item in detections} == {"linked-model", "unlinked-model"}
    unlinked = next(item for item in detections if item.model == "unlinked-model")
    assert unlinked.confidence.level == ConfidenceLevel.LOW
    assert not unlinked.eligible


def test_existing_json_tool_schema_permissions_and_prompt_redaction(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(
        "from openai import OpenAI\nc=OpenAI()\n"
        "TOOLS=[{'type':'function','function':{'name':'lookup','description':'Look up data',"
        "'parameters':{'type':'object','properties':{'query':{'type':'string'}},"
        "'required':['query']}}}]\n"
        "c.chat.completions.create(model='model', system='api_key=sk-abcdefghijklmnop',"
        "messages=[], tools=TOOLS, readable_paths=['docs/'], writable_paths=['output/'],"
        "protected_resources=['config/'])\n",
        encoding="utf-8",
    )
    detection = RepositoryScanner().scan(tmp_path).detections[0]
    assert detection.tools[0].name == "lookup"
    assert detection.tools[0].parameters["query"].required
    assert detection.readable_paths == ["docs/"]
    assert detection.writable_paths == ["output/"]
    assert detection.protected_resources == ["config/"]
    assert "sk-abcdefghijklmnop" not in detection.system_prompt
