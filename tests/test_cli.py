"""CLI behaviour: argument parsing, wiring, and error reporting.

No network access: the provider is stubbed where a run is needed.
"""

import sys

import pytest

from agentux import __version__
from agentux.cli import build_parser, main, resolve_config
from agentux.errors import AuthenticationError
from agentux.tools.filesystem import FilesystemTool
from agentux.tools.project import ProjectTool
from agentux.tools.shell import ShellTool


def test_parser_accepts_a_prompt():
    args = build_parser().parse_args(["fix the failing tests"])
    assert args.prompt == "fix the failing tests"


def test_parser_prompt_is_optional():
    assert build_parser().parse_args([]).prompt is None


def test_parser_flags():
    args = build_parser().parse_args(
        ["--model", "m", "--base-url", "http://x/v1", "--no-tools",
         "--read-only", "--max-steps", "3", "--timeout", "5", "--cwd", "/tmp", "-v"]
    )
    assert args.model == "m"
    assert args.base_url == "http://x/v1"
    assert args.no_tools is True
    assert args.read_only is True
    assert args.max_steps == 3
    assert args.timeout == 5
    assert args.cwd == "/tmp"
    assert args.verbose is True


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["--help"])
    assert exit_info.value.code == 0
    assert "agentux" in capsys.readouterr().out


def test_version_flag_prints_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_missing_api_key_is_reported(monkeypatch, capsys):
    monkeypatch.delenv("AGENTUX_API_KEY", raising=False)
    assert main(["say hello"]) == 1
    assert "Configuration error" in capsys.readouterr().err


def test_invalid_base_url_is_reported(monkeypatch, capsys):
    monkeypatch.setenv("AGENTUX_API_KEY", "k")
    monkeypatch.setenv("AGENTUX_BASE_URL", "not-a-url")
    assert main(["hello"]) == 1
    assert "Configuration error" in capsys.readouterr().err


def test_flags_override_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTUX_API_KEY", "env-key")
    monkeypatch.setenv("AGENTUX_MODEL", "env-model")
    monkeypatch.delenv("AGENTUX_MAX_STEPS", raising=False)

    args = build_parser().parse_args(["--model", "flag-model", "--max-steps", "7", "--cwd", str(tmp_path)])
    config = resolve_config(args)

    assert config.model == "flag-model"
    assert config.max_steps == 7
    assert config.working_dir == str(tmp_path)


def test_config_falls_back_to_environment(monkeypatch):
    monkeypatch.setenv("AGENTUX_API_KEY", "env-key")
    monkeypatch.setenv("AGENTUX_MODEL", "env-model")
    config = resolve_config(build_parser().parse_args(["hello"]))
    assert config.model == "env-model"


def test_no_api_key_is_ever_printed(monkeypatch, capsys):
    """A secret in the environment must not leak into CLI output."""
    monkeypatch.setenv("AGENTUX_API_KEY", "sk-super-secret-value")
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "sk-super-secret-value" not in capsys.readouterr().out
    assert "sk-super-secret-value" not in capsys.readouterr().err


def test_build_agent_wires_tools(monkeypatch, tmp_path):
    from agentux.cli import build_agent

    monkeypatch.setenv("AGENTUX_API_KEY", "k")
    args = build_parser().parse_args(["--cwd", str(tmp_path)])
    agent = build_agent(resolve_config(args), args)

    assert isinstance(agent.registry.get("shell"), ShellTool)
    assert isinstance(agent.registry.get("filesystem"), FilesystemTool)
    assert isinstance(agent.registry.get("project"), ProjectTool)
    assert agent.provider is not None


def test_no_tools_flag_registers_nothing(monkeypatch, tmp_path):
    from agentux.cli import build_agent

    monkeypatch.setenv("AGENTUX_API_KEY", "k")
    args = build_parser().parse_args(["--no-tools", "--cwd", str(tmp_path)])
    agent = build_agent(resolve_config(args), args)

    assert len(agent.registry) == 0


def test_read_only_flag_propagates(monkeypatch, tmp_path):
    from agentux.cli import build_agent

    monkeypatch.setenv("AGENTUX_API_KEY", "k")
    args = build_parser().parse_args(["--read-only", "--cwd", str(tmp_path)])
    agent = build_agent(resolve_config(args), args)

    assert agent.registry.get("filesystem").read_only is True


def test_shell_tool_confirms_by_default(monkeypatch, tmp_path):
    from agentux.cli import build_agent

    monkeypatch.setenv("AGENTUX_API_KEY", "k")
    args = build_parser().parse_args(["--cwd", str(tmp_path)])
    agent = build_agent(resolve_config(args), args)

    assert agent.registry.get("shell").confirm is not None


def test_allow_destructive_disables_the_prompt(monkeypatch, tmp_path):
    from agentux.cli import build_agent

    monkeypatch.setenv("AGENTUX_API_KEY", "k")
    args = build_parser().parse_args(["--allow-destructive", "--cwd", str(tmp_path)])
    agent = build_agent(resolve_config(args), args)

    assert agent.registry.get("shell").allow_destructive is True


def test_single_prompt_runs_and_prints(monkeypatch, capsys):
    """End-to-end through main(), with only the model stubbed out."""
    import asyncio

    from agentux.models import Message

    monkeypatch.setenv("AGENTUX_API_KEY", "k")

    async def fake_run(self, prompt, interactive=False):
        self.history.append(Message(role="assistant", content="All done."))
        return "All done."

    monkeypatch.setattr("agentux.agent.Agent.run", fake_run)

    assert main(["do a thing"]) == 0
    assert "All done." in capsys.readouterr().out


def test_provider_errors_are_reported_and_exit_nonzero(monkeypatch, capsys):
    monkeypatch.setenv("AGENTUX_API_KEY", "k")

    async def fake_run(self, prompt, interactive=False):
        raise AuthenticationError("bad key")

    monkeypatch.setattr("agentux.agent.Agent.run", fake_run)

    assert main(["do a thing"]) == 1
    assert "Authentication failed" in capsys.readouterr().err


def test_unexpected_errors_do_not_traceback(monkeypatch, capsys):
    monkeypatch.setenv("AGENTUX_API_KEY", "k")

    async def fake_run(self, prompt, interactive=False):
        raise ValueError("something odd")

    monkeypatch.setattr("agentux.agent.Agent.run", fake_run)

    assert main(["do a thing"]) == 1
    assert "Unexpected error" in capsys.readouterr().err


def test_explain_error_covers_each_type():
    """Every error class must render without raising."""
    from agentux.cli import explain_error
    from agentux.errors import (
        AgentuxError, MaxStepsError, NetworkError, ProviderError,
        RateLimitError, SafetyError,
    )

    for exc in (
        AuthenticationError("x"), RateLimitError("x"), NetworkError("x"),
        MaxStepsError("x"), SafetyError("x"), ProviderError("x"),
        AgentuxError("x"), ValueError("x"),
    ):
        explain_error(exc)


def test_make_confirmer_denies_on_eof(monkeypatch):
    from agentux.cli import make_confirmer

    confirm = make_confirmer()
    monkeypatch.setattr("rich.console.Console.input", lambda *a, **k: (_ for _ in ()).throw(EOFError()))
    assert confirm("rm -rf /", ["recursive delete"]) is False


def test_make_confirmer_accepts_yes(monkeypatch):
    from agentux.cli import make_confirmer

    confirm = make_confirmer()
    monkeypatch.setattr("rich.console.Console.input", lambda *a, **k: "y")
    assert confirm("rm -rf /", ["recursive delete"]) is True


def test_make_confirmer_denies_on_empty(monkeypatch):
    from agentux.cli import make_confirmer

    confirm = make_confirmer()
    monkeypatch.setattr("rich.console.Console.input", lambda *a, **k: "")
    assert confirm("rm -rf /", ["recursive delete"]) is False


def test_module_entry_point_exists():
    """``python -m agentux`` must be wired up."""
    import importlib

    module = importlib.import_module("agentux.__main__")
    assert callable(module.cli)


# ------------------------------------------------------------- --list-models
def test_list_models_flag_is_parsed():
    assert build_parser().parse_args(["--list-models"]).list_models is True


def test_list_models_prints_models(monkeypatch, capsys):
    monkeypatch.setenv("AGENTUX_API_KEY", "ollama")
    monkeypatch.setenv("AGENTUX_BASE_URL", "http://localhost:11434/v1")

    async def fake_detailed(self):
        return [{"name": "granite4.1:3b", "capabilities": ["completion", "tools"]}]

    monkeypatch.setattr(
        "agentux.providers.openai_compatible.OpenAICompatibleProvider.list_models_detailed",
        fake_detailed,
    )

    assert main(["--list-models"]) == 0
    assert "granite4.1:3b" in capsys.readouterr().out


def test_list_models_warns_about_models_without_tools(monkeypatch, capsys):
    monkeypatch.setenv("AGENTUX_API_KEY", "ollama")

    async def fake_detailed(self):
        return [{"name": "smollm:135m", "capabilities": ["completion"]}]

    monkeypatch.setattr(
        "agentux.providers.openai_compatible.OpenAICompatibleProvider.list_models_detailed",
        fake_detailed,
    )

    assert main(["--list-models"]) == 0
    out = capsys.readouterr().out
    assert "smollm:135m" in out
    assert "no tools" in out


def test_list_models_reports_an_unreachable_server(monkeypatch, capsys):
    from agentux.errors import NetworkError

    monkeypatch.setenv("AGENTUX_API_KEY", "ollama")

    async def boom(self):
        raise NetworkError("Could not reach the provider. Is Ollama running? Try: ollama serve")

    monkeypatch.setattr(
        "agentux.providers.openai_compatible.OpenAICompatibleProvider.list_models_detailed",
        boom,
    )

    assert main(["--list-models"]) == 1
    assert "Network error" in capsys.readouterr().err


def test_list_models_handles_no_models(monkeypatch, capsys):
    monkeypatch.setenv("AGENTUX_API_KEY", "ollama")

    async def empty(self):
        return []

    monkeypatch.setattr(
        "agentux.providers.openai_compatible.OpenAICompatibleProvider.list_models_detailed",
        empty,
    )

    assert main(["--list-models"]) == 1


def test_local_ollama_needs_no_api_key(monkeypatch, capsys):
    """The headline Ollama workflow: two env vars, no credential."""
    monkeypatch.delenv("AGENTUX_API_KEY", raising=False)
    monkeypatch.setenv("AGENTUX_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("AGENTUX_MODEL", "granite4.1:3b")

    async def fake_detailed(self):
        return [{"name": "granite4.1:3b", "capabilities": ["completion", "tools"]}]

    monkeypatch.setattr(
        "agentux.providers.openai_compatible.OpenAICompatibleProvider.list_models_detailed",
        fake_detailed,
    )

    assert main(["--list-models"]) == 0
