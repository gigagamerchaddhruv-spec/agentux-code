# Agentux

**A terminal-native, provider-agnostic AI coding agent.**

Agentux is an independent open-source agent that lives in your terminal. Give it a
task, and it will inspect your project, read and edit files, run commands, recover
from failures, and tell you what it did.

Agentux is the product. The model provider is infrastructure — Agentux talks to any
server that speaks the OpenAI-compatible API, so you can point it at OpenAI, Ollama,
LiteLLM, vLLM, LM Studio, Qwen, or your own gateway without changing the agent.

```bash
agentux "fix the failing tests"
```

## Features

- **Real structured tool calling.** Agentux uses the provider's native
  function-calling mechanism. It does not ask a model to print JSON and pretend the
  JSON was a tool call — if a model or provider cannot do tool calling, Agentux fails
  loudly rather than pretending. This includes detecting local models that return
  tool calls as plain text.
- **A genuine agent loop.** Model → tool call → validate → execute → result → model,
  for as many steps as the task needs, bounded by a configurable ceiling.
- **Filesystem tools** for reading, writing, targeted edits, directory listing,
  globbing, moving, and deleting — all confined to the workspace root.
- **Shell tool** with timeouts, working-directory support, and captured
  stdout/stderr/exit code.
- **Safety layer.** Destructive commands are shown to you and require explicit
  confirmation. Read-only commands never prompt.
- **Project awareness.** A dedicated tool maps the repository — language, build
  system, source and test layout — without reading the whole tree.
- **Provider-agnostic by construction.** The agent loop contains no HTTP and no
  provider-specific logic.

## Installation

```bash
git clone https://github.com/agentux/agentux.git
cd agentux
pip install -e .
```

Requires Python 3.10+. Dependencies are deliberately few: `httpx`, `pydantic`,
`rich`. No agent framework is required to build an agent.

Verify the install:

```bash
agentux --version
agentux --help
python -m agentux --help
```

## Configuration

Agentux is configured entirely through environment variables. **No credentials are
stored in the source, read from a config file, or printed in logs.**

| Variable | Description | Default |
|---|---|---|
| `AGENTUX_API_KEY` | API key for your provider | required for remote, optional for local |
| `AGENTUX_BASE_URL` | API root, ending in `/v1` | `https://api.openai.com/v1` |
| `AGENTUX_MODEL` | Model name | `gpt-4o-mini` |
| `AGENTUX_TEMPERATURE` | Sampling temperature | `0.0` |
| `AGENTUX_MAX_TOKENS` | Max tokens per response | `4096` |
| `AGENTUX_MAX_STEPS` | Max agent iterations per task | `25` |
| `AGENTUX_TIMEOUT` | Request/command timeout (seconds) | `120` |
| `AGENTUX_CWD` | Workspace root | current directory |

```bash
export AGENTUX_API_KEY="..."
export AGENTUX_BASE_URL="https://example.com/v1"
export AGENTUX_MODEL="some-model"
```

`AGENTUX_API_KEY` is required for remote providers. When `AGENTUX_BASE_URL` points
at localhost, Agentux fills in a placeholder and runs without one — local servers
don't authenticate.

### Provider setup

Because everything is OpenAI-compatible, switching providers is a two-line change.

#### Ollama (local)

```bash
ollama serve                      # if it is not already running
ollama pull granite4.1:3b         # or any tool-capable model

export AGENTUX_BASE_URL="http://localhost:11434/v1"
export AGENTUX_MODEL="granite4.1:3b"
agentux "explain this project"    # no API key needed
```

Find out what is available, and whether it can call tools:

```bash
agentux --list-models
```

```
  granite4.1:3b                            ✓ tools
  qwen2.5-coder:7b                         ✓ tools
  smollm:135m                              ✗ no tools
  gpt-oss:20b                              ✓ tools (thinking)

18 model(s) available.
Models marked 'no tools' cannot run Agentux's tools; use --no-tools with those.
```

Agentux asks Ollama's native endpoint for capabilities, because Ollama's
OpenAI-compatible `/v1/models` does not report them.

**Not every model that claims a `tools` capability actually emits native tool
calls.** Some return `{"name": ..., "arguments": ...}` as ordinary text. Agentux
detects that and refuses to run it:

```
qwen2.5-coder:7b returned a tool call as plain text instead of using native tool
calling.

Agentux will not run tool calls that were printed as text: parsing prose as a
command is how an agent runs something it was never asked to run.
```

That is a deliberate refusal, not a bug. Emulating tool calls by parsing prose
would mean any JSON-shaped text in a model's answer could execute a command, so
Agentux reports the mismatch and lets you pick a different model.

Two more Ollama notes:

- **Timeouts.** A cold model can take a while to load. Raise the timeout for
  large models: `AGENTUX_TIMEOUT=600`.
- **Thinking models** (`gpt-oss`, `qwen3`, `gemma4`) emit a separate `reasoning`
  field, which Agentux ignores and keeps out of the conversation.

#### OpenAI

```bash
export AGENTUX_API_KEY="sk-..."
export AGENTUX_BASE_URL="https://api.openai.com/v1"
export AGENTUX_MODEL="gpt-4o-mini"
```

#### LiteLLM / vLLM / LM Studio / OpenRouter / Together / Groq

```bash
export AGENTUX_API_KEY="..."
export AGENTUX_BASE_URL="http://localhost:4000/v1"    # or your gateway
export AGENTUX_MODEL="your/model-name"
```

If you get an authentication error, check the key; a 404 usually means the model
name is wrong or `AGENTUX_BASE_URL` is missing its `/v1` suffix. The CLI says which.

## Usage

### One-shot

```bash
agentux "inspect this project and explain its architecture"
agentux "fix the failing tests"
agentux "add a README section explaining installation"
```

### Interactive

```bash
agentux
```

```
Agentux 0.1.0
Model: gpt-4o-mini  |  Workspace: /path/to/project
Tools: filesystem, project, shell

❯ what does this repo do?
```

Conversation history persists across turns. `/exit` quits, `/reset` clears history,
`/tools` lists available tools.

### Options

| Flag | Meaning |
|---|---|
| `--model MODEL` | Override `AGENTUX_MODEL` |
| `--base-url URL` | Override `AGENTUX_BASE_URL` |
| `--api-key KEY` | Override the API key (prefer the environment variable) |
| `--list-models` | List the provider's models and whether they support tools |
| `--cwd PATH` | Workspace root for file and shell tools |
| `--max-steps N` | Maximum agent iterations before giving up |
| `--timeout S` | Shell command timeout in seconds |
| `--no-tools` | Plain conversation, no tool access |
| `--read-only` | Filesystem tool cannot write, move, or delete |
| `--allow-destructive` | Run destructive commands without asking (**use with care**) |
| `-v, --verbose` | Show full tool arguments and output |
| `--system-prompt TEXT` | Override the built-in system prompt |
| `--version` | Print the version |

## Tools

Agentux ships three tools. Each declares a name, a description, and a JSON Schema;
the model addresses tools by name and never sees the implementation.

### `filesystem`

Operations: `ls`, `read`, `write`, `edit`, `mkdir`, `rm`, `mv`, `glob`.

```jsonc
// read a file
{"operation": "read", "path": "src/agentux/agent.py"}

// targeted edit — replaces one exact occurrence, leaves everything else alone
{"operation": "edit", "path": "README.md",
 "old_string": "Install with pip", "new_string": "Install with uv"}

// create or fully rewrite
{"operation": "write", "path": "docs/usage.md", "content": "# Usage\n"}
```

`edit` is the important one: it refuses to proceed if the target text is missing or
ambiguous, and it returns a unified diff. That makes accidental clobbering hard
rather than merely unlikely.

Every path is resolved and checked against the workspace root, so `../../etc/passwd`
and symlinks pointing outside the tree are rejected. Reads are size-capped and
directory listings are truncated, so the model gets a summary instead of a firehose.

### `shell`

Runs a command and returns stdout, stderr, and the exit code.

```jsonc
{"command": "pytest -q", "cwd": "tests", "timeout": 60}
```

Commands have a timeout; on expiry the process is killed and the model is told, so a
hang cannot stall the session. Multi-step output is truncated in the middle, keeping
both the head and the tail.

### `project`

Maps the repository without reading all of it.

```jsonc
{"action": "overview"}          // language, build system, key files, src/test dirs
{"action": "tree", "max_depth": 3}
{"action": "tracked"}           // files tracked by git
```

Start here. It tells the model what kind of project it is in and where things live,
so the agent can target its next reads instead of wandering.

## Safety behaviour

Agentux treats the safety layer as a **prompt for you, not a security boundary**. A
blacklist can never be complete — commands can be obfuscated, aliased, generated at
runtime, or run through another interpreter. What actually protects you is that
nothing destructive runs without being shown to you first.

When a command is flagged (recursive deletes, `sudo`, `mkfs`, raw disk writes,
`git reset --hard`, `git push --force`, credential destruction, `curl | sh`, fork
bombs, and similar), Agentux shows you the exact command and the reason:

```
╭─ Destructive command ─────────────────────────────╮
│                                                    │
│   rm -rf build                                     │
│                                                    │
│ Flagged as: recursive delete                       │
╰────────────────────────────────────────────────────╯
Run it? [y/N]
```

- Interactive mode: you decide.
- One-shot mode: the command is refused and the model is told not to retry it.
- Read-only commands never prompt — `ls`, `pytest`, `git status` just run.

`--allow-destructive` skips the prompt. `--read-only` makes the filesystem tool
incapable of modifying anything. Both are opt-in.

## Architecture

```
CLI
 ↓
Agent                       (the loop; no HTTP, no provider specifics)
 ↓
Provider abstraction        (BaseProvider)
 ↓
OpenAI-compatible provider  (the only HTTP in the project)
 ↓
Model API
```

```
Agent
 ↓
Tool registry
 ├── filesystem
 ├── project
 └── shell
```

The agent depends only on `BaseProvider` and `ToolRegistry`. Adding a provider means
implementing one interface; adding a tool means implementing one class and registering
it. Neither requires touching the loop.

```
src/agentux/
├── cli.py            presentation: parsing, output, confirmation
├── agent.py          the loop
├── config.py         environment configuration
├── safety.py         destructive-command detection
├── models.py         provider-neutral Message / ToolCall / ToolDefinition
├── errors.py         exception hierarchy
├── providers/        base.py, openai_compatible.py, mock.py
└── tools/            base.py, registry.py, filesystem.py, shell.py, project.py
```

## Development

```bash
git clone https://github.com/agentux/agentux.git
cd agentux
pip install -e ".[dev]"
```

### Testing

```bash
pytest -q
```

The suite needs no API key and makes no network calls — HTTP is mocked and the
provider layer is exercised against recorded responses. Async tests run via a small
hook in `tests/conftest.py`, so `pytest-asyncio` is not required.

What is covered: configuration loading and validation, missing configuration,
provider initialization, mocked provider responses, structured tool calls, multiple
tool calls per turn, multi-iteration agent loops, tool-failure recovery, filesystem
read/write/edit/glob/listing, path-escape rejection, shell execution, shell timeouts,
destructive-command confirmation, Ollama-specific response shapes and error mapping,
emulated-tool-call refusal, and CLI behavior.

### Adding a tool

```python
from agentux.tools.base import BaseTool
from agentux.errors import ToolError

class GrepTool(BaseTool):
    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return "Search file contents with a regular expression."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
            "required": ["pattern"],
        }

    async def execute(self, pattern: str, path: str = ".", **kwargs) -> str:
        # Raise ToolError on failure; the model sees the message and can adapt.
        ...
```

Register it in `build_agent` in `cli.py`. The agent loop needs no changes.

### Adding a provider

Implement `BaseProvider` (a `chat` method and `supports_tool_calling`) and add it to
`REGISTRY` in `providers/__init__.py`.

## License

MIT — see [LICENSE](LICENSE).
