# Agentux

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org)
[![Tests](https://img.shields.io/badge/tests-519%20passing-brightgreen.svg)](https://github.com/agentux/agentux/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/ollama-supported-orange.svg)](https://ollama.com)

```
 █████╗  ██████╗ ███████╗███╗   ██╗████████╗██╗   ██╗██╗  ██╗
██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝██║   ██║╚██╗██╔╝
███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ██║   ██║ ╚███╔╝
██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ██║   ██║ ██╔██╗
██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ╚██████╔╝██╔╝ ██╗
╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
```

A terminal-native, provider-agnostic AI coding agent with an exquisite developer experience.

Agentux is an independent AI coding agent built in Python with a keyboard-driven terminal user interface powered by `prompt_toolkit` and `rich`. Give it a task and it will inspect your project, safely read and edit files, run shell commands, recover from failures, search technical documentation on the live web, render terminal graphics, inspect Git state, and keep you in total control.

The model provider is interchangeable infrastructure. Agentux speaks the OpenAI-compatible chat completions API, so it works with OpenAI, local Ollama, LiteLLM, vLLM, LM Studio, or your own proxy gateway.

---

## Features

**Agentux Solo.** A 0.5B-parameter autonomous research agent with a persistent SQLite knowledge store. It reads relevant memories every turn, researches factual questions on the web when its store is empty, saves what it learns, and answers from memory on repeat questions. The research cycle is enforced deterministically rather than left to the model's discretion.

- `agentux --solo` — run Solo in the interactive TUI
- `agentux --solo-engine "<question>"` — one-shot autonomous research cycle
- `agentux-solo` — dedicated Solo entry point
- Memory store at `~/.agentux/knowledge.db`, overridable with `AGENTUX_SOLO_MEMORY`

**Deterministic research enforcement.** The engine does not trust a small model to remember to look things up. It searches memory first, nudges the model when it tries to answer without researching, and auto-saves web results so the cycle completes even when the model cannot emit a well-formed tool call.

**Semantic code navigation.** LSP-backed `go_to_definition`, `find_references`, and `get_symbol_info` work across Python, Go, TypeScript, JavaScript, Rust, C/C++, and Java. Servers are started lazily per project and cached.

**Shadow checkpointing and time travel.** Every file-modifying operation is preceded by a snapshot. `/undo` restores the exact pre-turn state; `/checkpoints` browses and rolls back any recorded state.

**AST repository mapping.** A token-efficient symbol map that fits an entire codebase into a strict token budget, giving the model architectural awareness without blowing the context window.

**Persistent sessions.** Save, restore, and branch conversations across terminal restarts with `/save`, `/load`, and `/sessions`.

**Specialist subagents.** Hierarchical delegation to `researcher`, `coder`, `reviewer`, and `architect` roles, each with a scoped tool set.

**Self-healing test loop.** `/test` detects the project's runner; `/fix` runs tests and repairs failures autonomously.

**MCP and skills.** A stdio JSON-RPC 2.0 MCP client plus an autonomous skills engine with task decomposition, reflection, and a community marketplace.

**Local-model optimizations.** `--lean` cuts turn latency from 40s to 1.1s on small models. `--text-tools` lets quantized models execute tools through Markdown/XML when native tool calling is limited. Context compaction kicks in past 30 messages.

---

## Quick Start

```bash
pip install -e .
```

OpenAI or any compatible proxy:
```bash
export AGENTUX_API_KEY="sk-..."
export AGENTUX_MODEL="gpt-4o"
agentux
```

Local Ollama (no API key needed):
```bash
export AGENTUX_BASE_URL="http://localhost:11434/v1"
export AGENTUX_MODEL="qwen2.5-coder:14b"
agentux
```

Single task:
```bash
agentux "inspect tests and fix any failing assertions"
```

Agentux Solo:
```bash
agentux --solo
agentux --solo-engine "What is the capital of France?"
agentux-solo "Who invented the transistor?"
```

---

## Solo Research Cycle

```
RETRIEVE  -> search the knowledge database
IDENTIFY  -> is the stored knowledge sufficient?
RESEARCH  -> if not, use the web tools
VERIFY    -> check claims against a second source where possible
SAVE       -> store what was learned
VERIFY     -> read it back to confirm it was saved
ANSWER
```

The engine enforces this sequence deterministically. It inspects tool calls and injects a nudge when the model tries to skip a step. Defaults: 180s timeout, 12 iterations, 3 research attempts, 30 tool calls.

---

## Slash Commands

| Command | Description |
|---|---|
| `/help` | List all available commands and shortcuts |
| `/subagent <role> <task>` | Delegate to a specialist (`researcher`, `coder`, `reviewer`, `architect`) |
| `/undo` / `/revert` | Roll back file modifications from the last turn or checkpoint |
| `/checkpoints` | List recorded shadow checkpoints and snapshots |
| `/repomap [query]` | Generate a compact AST repository symbol map |
| `/save [name]` | Save the conversation session to disk |
| `/load <name>` | Restore a saved conversation session |
| `/sessions` | List all saved conversation sessions |
| `/test [cmd]` | Run the test suite with automatic runner detection |
| `/fix [cmd]` | Run tests and trigger autonomous self-healing on failure |
| `/model [name]` | List available models or switch the active model |
| `/mode [chat\|build\|plan]` | Switch operation mode |
| `/set [key] [value]` | Show or update configuration settings |
| `/tools` | List all active tools (built-in and MCP) |
| `/mcp` | Inspect or manage Model Context Protocol servers |
| `/marketplace` | Browse and install skills from the marketplace |
| `/context` | Inspect token usage and context metrics |
| `/status` | View git branch, workspace, and working tree status |
| `/diff` | View staged and unstaged git diffs |
| `/commit [msg]` | Stage changes and create a git commit |
| `/init` | Scan the workspace and generate `AGENTUX.md` |
| `/compact` | Compact conversation history to minimize tokens |
| `/search <query>` | Perform a live technical web search |
| `/image <path>` | Render an image in the terminal via Chafa |
| `/reset` | Clear conversation history and reset the session |
| `/exit` | Quit Agentux |

---

## Built-in Tools

`filesystem` (read, write, surgical `edit` with exact-match diff preview) · `shell` (timeout handling, safety gating) · `project` (stack detection, file-tree exploration) · `search` (regex content search) · `todo` (interactive task checklist) · `git` (status, branch, diffs, commit log, commits) · `outline_symbols` (AST symbol extraction for Python, Go, TypeScript, JavaScript, Rust) · `web_search` (DuckDuckGo, Tavily, or Brave) · `fetch_web_page` (HTML-to-markdown scraper) · `view_image` (terminal graphics via Chafa) · `go_to_definition` / `find_references` / `get_symbol_info` (LSP) · `memory_search` / `memory_save` (Solo memory) · MCP tool passthrough

---

## Architecture

```
agentux (CLI / TUI)
  ├── agent.py        multi-step tool-using conversation loop
  ├── solo_engine.py  deterministic research cycle enforcement
  ├── knowledge.py    SQLite knowledge store (knowledge, sources, research_history)
  ├── tui.py          prompt_toolkit + rich full-screen interface
  ├── checkpoint.py   shadow snapshotting and rollback
  ├── repomap.py      AST repository mapping
  ├── subagents/      specialist delegation manager
  ├── skills/         reasoning, self-improvement, marketplace
  ├── mcp.py          stdio JSON-RPC 2.0 MCP client
  └── tools/          19 built-in tools + LSP + MCP passthrough
```

---

## Testing

519 unit and integration tests covering the Solo research cycle, shadow checkpointing, AST repomapping, sessions, self-healing workflows, safety gating, providers, MCP, and LSP:

```bash
pytest
```

---

## License

MIT