# 🐦 WrenCode

A minimal agentic coding assistant in a single Python file.

Named after Harold Wren - the alias of a genius who built a superintelligent AI and operated quietly in the background.

-----

## What it is

WrenCode is a lightweight alternative to Claude Code. It runs a tool-calling agent loop locally or via API, giving an LLM the ability to read, write, and edit files, search codebases, and run shell commands - enough to autonomously navigate and modify a real project.

Under 800 lines. No framework. No dependencies beyond what the backend needs.

## Backends

|Backend       |Description                             |
|--------------|----------------------------------------|
|`mlx`         |Apple Silicon via MLX (default)         |
|`transformers`|HuggingFace Transformers (CPU/MPS/GPU)  |
|`anthropic`   |Claude via Anthropic API                |
|`openai`      |GPT models via OpenAI API               |
|`openrouter`  |Any model via OpenRouter                |
|`local`       |Local proxy via Anthropic-compatible API|

## Tools

The agent has access to six tools:

- **read** - read a file with line numbers, or list a directory
- **write** - write content to a file
- **edit** - replace a unique string in a file
- **glob** - find files by pattern, sorted by modification time
- **grep** - search files for a regex pattern (requires `rg`)
- **bash** - run a shell command with timeout and streaming output

All file operations are sandboxed to the workspace root by default.

## Installation

No installation required. Single file, standard library only (except the backend you choose).

```bash
git clone https://github.com/deburky/wrencode
cd wrencode
```

For MLX (Mac Silicon):

```bash
pip install mlx-lm
```

For Anthropic:

```bash
pip install anthropic  # not required - uses urllib directly
export ANTHROPIC_API_KEY=your_key
```

For OpenAI:

```bash
export OPENAI_API_KEY=your_key
```

For OpenRouter:

```bash
export OPENROUTER_API_KEY=your_key
```

For HuggingFace Transformers:

```bash
pip install transformers torch
```

## Usage

```bash
# Default: MLX backend
python3 wrencode.py

# Anthropic Claude
BACKEND=anthropic python3 wrencode.py

# OpenAI
BACKEND=openai MODEL=gpt-4o python3 wrencode.py

# OpenRouter
BACKEND=openrouter MODEL=anthropic/claude-3-haiku python3 wrencode.py

# HuggingFace model
BACKEND=transformers MODEL=deburky/gpt-oss-claude-code python3 wrencode.py

# Local proxy
BACKEND=local LOCAL_PORT=8082 python3 wrencode.py
```

## Slash Commands

|Command       |Description                                   |
|--------------|----------------------------------------------|
|`/help`       |Show available commands                       |
|`/c`          |Clear conversation history                    |
|`/compact`    |Summarize history to reduce context (mlx only)|
|`/q` or `exit`|Quit                                          |

## Environment Variables

|Variable                     |Default                |Description                       |
|-----------------------------|-----------------------|----------------------------------|
|`BACKEND`                    |`mlx`                  |Inference backend                 |
|`MODEL`                      |backend-dependent      |Model path or ID                  |
|`WRENCODE_WORKSPACE`         |cwd                    |Root directory for file operations|
|`WRENCODE_UNRESTRICTED_PATHS`|`0`                    |Allow paths outside workspace     |
|`MAX_TOKENS`                 |`4096`                 |Max tokens per response           |
|`MAX_READ_BYTES`             |`4MB`                  |Max file size to read             |
|`MAX_READ_LINES`             |`800`                  |Max lines returned per read       |
|`GREP_MAX_MATCHES`           |`80`                   |Max grep results                  |
|`BASH_TIMEOUT`               |`120`                  |Shell command timeout in seconds  |
|`MAX_TOOL_OUTPUT_CHARS`      |`48000`                |Max tool output before truncation |
|`GLOB_SKIP_DIRS`             |`.git,node_modules,...`|Directories to skip in glob       |
|`OPENROUTER_API_KEY`         |-                      |OpenRouter API key                |
|`OPENAI_API_KEY`             |-                      |OpenAI API key                    |
|`ANTHROPIC_API_KEY`          |-                      |Anthropic API key                 |
|`LOCAL_API_KEY`              |`local`                |Local proxy API key               |
|`LOCAL_PORT`                 |`8082`                 |Local proxy port                  |

## History

Conversation history is persisted to `.wrencode_history.json` in the same directory as the script. It is restored automatically on next launch.

To clear history: use `/c` in the session, or delete the file.

## License

MIT - Copyright 2026 Denis Burakov.
