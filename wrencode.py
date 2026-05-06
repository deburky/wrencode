#!/usr/bin/env python3
"""WrenCode — a minimal agentic coding assistant inspired by Harold Wren.

A lightweight alternative to Claude Code in a single Python file.

Supports multiple inference backends: local Apple Silicon via MLX,
HuggingFace Transformers, Anthropic, OpenAI, OpenRouter, and local proxy.
Provides a tool-calling agent loop with file read/write/edit, glob, grep,
and bash — enough to autonomously navigate and modify a codebase.

Copyright 2026 Denis Burakov. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import contextlib
import glob as globlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

# Load .env from parent directory
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# -----------------------------------------------------------------------------------------------
# Backend Configuration
# -----------------------------------------------------------------------------------------------
BACKEND = os.environ.get("BACKEND", "mlx")

if BACKEND == "openrouter":
    MODEL = os.environ.get("MODEL", "anthropic/claude-3-haiku")
    API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
    API_BASE = "https://openrouter.ai/api/v1/chat/completions"
elif BACKEND == "openai":
    MODEL = os.environ.get("MODEL", "gpt-4o-mini")
    API_KEY = os.environ.get("OPENAI_API_KEY", "")
    API_BASE = "https://api.openai.com/v1/chat/completions"
elif BACKEND == "anthropic":
    MODEL = os.environ.get("MODEL", "claude-haiku-4-5-20251001")
    API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
    API_BASE = "https://api.anthropic.com/v1/messages"
elif BACKEND == "local":
    MODEL = os.environ.get("MODEL", "gpt-oss-20b")
    API_KEY = os.environ.get("LOCAL_API_KEY", "local")
    LOCAL_PORT = os.environ.get("LOCAL_PORT", "8082")
    API_BASE = f"http://localhost:{LOCAL_PORT}/v1/messages"
elif BACKEND == "transformers":
    MODEL = os.environ.get("MODEL", "deburky/gpt-oss-claude-code")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
else:  # mlx
    MODEL = os.environ.get("MODEL", "mlx-community/Qwen2.5-3B-Instruct-4bit")
    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

# -----------------------------------------------------------------------------------------------
# Constants & Environment Variables
# -----------------------------------------------------------------------------------------------
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "4096"))
MAX_READ_BYTES = int(os.environ.get("MAX_READ_BYTES", str(4 * 1024 * 1024)))
MAX_READ_LINES = int(os.environ.get("MAX_READ_LINES", "800"))
GREP_MAX = int(os.environ.get("GREP_MAX_MATCHES", "80"))
BASH_TIMEOUT = int(os.environ.get("BASH_TIMEOUT", "120"))
MAX_OUT = int(os.environ.get("MAX_TOOL_OUTPUT_CHARS", "48000"))
_GLOB_SKIP: set[str] = {
    s
    for s in os.environ.get(
        "GLOB_SKIP_DIRS",
        ".git,node_modules,__pycache__,.venv,venv,dist,build,.mypy_cache,.pytest_cache,target",
    ).split(",")
    if s
}

# -----------------------------------------------------------------------------------------------
# Terminal Colors
# -----------------------------------------------------------------------------------------------
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
BLUE, CYAN, GREEN, YELLOW, RED = (
    "\033[34m",
    "\033[36m",
    "\033[32m",
    "\033[33m",
    "\033[31m",
)
BRIGHT_CYAN = "\033[96m"

WREN_BANNER = f"""{BRIGHT_CYAN}
\u2588\u2588     \u2588\u2588 \u2588\u2588\u2588\u2588\u2588\u2588  \u2588\u2588\u2588\u2588\u2588\u2588\u2588 \u2588\u2588\u2588    \u2588\u2588  \u2588\u2588\u2588\u2588\u2588\u2588  \u2588\u2588\u2588\u2588\u2588\u2588  \u2588\u2588\u2588\u2588\u2588\u2588  \u2588\u2588\u2588\u2588\u2588\u2588\u2588
\u2588\u2588     \u2588\u2588 \u2588\u2588   \u2588\u2588 \u2588\u2588      \u2588\u2588\u2588\u2588   \u2588\u2588 \u2588\u2588      \u2588\u2588    \u2588\u2588 \u2588\u2588   \u2588\u2588 \u2588\u2588
\u2588\u2588  \u2588  \u2588\u2588 \u2588\u2588\u2588\u2588\u2588\u2588  \u2588\u2588\u2588\u2588\u2588   \u2588\u2588 \u2588\u2588  \u2588\u2588 \u2588\u2588      \u2588\u2588    \u2588\u2588 \u2588\u2588   \u2588\u2588 \u2588\u2588\u2588\u2588\u2588
\u2588\u2588 \u2588\u2588\u2588 \u2588\u2588 \u2588\u2588   \u2588\u2588 \u2588\u2588      \u2588\u2588  \u2588\u2588 \u2588\u2588 \u2588\u2588      \u2588\u2588    \u2588\u2588 \u2588\u2588   \u2588\u2588 \u2588\u2588
 \u2588\u2588\u2588 \u2588\u2588\u2588  \u2588\u2588   \u2588\u2588 \u2588\u2588\u2588\u2588\u2588\u2588\u2588 \u2588\u2588   \u2588\u2588\u2588\u2588  \u2588\u2588\u2588\u2588\u2588\u2588  \u2588\u2588\u2588\u2588\u2588\u2588  \u2588\u2588\u2588\u2588\u2588\u2588  \u2588\u2588\u2588\u2588\u2588\u2588\u2588
{RESET}"""


# -----------------------------------------------------------------------------------------------
# Path Helpers
# -----------------------------------------------------------------------------------------------
def workspace_root() -> pathlib.Path:
    """Return the resolved workspace root path from env or cwd."""
    if w := os.environ.get("WRENCODE_WORKSPACE"):
        return pathlib.Path(w).expanduser().resolve()
    return pathlib.Path(os.getcwd()).resolve()


def paths_unrestricted() -> bool:
    """Return True if WRENCODE_UNRESTRICTED_PATHS is set to a truthy value."""
    return os.environ.get("WRENCODE_UNRESTRICTED_PATHS", "").lower() in (
        "1",
        "true",
        "yes",
    )


def resolve_tool_path(raw: Any) -> pathlib.Path:
    """Resolve a raw path argument to an absolute Path within the workspace."""
    if not raw or not str(raw).strip():
        raise ValueError("path is required")
    p = pathlib.Path(str(raw).strip()).expanduser()
    root = workspace_root()
    p = p.resolve() if p.is_absolute() else (root / p).resolve()
    if not paths_unrestricted():
        try:
            p.relative_to(root)
        except ValueError:
            raise ValueError(
                f"path {raw!r} resolves outside workspace {root} "
                f"(set WRENCODE_UNRESTRICTED_PATHS=1)"
            ) from None
    return p


# -----------------------------------------------------------------------------------------------
# Input Validation
# -----------------------------------------------------------------------------------------------
def _require_str(args: dict[str, Any], key: str) -> str:
    """Require a non-empty string value from args dict by key."""
    val = args.get(key)
    if not val or not str(val).strip():
        raise ValueError(f"'{key}' is required and must be a non-empty string")
    return str(val).strip()


def _optional_int(
    args: dict[str, Any], key: str, default: Optional[int] = None
) -> Optional[int]:
    """Return an optional integer from args dict, or default if absent."""
    val = args.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError) as e:
        raise ValueError(f"'{key}' must be an integer, got {val!r}") from e


# -----------------------------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------------------------
def read(args: dict[str, Any]) -> str:
    """Read a file with line numbers or list a directory."""
    path = resolve_tool_path(_require_str(args, "path"))
    if path.is_dir():
        entries = sorted(path.iterdir(), key=lambda e: (e.is_file(), e.name))
        return (
            "\n".join(f"  {e.name}{'/' if e.is_dir() else ''}" for e in entries)
            or "(empty)"
        )
    if not path.is_file():
        return f"error: not a file: {path}"
    size = path.stat().st_size
    if size > MAX_READ_BYTES:
        return f"error: file too large ({size} bytes, max {MAX_READ_BYTES})"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    offset = _optional_int(args, "offset", 0) or 0
    if not (0 <= offset <= len(lines)):
        return f"error: offset {offset} out of range (file has {len(lines)} lines)"
    limit_val = _optional_int(args, "limit")
    cap = min(
        limit_val
        if (args.get("limit") and limit_val is not None)
        else len(lines) - offset,
        MAX_READ_LINES,
    )
    out = "".join(
        f"{offset + i + 1:4}| {line}"
        for i, line in enumerate(lines[offset : offset + cap])
    )
    if offset + cap < len(lines):
        out += f"\n... ({len(lines) - offset - cap} more lines; use offset/limit or raise MAX_READ_LINES)"
    return out


def write(args: dict[str, Any]) -> str:
    """Write content to a file, creating parent directories as needed."""
    path = resolve_tool_path(_require_str(args, "path"))
    content = args.get("content", "")
    if not confirm(f"Write to {path!r}"):
        return "cancelled"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(content), encoding="utf-8")
    return "ok"


def edit(args: dict[str, Any]) -> str:
    """Replace a unique string in a file with a new string."""
    path = resolve_tool_path(_require_str(args, "path"))
    old = _require_str(args, "old")
    new = args.get("new", "")
    if not path.is_file():
        return f"error: not a file: {path}"
    if path.stat().st_size > MAX_READ_BYTES:
        return f"error: file too large (max {MAX_READ_BYTES} bytes)"
    text = path.read_text(encoding="utf-8", errors="replace")
    if old not in text:
        return "error: old_string not found"
    count = text.count(old)
    if not args.get("all") and count > 1:
        return f"error: old_string appears {count} times (use all=true)"
    if not confirm(f"Edit {path!r}"):
        return "cancelled"
    path.write_text(
        text.replace(old, str(new))
        if args.get("all")
        else text.replace(old, str(new), 1),
        encoding="utf-8",
    )
    return "ok"


def glob(args: dict[str, Any]) -> str:
    """Find files matching a glob pattern, sorted by modification time."""
    if "pattern" in args and "pat" not in args:
        args["pat"] = args.pop("pattern")
    pat = _require_str(args, "pat")
    base = resolve_tool_path(args.get("path", "."))
    if not base.is_dir():
        return f"error: not a directory: {base}"
    files = [
        f
        for f in globlib.glob(str(base / pat), recursive=True)
        if os.path.isfile(f) and all(p not in _GLOB_SKIP for p in pathlib.Path(f).parts)
    ]
    return "\n".join(sorted(files, key=os.path.getmtime, reverse=True)) or "none"


def grep(args: dict[str, Any]) -> str:
    """Search files for a regex pattern using ripgrep."""
    pat = _require_str(args, "pat")
    root = resolve_tool_path(args.get("path", "."))
    if not root.is_dir():
        return f"error: grep path must be a directory: {root}"
    rg = shutil.which("rg")
    grep_bin = shutil.which("grep")
    if not rg and not grep_bin:
        return "error: neither ripgrep (rg) nor grep is installed"
    cmd = (
        [rg, "-n", "--color", "never", "--no-heading", "-e", pat, "."]
        if rg
        else [grep_bin, "-R", "-n", "-I", "--", pat, "."]
    )
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "error: grep timed out (90s)"
    if proc.returncode not in (0, 1):
        return f"error: grep failed ({proc.returncode}): {(proc.stderr or '').strip()}"
    raw = proc.stdout.splitlines()
    body = "\n".join(raw[:GREP_MAX]) or "none"
    if len(raw) > GREP_MAX:
        body += f"\n... ({len(raw) - GREP_MAX} more; raise GREP_MAX_MATCHES)"
    return body


def confirm(prompt: str) -> bool:
    """Prompt the user for y/N confirmation and return True if confirmed."""
    return input(f"\n{YELLOW}⚠ {prompt} [y/N]{RESET} ").strip().lower() in ("y", "yes")


def bash(args: dict[str, Any]) -> str:
    """Run a shell command with a timeout, streaming output to the terminal."""
    cmd = _require_str(args, "cmd")
    if not confirm(f"Run: {cmd!r}"):
        return "cancelled"
    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=os.getcwd(),
    )
    output_lines: list[str] = []

    def reader() -> None:
        """Read subprocess stdout line by line and print to terminal."""
        with contextlib.suppress(Exception):
            assert proc.stdout is not None
            for line in proc.stdout:
                output_lines.append(line)
                print(f"{DIM}│ {line.rstrip()}{RESET}", flush=True)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    deadline = time.monotonic() + BASH_TIMEOUT
    timed_out = False
    while proc.poll() is None:
        if time.monotonic() > deadline:
            timed_out = True
            proc.kill()
            output_lines.append(f"\n(timed out after {BASH_TIMEOUT}s)\n")
            break
        time.sleep(0.05)
    t.join(timeout=2.0)
    if not timed_out:
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=2.0)
    return "".join(output_lines).strip() or "(empty)"


ToolFn = Callable[[dict[str, Any]], str]
ToolEntry = tuple[str, dict[str, str], ToolFn]

TOOLS: dict[str, ToolEntry] = {
    "read": (
        "Read file with line numbers, or list directory",
        {"path": "string", "offset": "number?", "limit": "number?"},
        read,
    ),
    "write": ("Write content to file", {"path": "string", "content": "string"}, write),
    "edit": (
        "Replace old with new in file",
        {"path": "string", "old": "string", "new": "string", "all": "boolean?"},
        edit,
    ),
    "glob": (
        "Find files by pattern, sorted by mtime",
        {"pat": "string", "path": "string?"},
        glob,
    ),
    "grep": (
        "Search files for regex",
        {"pat": "string", "path": "string?"},
        grep,
    ),
    "bash": ("Run shell command", {"cmd": "string"}, bash),
}


def run_tool(name: str, args: dict[str, Any]) -> str:
    """Execute a named tool with args, truncating output if it exceeds MAX_OUT."""
    try:
        result = TOOLS[name][2](args)
        if len(result) > MAX_OUT:
            result = (
                result[:MAX_OUT]
                + f"\n... [truncated {len(result) - MAX_OUT} chars; raise MAX_TOOL_OUTPUT_CHARS]"
            )
        return result
    except Exception as e:
        return f"error: {e}"


# -----------------------------------------------------------------------------------------------
# Message Formatting
# -----------------------------------------------------------------------------------------------
def flatten_content(content: Any) -> str:
    """Flatten Anthropic-style content list to plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append(block["text"])
        elif block.get("type") == "tool_use":
            parts.append(
                f'<tool_call>{{"tool": "{block["name"]}", "args": {json.dumps(block["input"])}}}</tool_call>'
            )
        elif block.get("type") == "tool_result":
            parts.append(f"Tool result: {block.get('content', '')}")
    return "\n".join(parts)


def render_markdown(text: str) -> str:
    """Render basic markdown bold syntax to terminal bold escape codes."""
    return re.sub(r"\*\*(.+?)\*\*", f"{BOLD}\\1{RESET}", text)


# -----------------------------------------------------------------------------------------------
# Token & Output Cleaning
# -----------------------------------------------------------------------------------------------
def strip_gptoss_tokens(text: str) -> str:
    """Strip GPT-OSS special tokens and channel markers from model output."""
    if "<|channel|>final<|message|>" in text:
        text = text.split("<|channel|>final<|message|>")[-1]
    return re.sub(r"<\|[^>]+\|>", "", text).strip()


def truncate_at_turn_leak(text: str) -> str:
    """Truncate text at the first sign of a leaked conversation turn marker."""
    return next(
        (
            text.split(m)[0].strip()
            for m in ("\nUser:", "\nSystem:", "\nHuman:", "\n\nUser:", "\n\nSystem:")
            if m in text
        ),
        text,
    )


def _tool_call_complete(text: str) -> int:
    """Return end index of first complete <tool_call> block, or -1."""
    start = text.find("<tool_call>")
    if start == -1:
        return -1
    end_tag = text.find("</tool_call>", start)
    if end_tag != -1:
        return end_tag + len("</tool_call>")
    brace_start = text.find("{", start)
    if brace_start == -1:
        return -1
    depth, last = 0, -1
    for i, ch in enumerate(text[brace_start:], brace_start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last = i + 1
                break
    return last


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse all <tool_call> blocks from model output into structured dicts."""
    calls: list[dict[str, Any]] = []
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL):
        with contextlib.suppress(Exception):
            d = json.loads(m.group(1))
            if d.get("tool") in TOOLS:
                calls.append(
                    {
                        "type": "tool_use",
                        "id": f"call_{len(calls)}",
                        "name": d["tool"],
                        "input": d.get("args", {}),
                    }
                )
    return calls


# -----------------------------------------------------------------------------------------------
# Anthropic Native Tools
# -----------------------------------------------------------------------------------------------
_TYPE_MAP: dict[str, str] = {
    "string": "string",
    "string?": "string",
    "number": "integer",
    "number?": "integer",
    "boolean": "boolean",
    "boolean?": "boolean",
}


def _build_anthropic_tools() -> list[dict[str, Any]]:
    """Build Anthropic-native tool definitions from the TOOLS registry."""
    result = []
    for name, (description, params, _) in TOOLS.items():
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param_name, param_type in params.items():
            properties[param_name] = {"type": _TYPE_MAP.get(param_type, "string")}
            if not param_type.endswith("?"):
                required.append(param_name)
        result.append(
            {
                "name": name,
                "description": description,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            }
        )
    return result


def _parse_anthropic_response(
    data: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Parse Anthropic API response into display text and tool_use blocks."""
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            text_parts.append(block["text"])
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "type": "tool_use",
                    "id": block["id"],
                    "name": block["name"],
                    "input": block.get("input", {}),
                }
            )
    return "\n".join(text_parts).strip(), tool_calls


def _build_openai_tools() -> list[dict[str, Any]]:
    """Build OpenAI-native function definitions from the TOOLS registry."""
    result = []
    for name, (description, params, _) in TOOLS.items():
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param_name, param_type in params.items():
            properties[param_name] = {"type": _TYPE_MAP.get(param_type, "string")}
            if not param_type.endswith("?"):
                required.append(param_name)
        result.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        )
    return result


def _parse_openai_response(
    data: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Parse OpenAI API response into display text and tool_call blocks."""
    message = data["choices"][0]["message"]
    display_text = message.get("content") or ""
    tool_calls: list[dict[str, Any]] = []
    for tc in message.get("tool_calls") or []:
        with contextlib.suppress(Exception):
            tool_calls.append(
                {
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "input": json.loads(tc["function"]["arguments"]),
                }
            )
    return display_text.strip(), tool_calls


# -----------------------------------------------------------------------------------------------
# HTTP Helper
# -----------------------------------------------------------------------------------------------
def _http_post(url: str, payload: dict[str, Any], headers: dict[str, str]) -> Any:
    """POST a JSON payload to a URL and return the parsed response."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise Exception(f"HTTP {e.code}: {e.read().decode()}") from e


# -----------------------------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------------------------
def get_response(
    messages: list[dict[str, Any]],
    system_prompt: str,
    mlx_state: Optional[tuple[Any, Any]],
) -> str:
    """Generate a response from the configured backend given the message history."""
    flat = [
        {"role": m["role"], "content": flatten_content(m["content"])} for m in messages
    ]

    # OpenAI — native function calling
    if BACKEND == "openai":
        data = _http_post(
            API_BASE,
            {
                "model": MODEL,
                "messages": [{"role": "system", "content": system_prompt}] + messages,
                "max_tokens": MAX_TOKENS,
                "temperature": 0.3,
                "tools": _build_openai_tools(),
                "tool_choice": "auto",
            },
            {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        )
        return json.dumps(data)  # return raw for agent loop to parse natively

    # OpenRouter — OpenAI-compatible chat completions (no native tools)
    if BACKEND == "openrouter":
        data = _http_post(
            API_BASE,
            {
                "model": MODEL,
                "messages": [{"role": "system", "content": system_prompt}] + flat,
                "max_tokens": MAX_TOKENS,
                "temperature": 0.3,
            },
            {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        )
        return str(data["choices"][0]["message"]["content"])

    # Anthropic — native tool use API
    if BACKEND == "anthropic":
        headers = {
            "Content-Type": "application/json",
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
        }
        data = _http_post(
            API_BASE,
            {
                "model": MODEL,
                "system": system_prompt,
                "messages": messages,
                "max_tokens": MAX_TOKENS,
                "tools": _build_anthropic_tools(),
            },
            headers,
        )
        return json.dumps(data)  # return raw for agent loop to parse natively

    # Local proxy — Anthropic messages API; tool calls returned as XML <tool_call> tags in text
    if BACKEND == "local":
        headers = {
            "Content-Type": "application/json",
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
        }
        data = _http_post(
            API_BASE,
            {
                "model": MODEL,
                "system": system_prompt,
                "messages": flat,
                "max_tokens": MAX_TOKENS,
            },
            headers,
        )
        text = "".join(
            b["text"] for b in data.get("content", []) if b.get("type") == "text"
        )
        return strip_gptoss_tokens(text)

    # Transformers (HuggingFace)
    if BACKEND == "transformers":
        model, tokenizer = mlx_state  # type: ignore[misc]
        inputs = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}] + flat,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs, max_new_tokens=MAX_TOKENS, temperature=0.3, do_sample=True
            )
        raw = tokenizer.decode(
            out_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=False
        )
        end = _tool_call_complete(raw)
        if end != -1:
            raw = raw[:end]
        return truncate_at_turn_leak(strip_gptoss_tokens(raw))

    # MLX (Apple Silicon)
    model, tokenizer = mlx_state  # type: ignore[misc]
    chat: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for m in messages:
        if c := flatten_content(m["content"]):
            chat.append({"role": m["role"], "content": c})
    prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    sampler = make_sampler(temp=0.3, top_p=0.95, min_p=0.0, min_tokens_to_keep=1)
    out = ""
    for chunk in stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, sampler=sampler
    ):
        out += chunk.text
        end = _tool_call_complete(out)
        if end != -1:
            out = out[:end]
            break
    if out.startswith(prompt):
        out = out[len(prompt) :].strip()
    return truncate_at_turn_leak(strip_gptoss_tokens(out))


# -----------------------------------------------------------------------------------------------
# History Management
# -----------------------------------------------------------------------------------------------
def history_file_path() -> str:
    """Return the history file path from env override or user-level default."""
    if p := os.environ.get("WRENCODE_HISTORY_FILE"):
        return str(pathlib.Path(p).expanduser())
    return str(pathlib.Path.home() / ".wrencode" / "history.json")


def load_history() -> list[dict[str, Any]]:
    """Load conversation history from the JSON history file."""
    history_file = history_file_path()
    if os.path.exists(history_file):
        with contextlib.suppress(Exception):
            with open(history_file) as f:
                return list(json.load(f))
    return []


def save_history(messages: list[dict[str, Any]]) -> None:
    """Persist conversation history to the JSON history file."""
    with contextlib.suppress(Exception):
        history_file = pathlib.Path(history_file_path())
        history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(history_file, "w") as f:
            json.dump(messages, f)


def _compact_via_api(history_text: str) -> str:
    """Call the active API backend to summarise history_text and return the summary."""
    summarise_prompt = (
        "Summarize this conversation in 3-5 concise bullet points, "
        "preserving any file paths, code decisions, or unresolved tasks:\n\n"
        + history_text
    )
    if BACKEND == "anthropic":
        data = _http_post(
            API_BASE,
            {
                "model": MODEL,
                "system": "You are a helpful assistant.",
                "messages": [{"role": "user", "content": summarise_prompt}],
                "max_tokens": 512,
            },
            {
                "Content-Type": "application/json",
                "x-api-key": API_KEY,
                "anthropic-version": "2023-06-01",
            },
        )
        return "\n".join(
            b["text"] for b in data.get("content", []) if b.get("type") == "text"
        ).strip()
    if BACKEND in ("openai", "openrouter"):
        data = _http_post(
            API_BASE,
            {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": summarise_prompt},
                ],
                "max_tokens": 512,
                "temperature": 0.3,
            },
            {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        )
        return (data["choices"][0]["message"].get("content") or "").strip()
    return ""


def compact_messages(
    messages: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Summarize conversation history to reduce context length."""
    if not messages:
        return messages
    history_text = "".join(
        f"{m['role']}: {flatten_content(m['content'])}\n" for m in messages
    )

    if BACKEND in ("anthropic", "openai", "openrouter"):
        summary = _compact_via_api(history_text)
    else:
        # MLX / Transformers path
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"Summarize this conversation in 3-5 bullet points:\n\n{history_text}",
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        sampler = make_sampler(temp=0.3, top_p=0.95, min_p=0.0, min_tokens_to_keep=1)
        summary = "".join(
            c.text
            for c in stream_generate(
                model, tokenizer, prompt=prompt, max_tokens=512, sampler=sampler
            )
        )
        if summary.startswith(prompt):
            summary = summary[len(prompt) :].strip()

    return [
        {"role": "user", "content": f"[Conversation summary]\n{summary}"},
        {
            "role": "assistant",
            "content": "Understood, I have the context from the summary.",
        },
    ]


# -----------------------------------------------------------------------------------------------
# Workspace & System Prompt
# -----------------------------------------------------------------------------------------------
def git_context() -> str:
    """Return a formatted git status string if inside a git repository."""
    with contextlib.suppress(Exception):
        r = subprocess.run(
            ["git", "status", "--short", "--branch"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if r.returncode == 0 and r.stdout.strip():
            return f"\nGit status:\n{r.stdout.strip()}"
    return ""


def build_system_prompt() -> str:
    """Build the system prompt with workspace context and tool definitions."""
    ws = workspace_root()
    path_rule = (
        "Paths are not restricted to the workspace."
        if paths_unrestricted()
        else "Relative paths resolve under the workspace. Absolute paths must stay inside it."
    )
    return f"""You are a helpful coding assistant with tools to interact with the file system.
Workspace root: {ws}
Process cwd: {os.getcwd()}
{path_rule}{git_context()}

IMPORTANT: You MUST use tools by formatting them exactly as shown below.

Available tools:
- read(path, offset, limit): Read a file or list a directory
- write(path, content): Write to a file
- edit(path, old, new): Replace text in a file (old must be unique unless all=true)
- glob(pat): Find files matching pattern
- grep(pat): Search for text in files
- bash(cmd): Run a shell command

To use a tool, format it EXACTLY like this:
<tool_call>{{"tool": "name", "args": {{"key": "value"}}}}</tool_call>

Examples:
<tool_call>{{"tool": "read", "args": {{"path": "file.py", "offset": 0, "limit": 20}}}}</tool_call>
<tool_call>{{"tool": "glob", "args": {{"pat": "*.py"}}}}</tool_call>

When reading a file, always pass offset and limit. When you finish a task, summarize what you changed.
CRITICAL: You MUST use tools for file operations. Never say you can't access files!"""


# -----------------------------------------------------------------------------------------------
# Agent Loop
# -----------------------------------------------------------------------------------------------
def run_agent_turn(
    messages: list[dict[str, Any]],
    system_prompt: str,
    mlx_state: Optional[tuple[Any, Any]],
) -> None:
    """Generate a response and execute any tool calls, repeating until no tools remain."""
    while True:
        print(f"{DIM}Generating...{RESET}", end="\r", flush=True)
        response_text = get_response(messages, system_prompt, mlx_state)
        print(" " * 20, end="\r")

        # Anthropic & OpenAI native tool use path
        if BACKEND in ("anthropic", "openai"):
            data = json.loads(response_text)
            if BACKEND == "anthropic":
                display_text, tool_calls = _parse_anthropic_response(data)
            else:
                display_text, tool_calls = _parse_openai_response(data)
            if display_text:
                print(f"\n{CYAN}>{RESET} {render_markdown(display_text)}")
            if BACKEND == "anthropic":
                messages.append(
                    {"role": "assistant", "content": data.get("content", [])}
                )
            else:
                messages.append(
                    data["choices"][0]["message"]
                )  # preserve tool_calls exactly
            if not tool_calls:
                break
            tool_results: list[dict[str, Any]] = []
            for tc in tool_calls:
                arg_preview = (
                    str(list(tc["input"].values())[0])[:50] if tc["input"] else ""
                )
                print(
                    f"\n{GREEN}{tc['name'].capitalize()}{RESET}({DIM}{arg_preview}{RESET})"
                )
                result = run_tool(tc["name"], tc["input"])
                lines = result.split("\n")
                preview = lines[0][:60] + (
                    f" ... +{len(lines) - 1} lines"
                    if len(lines) > 1
                    else ("..." if len(lines[0]) > 60 else "")
                )
                print(f"{DIM}⎿ {preview}{RESET}")
                if BACKEND == "anthropic":
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tc["id"],
                            "content": result,
                        }
                    )
                else:
                    tool_results.append(
                        {"role": "tool", "tool_call_id": tc["id"], "content": result}
                    )
            if BACKEND == "anthropic":
                messages.append({"role": "user", "content": tool_results})
            else:
                messages.extend(tool_results)
            continue

        # XML tool call path (mlx, transformers, openrouter, local)
        xml_tool_calls = parse_tool_calls(response_text)
        display_text = re.sub(
            r"<tool_call>.*?</tool_call>", "", response_text, flags=re.DOTALL
        ).strip()

        if display_text:
            print(f"\n{CYAN}>{RESET} {render_markdown(display_text)}")

        content_blocks: list[dict[str, Any]] = (
            [{"type": "text", "text": display_text}] if display_text else []
        )
        xml_tool_results: list[dict[str, Any]] = []
        for tc in xml_tool_calls:
            arg_preview = str(list(tc["input"].values())[0])[:50] if tc["input"] else ""
            print(
                f"\n{GREEN}{tc['name'].capitalize()}{RESET}({DIM}{arg_preview}{RESET})"
            )
            result = run_tool(tc["name"], tc["input"])
            lines = result.split("\n")
            preview = lines[0][:60] + (
                f" ... +{len(lines) - 1} lines"
                if len(lines) > 1
                else ("..." if len(lines[0]) > 60 else "")
            )
            print(f"{DIM}⎿ {preview}{RESET}")
            xml_tool_results.append(
                {"type": "tool_result", "tool_use_id": tc["id"], "content": result}
            )
            content_blocks.append(tc)

        messages.append({"role": "assistant", "content": content_blocks})
        if not xml_tool_results:
            break
        messages.append({"role": "user", "content": xml_tool_results})


# -----------------------------------------------------------------------------------------------
# Slash Commands
# -----------------------------------------------------------------------------------------------
def handle_slash_command(
    cmd: str,
    messages: list[dict[str, Any]],
    mlx_state: Optional[tuple[Any, Any]],
) -> Optional[str]:
    """Handle a slash command. Returns 'quit', 'handled', or None if not a command."""
    if cmd in {"/q", "exit"}:
        save_history(messages)
        return "quit"
    if cmd == "/c":
        messages.clear()
        save_history(messages)
        print(f"{GREEN}Cleared{RESET}")
        return "handled"
    if cmd == "/compact":
        if BACKEND in {"anthropic", "openai", "openrouter"} or (
            BACKEND in {"mlx", "transformers"} and mlx_state
        ):
            print(f"{DIM}Compacting history...{RESET}")
            model, tokenizer = mlx_state or (None, None)
            before = len(messages)
            messages[:] = compact_messages(messages, model, tokenizer)
            save_history(messages)
            print(f"{GREEN}Compacted {before} → {len(messages)} messages{RESET}")
        else:
            print(f"{YELLOW}/compact not available for backend '{BACKEND}'{RESET}")
        return "handled"
    if cmd == "/help":
        print(
            f"{DIM}/c — clear  /compact — summarize history  /q — quit{RESET}\n"
            f"{DIM}Backends: mlx | transformers | openrouter | openai | anthropic | local{RESET}"
        )
        return "handled"
    return None


# -----------------------------------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------------------------------
def load_model() -> Optional[tuple[Any, Any]]:
    """Load model for the current backend and return mlx_state (or None for API backends)."""
    if BACKEND == "mlx":
        print(f"{YELLOW}Loading model...{RESET}")
        model, tokenizer = load(MODEL)
        print(f"{GREEN}✓ Loaded: {getattr(model, 'name', MODEL)}{RESET}\n")
        return (model, tokenizer)
    if BACKEND == "transformers":
        print(f"{YELLOW}Loading model via transformers...{RESET}")
        _device = "mps" if torch.backends.mps.is_available() else "cpu"
        _tok = AutoTokenizer.from_pretrained(MODEL)
        _mdl = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, device_map=_device
        )
        print(f"{GREEN}✓ Loaded on {_device}: {MODEL}{RESET}\n")
        return (_mdl, _tok)
    if BACKEND == "local":
        print(f"{DIM}Local proxy at {API_BASE}{RESET}\n")
    elif BACKEND == "openai":
        if not API_KEY:
            print(f"{RED}OPENAI_API_KEY not set{RESET}")
            raise SystemExit(1)
        print(f"{DIM}OpenAI ({MODEL}){RESET}\n")
    elif BACKEND == "anthropic":
        if not API_KEY:
            print(f"{RED}ANTHROPIC_API_KEY not set{RESET}")
            raise SystemExit(1)
        print(f"{DIM}Anthropic ({MODEL}){RESET}\n")
    elif BACKEND == "openrouter":
        if not API_KEY:
            print(f"{RED}OPENROUTER_API_KEY not set{RESET}")
            raise SystemExit(1)
        print(f"{DIM}OpenRouter ({MODEL}){RESET}\n")
    return None


# -----------------------------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------------------------
def main() -> None:
    """Entry point — initialize the agent and run the interactive loop."""
    os.environ.setdefault("WRENCODE_WORKSPACE", str(pathlib.Path.cwd().resolve()))
    print(WREN_BANNER)
    print(f"{BOLD}wrencode{RESET} 🐦 | {DIM}{BACKEND}:{MODEL}{RESET}\n")
    mlx_state = load_model()
    system_prompt = build_system_prompt()
    messages = load_history()
    if messages:
        print(f"{DIM}Restored {len(messages)} messages{RESET}\n")

    while True:
        try:
            user_input = input(f"{BOLD}{BLUE}❯{RESET} ").strip()
            if not user_input:
                continue
            action = handle_slash_command(user_input, messages, mlx_state)
            if action == "quit":
                break
            if action == "handled":
                continue
            messages.append({"role": "user", "content": user_input})
            run_agent_turn(messages, system_prompt, mlx_state)
            save_history(messages)
        except KeyboardInterrupt:
            save_history(messages)
            print(f"\n{YELLOW}Interrupted{RESET}")
            break
        except EOFError:
            break
        except Exception as err:
            print(f"{RED}Error: {err}{RESET}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
