"""Anthropic API shim for Kimi K2.5 via LiteLLM/Bedrock."""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional
from urllib.parse import unquote

import httpx
import yaml
from fastapi import FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("claude-swap-shim")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def _load_config() -> Dict[str, Any]:
    """Load shim configuration from config.yaml, falling back to defaults."""
    defaults = {
        "host": "127.0.0.1",
        "port": 4001,
        "upstream_base_url": "http://127.0.0.1:4000",
        "upstream_timeout": 600,
        "connect_timeout": 30,
        "max_continuation_retries": 3,
        "emulate_stream_for_tools": True,
        "default_model": "kimi-k2.5",
        "enable_web_search": True,
        "web_search_max_results": 5,
        "log_body": False,
        "log_tokens": False,
    }

    # Look for config.yaml in standard locations
    config_paths = [
        Path(os.getenv("SHIM_CONFIG", "")),
        Path(__file__).parent.parent.parent / "config" / "config.yaml",
        Path.cwd() / "config" / "config.yaml",
        Path.home() / ".config" / "claude-swap" / "config.yaml",
    ]

    for config_path in config_paths:
        if config_path and config_path.exists():
            try:
                with open(config_path) as f:
                    loaded = yaml.safe_load(f) or {}
                    defaults.update(loaded)
                    logger.info("Loaded config from %s", config_path)
                    return defaults
            except Exception as e:
                logger.warning("Failed to load config from %s: %s", config_path, e)

    logger.info("Using default configuration")
    return defaults


# Load configuration
_CONFIG = _load_config()

# Server settings
APP_HOST = _CONFIG.get("host", "127.0.0.1")
APP_PORT = int(_CONFIG.get("port", 4001))

# Upstream LiteLLM settings
UPSTREAM_BASE = _CONFIG.get("upstream_base_url", "http://127.0.0.1:4000").rstrip("/")
UPSTREAM_TIMEOUT_S = float(_CONFIG.get("upstream_timeout", 600))
CONNECT_TIMEOUT_S = float(_CONFIG.get("connect_timeout", 30))

# Emulate streaming for tool calls by doing upstream stream=false and re-wrapping into SSE
EMULATE_STREAM_FOR_TOOLS = bool(_CONFIG.get("emulate_stream_for_tools", True))

# Model name normalization: rewrite all model names to this value.
# Claude Code may send secondary requests (for web search, etc.) with different
# model names like "claude-haiku-4-5-20251001". Since we only have one model
# configured in LiteLLM, normalize all requests to use it.
DEFAULT_MODEL = _CONFIG.get("default_model", "kimi-k2.5")

# Enable shim-side web search execution via DuckDuckGo (when the model calls web_search)
ENABLE_WEB_SEARCH = bool(_CONFIG.get("enable_web_search", True))
WEB_SEARCH_MAX_RESULTS = int(_CONFIG.get("web_search_max_results", 5))

app = FastAPI()


def _pick_forward_headers(request_headers: Dict[str, str]) -> Dict[str, str]:
    """
    Forward headers to upstream, blocking only problematic ones.

    IMPORTANT:
    - Claude Code CLI commonly uses `x-api-key` (not Authorization: Bearer).
    """
    blocklist = {
        "host",
        "content-length",
        "transfer-encoding",
        "connection",
        "keep-alive",
        "upgrade",
    }
    out: Dict[str, str] = {}
    for k, v in request_headers.items():
        if k.lower() not in blocklist:
            out[k] = v
    out.setdefault("content-type", "application/json")
    return out


def _upstream_url_for(request: Request) -> str:
    """
    Preserve the full path + query string exactly.
    Claude Code sends /v1/messages?beta=true and expects that to behave identically upstream.
    """
    q = request.url.query
    return f"{UPSTREAM_BASE}{request.url.path}" + (f"?{q}" if q else "")


def _as_dict(body: Any) -> Dict[str, Any]:
    if isinstance(body, dict):
        return body
    raise ValueError("Request JSON body must be an object")


def _sse(event: str, data_obj: Dict[str, Any]) -> bytes:
    # Anthropic SSE format: "event: ...\ndata: {...}\n\n"
    return f"event: {event}\n" f"data: {json.dumps(data_obj, ensure_ascii=False)}\n\n".encode("utf-8")


def _normalize_tool_id(x: Any) -> Any:
    # Handle tool_use ids like " functions.bash:0" (leading space)
    if isinstance(x, str):
        return x.strip()
    return x


def _normalize_tool_response(msg: Dict[str, Any], request_tools: Optional[list] = None) -> Dict[str, Any]:
    """
    Normalize tool_use blocks in the response to match Claude Code expectations:
    1. Strip whitespace from tool IDs
    2. Map tool names back to original case from request (Kimi returns lowercase)
    """
    if not isinstance(msg, dict):
        return msg

    # Build a case-insensitive map from request tools: lowercase -> original name
    tool_name_map: Dict[str, str] = {}
    if request_tools:
        for tool in request_tools:
            if isinstance(tool, dict) and "name" in tool:
                original_name = tool["name"]
                tool_name_map[original_name.lower()] = original_name

    content = msg.get("content")
    if not isinstance(content, list):
        return msg

    normalized_content = []
    for block in content:
        if not isinstance(block, dict):
            normalized_content.append(block)
            continue

        if block.get("type") == "tool_use":
            normalized_block = dict(block)
            # Normalize tool ID
            if "id" in normalized_block:
                normalized_block["id"] = _normalize_tool_id(normalized_block["id"])
            # Map tool name back to original case
            if "name" in normalized_block and tool_name_map:
                lower_name = normalized_block["name"].lower()
                if lower_name in tool_name_map:
                    normalized_block["name"] = tool_name_map[lower_name]
            normalized_content.append(normalized_block)
        else:
            normalized_content.append(block)

    result = dict(msg)
    result["content"] = normalized_content
    return result


# ---------------------------------------------------------------------------
# Server-side tool handling
# ---------------------------------------------------------------------------
# Anthropic server-side tools (web_search_20250305, etc.) use a different type
# than regular "custom" tools. LiteLLM/Kimi can't handle these, so we convert
# them to regular function tools and optionally execute them in the shim.

_WEB_SEARCH_TOOL_SCHEMA = {
    "name": "web_search",
    "description": (
        "Search the web for current information. "
        "Returns search results with titles, URLs, and descriptions. "
        "Use this when you need up-to-date information or to verify facts."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            }
        },
        "required": ["query"],
    },
}


def _convert_server_tools(tools: list) -> tuple[list, set[str]]:
    """
    Convert Anthropic server-side tools to regular function tools.

    Returns (converted_tools, server_tool_names).
    Server tool names are tracked so we can intercept their execution later.
    """
    converted: list = []
    server_tool_names: set[str] = set()

    for tool in tools:
        if not isinstance(tool, dict):
            converted.append(tool)
            continue

        tool_type = tool.get("type", "")

        # Server-side web search tool (type like "web_search_20250305")
        if isinstance(tool_type, str) and tool_type.startswith("web_search"):
            converted.append(_WEB_SEARCH_TOOL_SCHEMA)
            server_tool_names.add("web_search")
            logger.info("Converted server-side tool %r → regular function tool", tool_type)
            continue

        # Other unknown server-side tool types — strip them
        if tool_type and tool_type != "custom" and not tool.get("input_schema"):
            logger.info("Stripping unsupported server-side tool: type=%s name=%s", tool_type, tool.get("name"))
            continue

        converted.append(tool)

    return converted, server_tool_names


# ---------------------------------------------------------------------------
# Web search execution (DuckDuckGo)
# ---------------------------------------------------------------------------

async def _execute_web_search(query: str, max_results: int = WEB_SEARCH_MAX_RESULTS) -> list[dict]:
    """Execute a web search using DuckDuckGo lite and return results."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(15.0),
            follow_redirects=True,
        ) as client:
            r = await client.post(
                "https://lite.duckduckgo.com/lite/",
                data={"q": query},
                headers=headers,
            )

        if r.status_code != 200:
            logger.warning("DuckDuckGo search failed: status=%s", r.status_code)
            return []

        return _parse_ddg_lite(r.text, max_results)

    except Exception as e:
        logger.error("Web search error: %s", e)
        return []


def _parse_ddg_lite(html: str, max_results: int) -> list[dict]:
    """Parse DuckDuckGo lite HTML for search results."""
    results: list[dict] = []

    # DuckDuckGo lite puts results as links with class "result-link"
    link_re = re.compile(
        r'<a\s+rel="nofollow"\s+href="([^"]+)"\s+class=["\']result-link["\'][^>]*>(.*?)</a>',
        re.DOTALL,
    )
    snippet_re = re.compile(
        r'<td\s+class=["\']result-snippet["\'][^>]*>(.*?)</td>',
        re.DOTALL,
    )

    links = link_re.findall(html)
    snippets = snippet_re.findall(html)

    for i, (url, raw_title) in enumerate(links):
        if i >= max_results:
            break

        title = re.sub(r"<[^>]+>", "", raw_title).strip()
        snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip() if i < len(snippets) else ""

        # DuckDuckGo lite wraps URLs in a redirect — extract the actual URL
        uddg = re.search(r"[?&]uddg=([^&]+)", url)
        actual_url = unquote(uddg.group(1)) if uddg else url

        results.append({
            "type": "web_search_result",
            "url": actual_url,
            "title": title,
            "description": snippet,
        })

    logger.info("Web search for %r: %d results", html[:60] if not results else results[0].get("title", "?"), len(results))
    return results


async def _execute_server_tool_calls(
    response_msg: dict,
    server_tool_names: set[str],
    original_body: dict,
    upstream_url: str,
    headers: dict,
) -> dict:
    """
    If the model's response includes tool calls for server-side tools (web_search),
    execute them in the shim and feed results back for another model turn.

    Returns the final response message (with search results incorporated).
    """
    content = response_msg.get("content")
    if not isinstance(content, list):
        return response_msg

    # Find server-side tool calls
    server_calls = [
        block for block in content
        if isinstance(block, dict)
        and block.get("type") == "tool_use"
        and block.get("name") in server_tool_names
    ]

    if not server_calls:
        return response_msg

    logger.info("Intercepting %d server-side tool call(s)", len(server_calls))

    # Execute each server tool call
    tool_result_blocks: list[dict] = []
    server_tool_blocks: list[dict] = []
    web_search_requests = 0
    for call in server_calls:
        name = call.get("name")
        inp = call.get("input") or {}
        tid = call.get("id")

        if name == "web_search" and ENABLE_WEB_SEARCH:
            query = inp.get("query", "")
            results = await _execute_web_search(query)
            web_search_requests += 1

            server_tool_blocks.append(
                {
                    "type": "server_tool_use",
                    "id": tid,
                    "name": "web_search",
                    "input": {"query": query},
                }
            )
            server_tool_blocks.append(
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": tid,
                    "content": results,
                }
            )

            result_text = "\n\n".join(
                f"[{r['title']}]({r['url']})\n{r.get('description', '')}"
                for r in results
            ) if results else "No results found."
            tool_result_blocks.append({
                "type": "tool_result",
                "tool_use_id": tid,
                "content": result_text,
            })
        else:
            # Unknown server tool — return empty result
            tool_result_blocks.append({
                "type": "tool_result",
                "tool_use_id": tid,
                "content": f"Server-side tool '{name}' is not supported.",
            })

    # Build follow-up request: original messages + assistant response + tool results
    follow_up_messages = list(original_body.get("messages", []))
    follow_up_messages.append({"role": "assistant", "content": content})
    follow_up_messages.append({"role": "user", "content": tool_result_blocks})

    follow_up_body = dict(original_body)
    follow_up_body["messages"] = follow_up_messages
    follow_up_body["stream"] = False

    timeout = httpx.Timeout(UPSTREAM_TIMEOUT_S, connect=CONNECT_TIMEOUT_S)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(upstream_url, headers=headers, json=follow_up_body)

        if r.status_code >= 400:
            logger.warning("Server-tool follow-up failed: status=%s", r.status_code)
            return _attach_server_tool_metadata(response_msg, server_tool_blocks, web_search_requests)

        return _attach_server_tool_metadata(r.json(), server_tool_blocks, web_search_requests)
    except Exception as e:
        logger.error("Server-tool follow-up error: %s", e)
        return _attach_server_tool_metadata(response_msg, server_tool_blocks, web_search_requests)


def _attach_server_tool_metadata(
    msg: dict,
    server_tool_blocks: list[dict],
    web_search_requests: int,
) -> dict:
    """
    Add Anthropic server-tool metadata to the final assistant message.

    Claude Code uses these blocks/counters to report web-search activity.
    """
    if not isinstance(msg, dict):
        return msg

    out = dict(msg)
    content = out.get("content")
    if not isinstance(content, list):
        content = []

    if server_tool_blocks:
        existing_keys = {
            (
                b.get("type"),
                b.get("id") if b.get("type") == "server_tool_use" else b.get("tool_use_id"),
            )
            for b in content
            if isinstance(b, dict)
        }
        new_blocks = [
            b
            for b in server_tool_blocks
            if (
                b.get("type"),
                b.get("id") if b.get("type") == "server_tool_use" else b.get("tool_use_id"),
            )
            not in existing_keys
        ]
        out["content"] = new_blocks + content if new_blocks else content

    if web_search_requests > 0:
        usage = out.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        server_tool_use = usage.get("server_tool_use")
        if not isinstance(server_tool_use, dict):
            server_tool_use = {}
        try:
            existing_count = int(server_tool_use.get("web_search_requests", 0))
        except Exception:
            existing_count = 0
        server_tool_use["web_search_requests"] = max(existing_count, web_search_requests)
        usage["server_tool_use"] = server_tool_use
        out["usage"] = usage

    return out


_THINKING_TYPES = {"thinking", "redacted_thinking"}


def _normalize_thinking_blocks(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize thinking/redacted_thinking blocks to Anthropic format.

    Kimi may put thinking text in the "text" field; Anthropic expects it
    in the "thinking" field. This ensures blocks match the Anthropic schema
    so Claude Code can display them.
    """
    if not isinstance(msg, dict):
        return msg
    content = msg.get("content")
    if not isinstance(content, list) or not content:
        return msg

    normalized = []
    for block in content:
        if not isinstance(block, dict):
            normalized.append(block)
            continue
        btype = block.get("type")
        if btype == "thinking":
            # Ensure thinking text is in the "thinking" field (Anthropic format)
            thinking_text = block.get("thinking") or block.get("text") or ""
            normalized.append({"type": "thinking", "thinking": thinking_text})
        elif btype == "redacted_thinking":
            # Pass through as-is (Anthropic format already)
            normalized.append(block)
        else:
            normalized.append(block)

    result = dict(msg)
    result["content"] = normalized
    return result


# Kimi/OpenAI stop_reason → Anthropic stop_reason
_STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    # Anthropic-native values pass through
    "end_turn": "end_turn",
    "max_tokens": "max_tokens",
    "tool_use": "tool_use",
    "stop_sequence": "stop_sequence",
}

# Maximum retries for incomplete responses (kimi-k2.5 sometimes stalls mid-generation)
# Configured in config.yaml
_MAX_CONTINUATION_RETRIES = int(_CONFIG.get("max_continuation_retries", 3))


def _normalize_stop_reason(raw: Any, content: list | None = None) -> str:
    """
    Map upstream stop_reason to Anthropic-compatible value.

    Falls back to inference from content blocks if raw is None.
    """
    if raw and isinstance(raw, str):
        mapped = _STOP_REASON_MAP.get(raw)
        if mapped:
            return mapped
        logger.warning("Unknown stop_reason=%r, defaulting to end_turn", raw)
        return "end_turn"

    # raw is None or falsy — infer from content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") in {"tool_use", "server_tool_use"}:
                return "tool_use"
    return "end_turn"


def _is_message_incomplete(msg: Dict[str, Any]) -> bool:
    """
    Detect if a response appears incomplete (kimi-k2.5 stalled mid-generation).

    Returns True if:
    - stop_reason is None (model stopped without indicating completion)
    - Content ends mid-thinking block (thinking with no end marker)
    - Content ends mid-tool-use (partial JSON input)
    """
    if not isinstance(msg, dict):
        return False

    stop_reason = msg.get("stop_reason")
    content = msg.get("content")

    # If stop_reason is explicitly set, response is complete
    if stop_reason:
        return False

    # No stop_reason - check if content looks complete
    if isinstance(content, list) and content:
        last_block = content[-1]
        if not isinstance(last_block, dict):
            return False

        btype = last_block.get("type")

        # Mid-thinking: thinking block exists but no text or sentinel
        if btype == "thinking":
            # Check if thinking looks truncated (ends abruptly without punctuation)
            thinking_text = last_block.get("thinking") or last_block.get("text") or ""
            if thinking_text and not thinking_text.rstrip().endswith((".", "!", "?", "`", ")", "}")):
                logger.info("Detected incomplete thinking block")
                return True

        # Mid-tool-use: tool_use with partial/empty input might indicate truncation
        if btype == "tool_use":
            inp = last_block.get("input")
            if inp is None or (isinstance(inp, dict) and not inp):
                logger.info("Detected potentially incomplete tool_use block")
                return True

    # No stop_reason with non-empty content suggests incomplete generation
    if content:
        logger.info("Detected incomplete response: no stop_reason with content present")
        return True

    return False


def _build_continuation_request(original_body: Dict[str, Any], accumulated_content: list) -> Dict[str, Any]:
    """
    Build a follow-up request to continue an incomplete generation.

    Appends the assistant's partial response as a message, then prompts with 'continue'.
    """
    new_body = dict(original_body)
    messages = list(new_body.get("messages", []))

    # Add the assistant's partial response
    messages.append({"role": "assistant", "content": accumulated_content})

    # Add continue prompt
    messages.append({"role": "user", "content": "continue"})

    new_body["messages"] = messages
    return new_body


def _emulate_anthropic_sse_from_message(msg: Dict[str, Any], request_tools: Optional[list] = None) -> AsyncIterator[bytes]:
    """
    Convert a non-stream /v1/messages JSON response into an Anthropic SSE stream.

    We emit:
      message_start
      content_block_start (+ optional content_block_delta for text)
      content_block_stop
      message_delta (stop_reason + usage)
      message_stop
    """
    # Normalize tool responses first
    msg = _normalize_tool_response(msg, request_tools)
    # Normalize thinking blocks to Anthropic format
    msg = _normalize_thinking_blocks(msg)

    # Build a minimal message skeleton for message_start (content empty).
    message_start = {
        "type": "message_start",
        "message": {
            "id": msg.get("id"),
            "type": msg.get("type", "message"),
            "role": msg.get("role", "assistant"),
            "content": [],
            "model": msg.get("model"),
            "stop_reason": None,
            "stop_sequence": None,
            "usage": msg.get("usage") or {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        },
    }

    async def gen() -> AsyncIterator[bytes]:
        yield _sse("message_start", message_start)

        content = msg.get("content") or []
        if not isinstance(content, list):
            content = []

        for i, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            btype = block.get("type")

            if btype == "text":
                # Start block with empty text, then one delta with full text, then stop.
                yield _sse(
                    "content_block_start",
                    {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}},
                )
                text = block.get("text") or ""
                if text:
                    yield _sse(
                        "content_block_delta",
                        {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": text}},
                    )
                yield _sse("content_block_stop", {"type": "content_block_stop", "index": i})

            elif btype == "tool_use":
                # Anthropic streaming protocol:
                #   content_block_start → input is always empty {}
                #   content_block_delta → input_json_delta with actual JSON
                #   content_block_stop
                # Claude Code reads input ONLY from deltas, not from start.
                tool_block: dict[str, Any] = {
                    "type": "tool_use",
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input": {},
                }
                yield _sse(
                    "content_block_start",
                    {"type": "content_block_start", "index": i, "content_block": tool_block},
                )
                # Send the actual input as an input_json_delta
                tool_input = block.get("input") or {}
                input_json_str = json.dumps(tool_input, ensure_ascii=False)
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": i,
                        "delta": {"type": "input_json_delta", "partial_json": input_json_str},
                    },
                )
                yield _sse("content_block_stop", {"type": "content_block_stop", "index": i})

            elif btype == "server_tool_use":
                # Anthropic server-tool block emitted directly as a content block.
                yield _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": i,
                        "content_block": {
                            "type": "server_tool_use",
                            "id": block.get("id"),
                            "name": block.get("name"),
                            "input": block.get("input") or {},
                        },
                    },
                )
                yield _sse("content_block_stop", {"type": "content_block_stop", "index": i})

            elif btype == "web_search_tool_result":
                # Anthropic web-search result block.
                yield _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": i,
                        "content_block": {
                            "type": "web_search_tool_result",
                            "tool_use_id": block.get("tool_use_id"),
                            "content": block.get("content") or [],
                        },
                    },
                )
                yield _sse("content_block_stop", {"type": "content_block_stop", "index": i})

            elif btype == "thinking":
                # Anthropic streaming protocol for thinking blocks:
                #   content_block_start → {"type": "thinking", "thinking": ""}
                #   content_block_delta → thinking_delta with actual text
                #   content_block_stop
                yield _sse(
                    "content_block_start",
                    {"type": "content_block_start", "index": i, "content_block": {"type": "thinking", "thinking": ""}},
                )
                thinking_text = block.get("thinking") or ""
                if thinking_text:
                    yield _sse(
                        "content_block_delta",
                        {"type": "content_block_delta", "index": i, "delta": {"type": "thinking_delta", "thinking": thinking_text}},
                    )
                yield _sse("content_block_stop", {"type": "content_block_stop", "index": i})

            else:
                logger.warning("Skipping unknown content block type=%s (index=%d)", btype, i)
                continue

        # message_delta + stop
        stop_reason = _normalize_stop_reason(msg.get("stop_reason"), msg.get("content"))
        usage = msg.get("usage") or {"input_tokens": 0, "output_tokens": 0}
        yield _sse(
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": stop_reason}, "usage": usage},
        )
        yield _sse("message_stop", {"type": "message_stop"})

    return gen()


async def _proxy_json(
    url: str,
    headers: Dict[str, str],
    json_body: Dict[str, Any],
    server_tool_names: set[str] | None = None,
) -> Response:
    timeout = httpx.Timeout(UPSTREAM_TIMEOUT_S, connect=CONNECT_TIMEOUT_S)
    request_tools = json_body.get("tools") if isinstance(json_body, dict) else None

    body = dict(json_body)
    accumulated_msg: Dict[str, Any] | None = None
    accumulated_usage: Dict[str, Any] = {"input_tokens": 0, "output_tokens": 0}

    for attempt in range(_MAX_CONTINUATION_RETRIES + 1):
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, headers=headers, json=body)
            content_type = r.headers.get("content-type", "application/json")
            logger.info(
                "_proxy_json attempt=%d/%d status=%s",
                attempt + 1, _MAX_CONTINUATION_RETRIES + 1, r.status_code
            )

            # Normalize tool responses for successful JSON responses
            if r.status_code < 400 and "application/json" in content_type:
                try:
                    msg = r.json()
                    # Execute server-side tool calls (web_search, etc.) if present
                    if server_tool_names:
                        msg = await _execute_server_tool_calls(
                            msg, server_tool_names, body, url, headers,
                        )

                    # Merge accumulated content if this is a continuation
                    if accumulated_msg is not None:
                        prev_content = accumulated_msg.get("content") or []
                        new_content = msg.get("content") or []
                        if isinstance(prev_content, list) and isinstance(new_content, list):
                            msg["content"] = prev_content + new_content
                        # Merge usage
                        usage = msg.get("usage") or {}
                        accumulated_usage["input_tokens"] += usage.get("input_tokens", 0)
                        accumulated_usage["output_tokens"] += usage.get("output_tokens", 0)
                        msg["usage"] = accumulated_usage
                    else:
                        # First attempt - capture initial usage
                        usage = msg.get("usage") or {}
                        accumulated_usage = {
                            "input_tokens": usage.get("input_tokens", 0),
                            "output_tokens": usage.get("output_tokens", 0),
                        }

                    # Check if response is incomplete and we should continue
                    if _is_message_incomplete(msg) and attempt < _MAX_CONTINUATION_RETRIES:
                        logger.info("Detected incomplete response in _proxy_json, continuing (attempt %d/%d)",
                                   attempt + 1, _MAX_CONTINUATION_RETRIES)
                        accumulated_msg = msg
                        # Build continuation request
                        body = _build_continuation_request(json_body, msg.get("content", []))
                        continue

                    # Response is complete or exhausted retries - normalize and return
                    normalized = _normalize_tool_response(msg, request_tools)
                    normalized = _normalize_thinking_blocks(normalized)
                    # Normalize stop_reason for non-streaming responses too
                    if "stop_reason" in normalized:
                        normalized["stop_reason"] = _normalize_stop_reason(
                            normalized.get("stop_reason"), normalized.get("content")
                        )
                    return Response(
                        content=json.dumps(normalized, ensure_ascii=False).encode("utf-8"),
                        status_code=r.status_code,
                        media_type=content_type,
                    )
                except Exception:
                    pass  # Fall through to return raw response

            # Error or non-JSON response - don't retry
            return Response(content=r.content, status_code=r.status_code, media_type=content_type)

    # Should not reach here, but just in case
    return Response(content=b"{}", status_code=200, media_type="application/json")


async def _proxy_stream_passthrough(url: str, headers: Dict[str, str], json_body: Dict[str, Any]) -> StreamingResponse:
    """
    Pass-through upstream streaming bytes.
    """
    timeout = httpx.Timeout(UPSTREAM_TIMEOUT_S, connect=CONNECT_TIMEOUT_S)

    async def gen() -> AsyncIterator[bytes]:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, headers=headers, json=json_body) as r:
                ct = (r.headers.get("content-type") or "").lower()
                logger.info("Upstream stream status=%s content-type=%s url=%s", r.status_code, ct, url)
                async for chunk in r.aiter_raw():
                    if chunk:
                        yield chunk

    return StreamingResponse(gen(), media_type="text/event-stream")


async def _proxy_stream_emulated_from_nonstream(
    url: str,
    headers: Dict[str, str],
    json_body: Dict[str, Any],
    server_tool_names: set[str] | None = None,
) -> StreamingResponse:
    """
    Option A:
      - Send upstream request with stream=false
      - Execute any server-side tool calls (web_search, etc.)
      - Convert the final JSON response into Anthropic SSE events
    """
    timeout = httpx.Timeout(UPSTREAM_TIMEOUT_S, connect=CONNECT_TIMEOUT_S)
    request_tools = json_body.get("tools") if isinstance(json_body, dict) else None

    async def gen() -> AsyncIterator[bytes]:
        # Force stream=false upstream
        body2 = dict(json_body)
        body2["stream"] = False

        accumulated_msg: Dict[str, Any] | None = None
        accumulated_usage: Dict[str, Any] = {"input_tokens": 0, "output_tokens": 0}

        for attempt in range(_MAX_CONTINUATION_RETRIES + 1):
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(url, headers=headers, json=body2)
                ct = (r.headers.get("content-type") or "").lower()
                logger.info(
                    "Upstream nonstream-for-emulated attempt=%d/%d status=%s content-type=%s url=%s",
                    attempt + 1, _MAX_CONTINUATION_RETRIES + 1, r.status_code, ct, url
                )

                if r.status_code >= 400:
                    # Emit an SSE "error" frame with the upstream body (best-effort).
                    try:
                        payload = r.json()
                    except Exception:
                        payload = {"raw": r.text}
                    yield _sse("error", {"type": "error", "error": payload})
                    yield _sse("message_stop", {"type": "message_stop"})
                    return

                try:
                    msg = r.json()
                except Exception:
                    # If upstream didn't return JSON, emit as an error.
                    yield _sse("error", {"type": "error", "error": {"raw": r.text}})
                    yield _sse("message_stop", {"type": "message_stop"})
                    return

            # Execute server-side tool calls (web_search, etc.)
            if server_tool_names:
                msg = await _execute_server_tool_calls(
                    msg, server_tool_names, body2, url, headers,
                )

            # Merge accumulated content if this is a continuation
            if accumulated_msg is not None:
                prev_content = accumulated_msg.get("content") or []
                new_content = msg.get("content") or []
                if isinstance(prev_content, list) and isinstance(new_content, list):
                    msg["content"] = prev_content + new_content
                # Merge usage
                usage = msg.get("usage") or {}
                accumulated_usage["input_tokens"] += usage.get("input_tokens", 0)
                accumulated_usage["output_tokens"] += usage.get("output_tokens", 0)
                msg["usage"] = accumulated_usage
            else:
                # First attempt - capture initial usage
                usage = msg.get("usage") or {}
                accumulated_usage = {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                }

            # Check if response is incomplete and we should continue
            if _is_message_incomplete(msg) and attempt < _MAX_CONTINUATION_RETRIES:
                logger.info("Detected incomplete response, will retry with continuation (attempt %d/%d)",
                           attempt + 1, _MAX_CONTINUATION_RETRIES)
                accumulated_msg = msg
                # Build continuation request
                body2 = _build_continuation_request(json_body, msg.get("content", []))
                body2["stream"] = False
                continue

            # Response is complete or we've exhausted retries
            accumulated_msg = msg
            break

        # Log upstream response metadata
        final_msg = accumulated_msg or {}
        content_types = [b.get("type") for b in (final_msg.get("content") or []) if isinstance(b, dict)]
        logger.info(
            "Upstream response: stop_reason=%s content_types=%s output_tokens=%s (after %d attempts)",
            final_msg.get("stop_reason"),
            content_types,
            (final_msg.get("usage") or {}).get("output_tokens"),
            attempt + 1,
        )

        async for ev in _emulate_anthropic_sse_from_message(final_msg, request_tools):
            yield ev

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "service": "claude-swap-shim",
            "upstream": UPSTREAM_BASE,
            "stream_emulation": {
                "enabled_for_tools": EMULATE_STREAM_FOR_TOOLS,
            },
        }
    )


@app.post("/v1/messages")
async def messages(
    request: Request,
    anthropic_version: Optional[str] = Header(default=None),
    anthropic_beta: Optional[str] = Header(default=None),
):
    body = _as_dict(await request.json())

    # --- Model name normalization ---
    # Claude Code may send requests with model names the shim doesn't know
    # (e.g., "claude-haiku-4-5-20251001" for secondary web search processing).
    # Normalize to the one model we have configured.
    requested_model = body.get("model", "")
    if requested_model != DEFAULT_MODEL:
        body["model"] = DEFAULT_MODEL
        if requested_model:
            logger.info("Model rewritten: %s → %s", requested_model, DEFAULT_MODEL)

    # --- Server-side tool conversion ---
    # Anthropic server-side tools (type "web_search_20250305", etc.) can't be
    # forwarded to Kimi via LiteLLM.  Convert them to regular function tools.
    server_tool_names: set[str] = set()
    if isinstance(body.get("tools"), list) and body["tools"]:
        body["tools"], server_tool_names = _convert_server_tools(body["tools"])

    stream = bool(body.get("stream", False))
    tools_present = isinstance(body.get("tools"), list) and len(body.get("tools") or []) > 0

    upstream_url = _upstream_url_for(request)
    headers = _pick_forward_headers(dict(request.headers))

    logger.info(
        "Incoming %s stream=%s tools=%s model=%s server_tools=%s",
        str(request.url.path) + (("?" + request.url.query) if request.url.query else ""),
        stream,
        tools_present,
        body.get("model"),
        server_tool_names or None,
    )

    if _CONFIG.get("log_body", False):
        logger.info("Body: %s", json.dumps(body, indent=2)[:20000])

    if stream:
        # Option A: if tools are present, emulate stream using a non-stream upstream call.
        if EMULATE_STREAM_FOR_TOOLS and tools_present:
            return await _proxy_stream_emulated_from_nonstream(
                upstream_url, headers=headers, json_body=body, server_tool_names=server_tool_names,
            )
        # Otherwise, pass-through streaming.
        return await _proxy_stream_passthrough(upstream_url, headers=headers, json_body=body)

    return await _proxy_json(upstream_url, headers=headers, json_body=body, server_tool_names=server_tool_names)


@app.post("/v1/messages/count_tokens")
async def count_tokens(
    request: Request,
    anthropic_version: Optional[str] = Header(default=None),
    anthropic_beta: Optional[str] = Header(default=None),
):
    body = _as_dict(await request.json())

    # Normalize model name
    if body.get("model") and body["model"] != DEFAULT_MODEL:
        body["model"] = DEFAULT_MODEL

    # Strip server-side tools from count_tokens too
    if isinstance(body.get("tools"), list) and body["tools"]:
        body["tools"], _ = _convert_server_tools(body["tools"])

    if _CONFIG.get("log_tokens", False):
        logger.info("COUNT_TOKENS REQUEST: %s", json.dumps(body, indent=2)[:20000])
    else:
        logger.info("COUNT_TOKENS model=%s tools=%s", body.get("model"), len(body.get("tools") or []))

    headers = _pick_forward_headers(dict(request.headers))
    upstream_url = _upstream_url_for(request)
    return await _proxy_json(upstream_url, headers=headers, json_body=body)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"ok": True})


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    """
    Return available models for Claude Code.

    Exposes kimi-k2.5 as the primary model, plus aliases for common
    Anthropic model names that we transparently rewrite to kimi-k2.5.
    """
    models = [
        {
            "id": DEFAULT_MODEL,  # kimi-k2.5
            "object": "model",
            "created": 1700000000,
            "owned_by": "moonshot-ai",
        },
        # Aliases for transparent compatibility
        {
            "id": "claude-opus-4",
            "object": "model",
            "created": 1700000000,
            "owned_by": "anthropic",
        },
        {
            "id": "claude-sonnet-4",
            "object": "model",
            "created": 1700000000,
            "owned_by": "anthropic",
        },
        {
            "id": "claude-3-opus-20240229",
            "object": "model",
            "created": 1700000000,
            "owned_by": "anthropic",
        },
        {
            "id": "claude-3-sonnet-20240229",
            "object": "model",
            "created": 1700000000,
            "owned_by": "anthropic",
        },
    ]
    return JSONResponse({"object": "list", "data": models})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "claude_swap.shim:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=False,
        log_level=os.getenv("UVICORN_LOG_LEVEL", "info"),
    )
