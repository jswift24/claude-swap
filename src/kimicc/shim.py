"""Anthropic API shim for Kimi K2.5 via LiteLLM/Bedrock."""
from __future__ import annotations

import json
import logging
import os
from typing import Any, AsyncIterator, Dict, Optional

import httpx
from fastapi import FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("kimicc-shim")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

APP_HOST = os.getenv("SHIM_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("SHIM_PORT", "4001"))

UPSTREAM_BASE = os.getenv("LITELLM_BASE_URL", "http://127.0.0.1:4000").rstrip("/")
UPSTREAM_TIMEOUT_S = float(os.getenv("UPSTREAM_TIMEOUT_S", "600"))

# Emulate streaming for tool calls by doing upstream stream=false and re-wrapping into SSE
EMULATE_STREAM_FOR_TOOLS = os.getenv("SHIM_EMULATE_STREAM_FOR_TOOLS", "1") == "1"

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
            if isinstance(block, dict) and block.get("type") == "tool_use":
                return "tool_use"
    return "end_turn"


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
                tool_block = {
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


async def _proxy_json(url: str, headers: Dict[str, str], json_body: Dict[str, Any]) -> Response:
    timeout = httpx.Timeout(UPSTREAM_TIMEOUT_S, connect=30.0)
    request_tools = json_body.get("tools") if isinstance(json_body, dict) else None
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, headers=headers, json=json_body)
        content_type = r.headers.get("content-type", "application/json")

        # Normalize tool responses for successful JSON responses
        if r.status_code < 400 and "application/json" in content_type:
            try:
                msg = r.json()
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

        return Response(content=r.content, status_code=r.status_code, media_type=content_type)


async def _proxy_stream_passthrough(url: str, headers: Dict[str, str], json_body: Dict[str, Any]) -> StreamingResponse:
    """
    Pass-through upstream streaming bytes.
    """
    timeout = httpx.Timeout(UPSTREAM_TIMEOUT_S, connect=30.0)

    async def gen() -> AsyncIterator[bytes]:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, headers=headers, json=json_body) as r:
                ct = (r.headers.get("content-type") or "").lower()
                logger.info("Upstream stream status=%s content-type=%s url=%s", r.status_code, ct, url)
                async for chunk in r.aiter_raw():
                    if chunk:
                        yield chunk

    return StreamingResponse(gen(), media_type="text/event-stream")


async def _proxy_stream_emulated_from_nonstream(url: str, headers: Dict[str, str], json_body: Dict[str, Any]) -> StreamingResponse:
    """
    Option A:
      - Send upstream request with stream=false
      - Convert the single JSON response into Anthropic SSE events
    """
    timeout = httpx.Timeout(UPSTREAM_TIMEOUT_S, connect=30.0)
    request_tools = json_body.get("tools") if isinstance(json_body, dict) else None

    async def gen() -> AsyncIterator[bytes]:
        # Force stream=false upstream
        body2 = dict(json_body)
        body2["stream"] = False

        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, headers=headers, json=body2)
            ct = (r.headers.get("content-type") or "").lower()
            logger.info("Upstream nonstream-for-emulated status=%s content-type=%s url=%s", r.status_code, ct, url)

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

        # Log upstream response metadata
        content_types = [b.get("type") for b in (msg.get("content") or []) if isinstance(b, dict)]
        logger.info(
            "Upstream response: stop_reason=%s content_types=%s output_tokens=%s",
            msg.get("stop_reason"),
            content_types,
            (msg.get("usage") or {}).get("output_tokens"),
        )

        async for ev in _emulate_anthropic_sse_from_message(msg, request_tools):
            yield ev

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "service": "kimicc-shim",
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
    stream = bool(body.get("stream", False))
    tools_present = isinstance(body.get("tools"), list) and len(body.get("tools") or []) > 0

    upstream_url = _upstream_url_for(request)
    headers = _pick_forward_headers(dict(request.headers))

    logger.info(
        "Incoming %s stream=%s tools=%s model=%s",
        str(request.url.path) + (("?" + request.url.query) if request.url.query else ""),
        stream,
        tools_present,
        body.get("model"),
    )

    if os.getenv("LOG_BODY", "0") == "1":
        logger.info("Body: %s", json.dumps(body, indent=2)[:20000])

    if stream:
        # Option A: if tools are present, emulate stream using a non-stream upstream call.
        if EMULATE_STREAM_FOR_TOOLS and tools_present:
            return await _proxy_stream_emulated_from_nonstream(upstream_url, headers=headers, json_body=body)
        # Otherwise, pass-through streaming.
        return await _proxy_stream_passthrough(upstream_url, headers=headers, json_body=body)

    return await _proxy_json(upstream_url, headers=headers, json_body=body)


@app.post("/v1/messages/count_tokens")
async def count_tokens(
    request: Request,
    anthropic_version: Optional[str] = Header(default=None),
    anthropic_beta: Optional[str] = Header(default=None),
):
    body = _as_dict(await request.json())

    if os.getenv("LOG_TOKENS", "0") == "1":
        logger.info("COUNT_TOKENS REQUEST: %s", json.dumps(body, indent=2)[:20000])
    else:
        logger.info("COUNT_TOKENS model=%s tools=%s", body.get("model"), len(body.get("tools") or []))

    headers = _pick_forward_headers(dict(request.headers))
    upstream_url = _upstream_url_for(request)
    return await _proxy_json(upstream_url, headers=headers, json_body=body)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"ok": True})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "kimicc.shim:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=False,
        log_level=os.getenv("UVICORN_LOG_LEVEL", "info"),
    )
