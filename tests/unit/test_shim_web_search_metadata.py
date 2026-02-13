from __future__ import annotations

import pytest

from cbridge import shim


class _DummyResponse:
    status_code = 200

    def __init__(self, payload: dict):
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _DummyAsyncClient:
    def __init__(self, payload: dict):
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        return _DummyResponse(self._payload)


@pytest.mark.asyncio
async def test_execute_server_tool_calls_preserves_web_search_metadata(monkeypatch):
    async def fake_execute_web_search(query: str, max_results: int = 5):
        return [
            {
                "type": "web_search_result",
                "url": "https://example.com",
                "title": "Example",
                "description": "Result description",
            }
        ]

    final_payload = {
        "id": "msg_final",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Found it."}],
        "usage": {"input_tokens": 10, "output_tokens": 20},
        "stop_reason": "end_turn",
    }

    monkeypatch.setattr(shim, "_execute_web_search", fake_execute_web_search)
    monkeypatch.setattr(shim.httpx, "AsyncClient", lambda *args, **kwargs: _DummyAsyncClient(final_payload))

    first_response = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "web_search",
                "input": {"query": "kimi k2.5"},
            }
        ],
        "stop_reason": "tool_use",
    }

    out = await shim._execute_server_tool_calls(
        response_msg=first_response,
        server_tool_names={"web_search"},
        original_body={"messages": [{"role": "user", "content": "search for kimi"}]},
        upstream_url="http://127.0.0.1:4000/v1/messages",
        headers={"content-type": "application/json"},
    )

    assert out["usage"]["server_tool_use"]["web_search_requests"] == 1
    block_types = [b["type"] for b in out["content"] if isinstance(b, dict)]
    assert "server_tool_use" in block_types
    assert "web_search_tool_result" in block_types


@pytest.mark.asyncio
async def test_emulated_sse_serializes_server_tool_blocks():
    msg = {
        "id": "msg_server_blocks",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "web_search",
                "input": {"query": "moonshot ai"},
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_1",
                "content": [
                    {
                        "type": "web_search_result",
                        "url": "https://example.com",
                        "title": "Example",
                    }
                ],
            },
            {"type": "text", "text": "Done."},
        ],
        "usage": {"input_tokens": 1, "output_tokens": 2},
        "stop_reason": "end_turn",
    }

    frames: list[str] = []
    async for chunk in shim._emulate_anthropic_sse_from_message(msg):
        frames.append(chunk.decode("utf-8"))
    stream_text = "".join(frames)

    assert '"type": "server_tool_use"' in stream_text
    assert '"type": "web_search_tool_result"' in stream_text
