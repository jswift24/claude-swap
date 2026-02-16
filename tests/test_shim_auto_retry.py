#!/usr/bin/env python3
"""
Test script for shim auto-retry functionality.

This script:
1. Checks shim health
2. Makes normal requests
3. Simulates incomplete responses to test retry logic
"""

import json
import sys
import os

SHIM_URL = "http://127.0.0.1:4001"

# Copy the functions we need to test
def _is_message_incomplete(msg):
    """Detect if a response appears incomplete (kimi-k2.5 stalled mid-generation)."""
    if not isinstance(msg, dict):
        return False

    stop_reason = msg.get("stop_reason")
    content = msg.get("content")

    if stop_reason:
        return False

    if isinstance(content, list) and content:
        last_block = content[-1]
        if not isinstance(last_block, dict):
            return False

        btype = last_block.get("type")

        if btype == "thinking":
            thinking_text = last_block.get("thinking") or last_block.get("text") or ""
            if thinking_text and not thinking_text.rstrip().endswith((".", "!", "?", "`", ")", "}")):
                print(f"    Detected incomplete thinking: ...{thinking_text[-30:]}")
                return True

        if btype == "tool_use":
            inp = last_block.get("input")
            if inp is None or (isinstance(inp, dict) and not inp):
                print("    Detected incomplete tool_use")
                return True

    if content:
        print("    Detected incomplete: no stop_reason with content")
        return True

    return False


def _build_continuation_request(original_body, accumulated_content):
    """Build a follow-up request to continue an incomplete generation."""
    new_body = dict(original_body)
    messages = list(new_body.get("messages", []))
    messages.append({"role": "assistant", "content": accumulated_content})
    messages.append({"role": "user", "content": "continue"})
    new_body["messages"] = messages
    return new_body


def test_health() -> bool:
    """Test 1: Check shim is running."""
    print("=" * 60)
    print("TEST 1: Shim Health Check")
    print("=" * 60)

    try:
        import urllib.request
        req = urllib.request.Request(f"{SHIM_URL}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            if data.get("ok"):
                print(f"✓ Shim is healthy")
                return True
    except Exception as e:
        print(f"✗ Shim health check failed: {e}")
        return False
    return False


def test_normal_request() -> bool:
    """Test 2: Make a normal request through the shim."""
    print("\n" + "=" * 60)
    print("TEST 2: Normal Request")
    print("=" * 60)

    try:
        import urllib.request
        req = urllib.request.Request(
            f"{SHIM_URL}/v1/messages",
            data=json.dumps({
                "model": "kimi-k2.5",
                "messages": [{"role": "user", "content": "Say hello"}],
                "stream": False,
                "max_tokens": 100
            }).encode(),
            headers={
                "Content-Type": "application/json",
                "x-api-key": "test-key"
            }
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            text = data.get("content", [{}])[0].get("text", "")
            stop_reason = data.get("stop_reason")

            print(f"Response: {text[:50]}...")
            print(f"Stop reason: {stop_reason}")

            if stop_reason == "end_turn":
                print("✓ Normal request completed")
                return True
            else:
                print(f"✗ Unexpected stop_reason: {stop_reason}")
                return False

    except Exception as e:
        print(f"✗ Request failed: {e}")
        return False


def test_incomplete_detection() -> bool:
    """Test 3: Verify incomplete detection logic."""
    print("\n" + "=" * 60)
    print("TEST 3: Incomplete Response Detection")
    print("=" * 60)

    test_cases = [
        ("Complete response", {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Hello!"}]
        }, False),

        ("Incomplete - no stop_reason", {
            "stop_reason": None,
            "content": [{"type": "text", "text": "Let me think"}]
        }, True),

        ("Incomplete - mid-thinking", {
            "stop_reason": None,
            "content": [{"type": "thinking", "thinking": "Analyzing code"}]
        }, True),

        ("Incomplete - mid-tool-call", {
            "stop_reason": None,
            "content": [{"type": "thinking", "thinking": "I'll use Edit("}]
        }, True),

        ("Complete thinking", {
            "stop_reason": "end_turn",
            "content": [{"type": "thinking", "thinking": "Analysis complete."}]
        }, False),

        ("Mid-JSON in thinking", {
            "stop_reason": None,
            "content": [{"type": "thinking", "thinking": '{"file_path": "/tmp/test"'}]
        }, True),
    ]

    all_passed = True
    for desc, msg, expected in test_cases:
        result = _is_message_incomplete(msg)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_passed = False
        print(f"  {status} {desc}")

    return all_passed


def test_continuation_request() -> bool:
    """Test 4: Verify continuation request building."""
    print("\n" + "=" * 60)
    print("TEST 4: Continuation Request Building")
    print("=" * 60)

    original = {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "kimi-k2.5"
    }
    accumulated = [{"type": "text", "text": "Partial..."}]

    new_req = _build_continuation_request(original, accumulated)
    messages = new_req.get("messages", [])

    if len(messages) != 3:
        print(f"✗ Expected 3 messages, got {len(messages)}")
        return False

    if messages[-1].get("content") != "continue":
        print(f"✗ Expected 'continue', got {messages[-1]}")
        return False

    print("✓ Continuation request built correctly")
    print(f"  User: {messages[0]['content']}")
    print(f"  Assistant (partial): {messages[1]['content'][0]['text']}")
    print(f"  User (continue): {messages[2]['content']}")
    return True


def test_simulated_flow() -> bool:
    """Test 5: Full retry flow simulation."""
    print("\n" + "=" * 60)
    print("TEST 5: Simulated Incomplete -> Retry Flow")
    print("=" * 60)

    original_body = {
        "messages": [{"role": "user", "content": "Write code"}],
        "model": "kimi-k2.5"
    }

    # Simulate first incomplete response
    partial_content = [{"type": "thinking", "thinking": "I'll use functions.Edit:"}]
    partial_msg = {
        "stop_reason": None,
        "content": partial_content,
        "usage": {"input_tokens": 10, "output_tokens": 5}
    }

    print("Step 1: First response (incomplete)")
    if not _is_message_incomplete(partial_msg):
        print("✗ Failed to detect incomplete")
        return False
    print("  ✓ Detected incomplete")

    print("\nStep 2: Build continuation request")
    new_body = _build_continuation_request(original_body, partial_content)
    print(f"  ✓ Built request with {len(new_body['messages'])} messages")

    print("\nStep 3: Simulate second response (complete)")
    second_content = [{"type": "text", "text": "Completed!"}]
    second_msg = {
        "stop_reason": "end_turn",
        "content": second_content,
        "usage": {"input_tokens": 15, "output_tokens": 5}
    }

    # Merge content (as shim does)
    merged_content = partial_content + second_content
    merged_usage = {
        "input_tokens": 10 + 15,
        "output_tokens": 5 + 5
    }

    print(f"  ✓ Merged content: {len(merged_content)} blocks")
    print(f"  ✓ Merged usage: {merged_usage}")

    print("\nStep 4: Verify final response is complete")
    final_msg = {
        "stop_reason": "end_turn",
        "content": merged_content,
        "usage": merged_usage
    }
    if _is_message_incomplete(final_msg):
        print("✗ Marked as incomplete when it should be complete")
        return False
    print("  ✓ Final response is complete")

    return True


def main():
    print("\n" + "=" * 60)
    print("SHIM AUTO-RETRY TEST SUITE")
    print("=" * 60)
    print(f"Testing shim at: {SHIM_URL}")

    results = []
    results.append(("Health Check", test_health()))
    results.append(("Normal Request", test_normal_request()))
    results.append(("Incomplete Detection", test_incomplete_detection()))
    results.append(("Continuation Building", test_continuation_request()))
    results.append(("Simulated Flow", test_simulated_flow()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed!")
        print("\nTo use the shim with auto-retry:")
        print("  1. Ensure shim is running from source (not pipx)")
        print("     pkill -f claude_swap.shim")
        print("     python3 -m uvicorn claude_swap.shim:app --host 127.0.0.1 --port 4001")
        print("  2. Use Claude Code with kimi-k2.5 normally")
        print("  3. When it would stall, shim will auto-continue (up to 3 retries)")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
