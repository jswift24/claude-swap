#!/usr/bin/env python3
"""Test the continuation logic for kimi-k2.5 incomplete responses."""


def _is_message_incomplete(msg):
    """
    Detect if a response appears incomplete (kimi-k2.5 stalled mid-generation).
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
                print(f"    Incomplete: thinking ends with '{thinking_text[-30:]}'")
                return True

        # Mid-tool-use: tool_use with partial/empty input might indicate truncation
        if btype == "tool_use":
            inp = last_block.get("input")
            if inp is None or (isinstance(inp, dict) and not inp):
                print(f"    Incomplete: tool_use with empty input")
                return True

    # No stop_reason with non-empty content suggests incomplete generation
    if content:
        print(f"    Incomplete: no stop_reason with content present")
        return True

    return False


def _build_continuation_request(original_body, accumulated_content):
    """
    Build a follow-up request to continue an incomplete generation.
    """
    new_body = dict(original_body)
    messages = list(new_body.get("messages", []))

    # Add the assistant's partial response
    messages.append({"role": "assistant", "content": accumulated_content})

    # Add continue prompt
    messages.append({"role": "user", "content": "continue"})

    new_body["messages"] = messages
    return new_body


def main():
    test_cases = [
        # (description, message, expected_incomplete)
        ("Complete with stop_reason", {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Hello"}]}, False),
        ("Incomplete - no stop_reason", {"stop_reason": None, "content": [{"type": "text", "text": "Hello"}]}, True),
        ("Incomplete - mid thinking (no period)", {"stop_reason": None, "content": [{"type": "thinking", "thinking": "Let me analyze this step by step"}]}, True),
        ("Incomplete - mid thinking (ends with code)", {"stop_reason": None, "content": [{"type": "thinking", "thinking": "Let me write the code `functions.Read:"}]}, True),
        ("Complete - thinking with punctuation", {"stop_reason": "end_turn", "content": [{"type": "thinking", "thinking": "Analysis complete."}]}, False),
        ("Empty message", {"stop_reason": None, "content": []}, False),
    ]

    print("Testing _is_message_incomplete:")
    all_passed = True
    for desc, msg, expected in test_cases:
        result = _is_message_incomplete(msg)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_passed = False
        print(f"  {status} {desc}: got={result}, expected={expected}")

    print("\nTesting _build_continuation_request:")
    original = {"messages": [{"role": "user", "content": "Hello"}], "model": "kimi-k2.5"}
    accumulated = [{"type": "text", "text": "Partial response"}]
    new_req = _build_continuation_request(original, accumulated)

    print(f"  Original had {len(original['messages'])} messages")
    print(f"  New request has {len(new_req['messages'])} messages")
    print(f"  Assistant message: {new_req['messages'][-2]}")
    print(f"  Continue message: {new_req['messages'][-1]}")

    if new_req['messages'][-1]['content'] == 'continue':
        print("  ✓ Continue message added correctly")
    else:
        print("  ✗ Continue message not added correctly")
        all_passed = False

    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
