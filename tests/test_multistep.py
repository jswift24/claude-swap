"""Multi-step integration test for the anthropic shim.

Sends a task that requires 10+ tool-use round-trips through the streaming
SSE emulation path (same path Claude Code uses). Parses SSE responses,
executes Bash tool calls in a sandbox dir, feeds results back, and repeats
until the model produces a final text-only response or we hit the step cap.

Detects stalls: if a response has no visible content (empty text, no tool_use),
that's exactly the bug this test is designed to catch.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any

import httpx


def parse_sse_events(raw: str) -> list[dict[str, Any]]:
    """Parse Anthropic SSE text into a list of (event_type, data_dict) pairs."""
    events = []
    current_event = None
    current_data = []

    for line in raw.split("\n"):
        if line.startswith("event: "):
            current_event = line[7:].strip()
            current_data = []
        elif line.startswith("data: "):
            current_data.append(line[6:])
        elif line == "" and current_event is not None:
            data_str = "\n".join(current_data)
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                data = {"raw": data_str}
            events.append({"event": current_event, "data": data})
            current_event = None
            current_data = []

    return events


def extract_response(events: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Reconstruct a message-like dict from SSE events.
    Returns: {content: [...], stop_reason: str, usage: dict, thinking_blocks: int}
    """
    content = []
    stop_reason = None
    usage = {}
    msg_id = None
    model = None
    thinking_count = 0

    for ev in events:
        etype = ev["event"]
        data = ev["data"]

        if etype == "message_start":
            msg = data.get("message", {})
            msg_id = msg.get("id")
            model = msg.get("model")
            usage = msg.get("usage", {})

        elif etype == "content_block_start":
            block = data.get("content_block", {})
            btype = block.get("type")
            if btype == "text":
                content.append({"type": "text", "text": "", "_index": data.get("index")})
            elif btype == "tool_use":
                content.append({
                    "type": "tool_use",
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input": {},
                    "_index": data.get("index"),
                })
            elif btype == "thinking":
                thinking_count += 1
                content.append({"type": "thinking", "thinking": "", "_index": data.get("index")})

        elif etype == "content_block_delta":
            idx = data.get("index")
            delta = data.get("delta", {})
            dtype = delta.get("type")
            # Find matching content block
            for block in content:
                if block.get("_index") == idx:
                    if dtype == "text_delta":
                        block["text"] += delta.get("text", "")
                    elif dtype == "input_json_delta":
                        partial = delta.get("partial_json", "")
                        if partial:
                            try:
                                block["input"] = json.loads(partial)
                            except json.JSONDecodeError:
                                # Accumulate partial JSON
                                block.setdefault("_partial", "")
                                block["_partial"] += partial
                    elif dtype == "thinking_delta":
                        block["thinking"] += delta.get("thinking", "")
                    break

        elif etype == "message_delta":
            d = data.get("delta", {})
            if "stop_reason" in d:
                stop_reason = d["stop_reason"]
            if "usage" in data:
                usage = data["usage"]

        elif etype == "error":
            raise RuntimeError(f"SSE error from shim: {data}")

    # Clean up internal fields
    for block in content:
        block.pop("_index", None)
        if "_partial" in block:
            try:
                block["input"] = json.loads(block.pop("_partial"))
            except json.JSONDecodeError:
                pass

    return {
        "id": msg_id,
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "usage": usage,
        "thinking_blocks": thinking_count,
    }


def execute_tool(name: str, tool_input: dict, work_dir: str) -> str:
    """Execute a tool call and return the output string."""
    if name.lower() == "bash":
        cmd = tool_input.get("command", "")
        try:
            result = subprocess.run(
                ["bash", "-lc", cmd],
                capture_output=True, text=True, timeout=30,
                cwd=work_dir,
            )
            output = result.stdout
            if result.stderr:
                output += result.stderr
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return "[command timed out after 30s]"
        except Exception as e:
            return f"[error: {e}]"
    else:
        return f"[unknown tool: {name}]"


TOOLS = [
    {
        "name": "Bash",
        "description": "Run a bash command and return stdout+stderr.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to run"}},
            "required": ["command"],
        },
    },
    {
        "name": "Read",
        "description": "Read a file and return its contents.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to the file to read"}},
            "required": ["path"],
        },
    },
    {
        "name": "Write",
        "description": "Write content to a file, creating it if needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
    },
]

INITIAL_PROMPT = """\
You have access to Bash, Read, and Write tools. Complete ALL of the following tasks, \
one tool call at a time (do NOT batch multiple commands into one call):

1. Create a directory called "project" in the current working directory
2. Write a Python file project/utils.py with a function `add(a, b)` that returns a+b
3. Write a Python file project/test_utils.py that imports add from utils and tests add(2,3)==5
4. Run the test with: cd project && python test_utils.py
5. Write project/greet.py with a function `greet(name)` returning "Hello, {name}!"
6. Write project/test_greet.py that tests greet("World") == "Hello, World!"
7. Run the greet test with: cd project && python test_greet.py
8. Write project/concat.py with `concat(*args)` that joins args with spaces
9. Write project/test_concat.py that tests concat("a","b","c") == "a b c"
10. Run the concat test with: cd project && python test_concat.py
11. List all files in the project directory
12. Write project/summary.txt with the text "All 3 modules created and tested successfully."
13. Read project/summary.txt to confirm it was written correctly

Do each step with exactly one tool call. Report "DONE" when all steps are complete.\
"""


def run_test(args: argparse.Namespace) -> None:
    base_url = args.base_url.rstrip("/")
    url = f"{base_url}/v1/messages"
    headers = {
        "Authorization": f"Bearer {args.token}",
        "Content-Type": "application/json",
    }
    timeout = httpx.Timeout(float(args.timeout), connect=30.0)

    messages: list[dict] = [{"role": "user", "content": INITIAL_PROMPT}]
    step = 0
    total_thinking_blocks = 0
    stall_count = 0
    start_time = time.time()

    print(f"\n--- Starting conversation ---")
    print(f"Task: 13-step file creation and testing task\n")

    while step < args.max_steps:
        step += 1
        elapsed = time.time() - start_time

        body = {
            "model": args.model,
            "max_tokens": 4096,
            "stream": True,
            "tools": TOOLS,
            "messages": messages,
        }

        print(f"[Step {step:2d}] Sending request ({len(messages)} messages, {elapsed:.0f}s elapsed)...", end=" ", flush=True)

        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, headers=headers, json=body)

        if r.status_code >= 400:
            print(f"ERROR: HTTP {r.status_code}")
            print(r.text[:1000])
            sys.exit(1)

        events = parse_sse_events(r.text)
        resp = extract_response(events)

        total_thinking_blocks += resp["thinking_blocks"]
        output_tokens = resp["usage"].get("output_tokens", "?")
        stop_reason = resp["stop_reason"]

        tool_uses = [b for b in resp["content"] if b.get("type") == "tool_use"]
        text_blocks = [b for b in resp["content"] if b.get("type") == "text"]
        thinking_blocks_in_resp = [b for b in resp["content"] if b.get("type") == "thinking"]

        visible_text = " ".join(b.get("text", "").strip() for b in text_blocks).strip()
        has_visible_content = bool(tool_uses) or bool(visible_text)

        parts = []
        if thinking_blocks_in_resp:
            total_chars = sum(len(b.get("thinking", "")) for b in thinking_blocks_in_resp)
            parts.append(f"thinking={len(thinking_blocks_in_resp)} ({total_chars} chars)")
        if text_blocks:
            parts.append(f"text={len(text_blocks)}")
        if tool_uses:
            names = [t.get("name", "?") for t in tool_uses]
            parts.append(f"tools={names}")
        parts.append(f"stop={stop_reason}")
        parts.append(f"tokens={output_tokens}")

        print(" | ".join(parts))

        if not has_visible_content:
            stall_count += 1
            print(f"  !! STALL DETECTED: no visible content (stall #{stall_count})")
            if stall_count >= 3:
                print(f"\nFAIL: 3 consecutive stalls detected.")
                sys.exit(1)
            messages.append({"role": "assistant", "content": resp["content"] or [{"type": "text", "text": ""}]})
            messages.append({"role": "user", "content": "Continue."})
            continue
        else:
            stall_count = 0

        if visible_text:
            preview = visible_text[:120] + ("..." if len(visible_text) > 120 else "")
            print(f"  text: {preview}")

        if tool_uses:
            messages.append({"role": "assistant", "content": resp["content"]})

            tool_results = []
            for tool in tool_uses:
                name = tool.get("name", "")
                tool_input = tool.get("input", {})
                tool_id = tool.get("id", "")

                if name.lower() == "read":
                    path = tool_input.get("path", "")
                    full_path = os.path.join(args.work_dir, path)
                    try:
                        with open(full_path) as f:
                            output = f.read()
                    except Exception as e:
                        output = f"[error: {e}]"
                elif name.lower() == "write":
                    path = tool_input.get("path", "")
                    content = tool_input.get("content", "")
                    full_path = os.path.join(args.work_dir, path)
                    try:
                        os.makedirs(os.path.dirname(full_path), exist_ok=True)
                        with open(full_path, "w") as f:
                            f.write(content)
                        output = f"Wrote {len(content)} bytes to {path}"
                    except Exception as e:
                        output = f"[error: {e}]"
                else:
                    output = execute_tool(name, tool_input, args.work_dir)

                print(f"  tool({name}): {output[:100]}{'...' if len(output)>100 else ''}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": output,
                })

            messages.append({"role": "user", "content": tool_results})

        else:
            if "DONE" in visible_text.upper() or stop_reason == "end_turn":
                print(f"\n--- Model finished at step {step} ---")
                break
            messages.append({"role": "assistant", "content": resp["content"]})
            messages.append({"role": "user", "content": "Continue with the remaining steps."})

    elapsed = time.time() - start_time
    tool_steps = sum(1 for m in messages if m["role"] == "assistant" and
                     any(b.get("type") == "tool_use" for b in (m.get("content") or []) if isinstance(b, dict)))

    print(f"\n{'='*50}")
    print(f"RESULTS:")
    print(f"  Total round-trips:    {step}")
    print(f"  Tool-use steps:       {tool_steps}")
    print(f"  Thinking blocks seen: {total_thinking_blocks}")
    print(f"  Stalls detected:      {stall_count}")
    print(f"  Elapsed:              {elapsed:.1f}s")
    print(f"{'='*50}")

    if tool_steps < args.min_steps:
        print(f"\nFAIL: Only {tool_steps} tool-use steps, expected at least {args.min_steps}.")
        print(f"The model may have batched commands or skipped steps.")
        sys.exit(1)

    if stall_count > 0:
        print(f"\nWARN: {stall_count} stall(s) detected but recovered.")
        sys.exit(1)

    print(f"\nPASS: {tool_steps} tool-use steps completed without stalling.")


def main():
    parser = argparse.ArgumentParser(description="Multi-step shim integration test")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--min-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--work-dir", required=True)
    run_test(parser.parse_args())


if __name__ == "__main__":
    main()
