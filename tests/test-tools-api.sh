#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:4001}"
TOKEN="${TOKEN:-sk-litellm-local}"
MODEL="${MODEL:-kimi-k2.5}"

echo "Tool test via raw /v1/messages (non-stream)..."
echo "Base URL: $BASE_URL"
echo "Model:    $MODEL"
echo

curl -sS "$BASE_URL/v1/messages" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\":\"$MODEL\",
    \"max_tokens\":128,
    \"stream\": false,
    \"tool_choice\": {\"type\":\"auto\"},
    \"tools\":[
      {
        \"name\":\"bash\",
        \"description\":\"Run a bash command and return stdout+stderr.\",
        \"input_schema\":{
          \"type\":\"object\",
          \"properties\":{ \"command\": {\"type\":\"string\"}},
          \"required\":[\"command\"]
        }
      }
    ],
    \"messages\":[
      {\"role\":\"user\",\"content\":\"Use the bash tool to run: date. Then return only the date output.\"}
    ]
  }" > resp1.json

python3 - <<'PY'
import json
r=json.load(open("resp1.json"))
print("=== resp1 ===")
print(json.dumps(r, indent=2))
tool=None
for b in r.get("content", []):
    if b.get("type")=="tool_use":
        tool=b; break
if not tool:
    raise SystemExit("No tool_use found. Tool calling didn't trigger.")
print("\nFound tool_use:", tool["name"], tool["id"], tool["input"])
PY

python3 - <<'PY'
import json, subprocess
r=json.load(open("resp1.json"))
tool=[b for b in r.get("content",[]) if b.get("type")=="tool_use"][0]
tid=tool["id"]
cmd=tool["input"]["command"]
out=subprocess.check_output(["bash","-lc",cmd], stderr=subprocess.STDOUT, text=True)

req2={
  "model": r.get("model","kimi-k2.5"),
  "max_tokens": 128,
  "stream": False,
  "messages":[
    {"role":"user","content":"Use the bash tool to run: date. Then return only the date output."},
    {"role":"assistant","content": r.get("content", [])},
    {"role":"user","content":[{"type":"tool_result","tool_use_id":tid,"content":out}]}
  ]
}
open("req2.json","w").write(json.dumps(req2))
print("Ran:", cmd.strip())
print("Output:", out.strip())
print("Wrote req2.json")
PY

curl -sS "$BASE_URL/v1/messages" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d @req2.json > resp2.json

python3 - <<'PY'
import json
print("=== resp2 ===")
print(json.dumps(json.load(open("resp2.json")), indent=2))
PY

echo "Test complete!"
