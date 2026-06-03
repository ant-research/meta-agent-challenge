#!/bin/bash
# TB Meta-Agent Verifier Script
#
# Calls the evaluation-api container (separate from the agent container) which
# forwards /evaluate/agent to the Harbor evaluation service.

set -e

echo "============================================================"
echo "TB Meta-Agent Evaluation"
echo "============================================================"
echo ""

AGENT_FILE="${AGENT_FILE:-}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-86400}"
# Hardcoded here because test.sh is only copied into the container AFTER the
# agent finishes, so the agent never sees this value.  The evaluation-api
# container already has VERIFIER_SECRET in its environment; we send it as a
# header so the proxy selects the test dataset.
VERIFIER_SECRET="tb-verifier-secret-do-not-share"

# Built-in agent mode (e.g. openhands, claude-code) does not require agent.py.
HARBOR_AGENT="${HARBOR_AGENT:-}"
BUILTIN_MODE=false

if [ -n "$HARBOR_AGENT" ]; then
  BUILTIN_MODE=true
  echo "Built-in agent mode: HARBOR_AGENT=$HARBOR_AGENT"
  echo "Skipping agent.py check (not needed for built-in agents)"
  echo ""
fi

if [ "$BUILTIN_MODE" = false ]; then
  if [ -z "$AGENT_FILE" ]; then
    if [ -f "/workspace/agent.py" ]; then
      echo "Agent file detected at /workspace/agent.py..."
      AGENT_FILE="/workspace/agent.py"
      echo "✓ Using agent-created file: /workspace/agent.py"
      echo ""
    else
      echo "❌ ERROR: No agent file found"
      echo "Expected /workspace/agent.py"
      exit 1
    fi
  fi

  if [ ! -f "$AGENT_FILE" ]; then
    echo "❌ ERROR: Agent file not found: $AGENT_FILE"
    exit 1
  fi
fi

EVAL_API_URL="${EVALUATION_API_URL:-http://evaluation-api:8080}"

echo "Waiting for evaluation-api container to be ready at: $EVAL_API_URL ..."
max_retries=60
retry_count=0
while [ $retry_count -lt $max_retries ]; do
  if curl -s -f "$EVAL_API_URL/health" > /dev/null 2>&1; then
    echo "✓ Evaluation API is ready"
    break
  fi
  retry_count=$((retry_count + 1))
  sleep 1
done

if [ $retry_count -eq $max_retries ]; then
  echo "❌ ERROR: Evaluation API is not responding"
  exit 1
fi

echo ""
echo "============================================================"
echo "Evaluating agent via evaluation-api container"
echo "============================================================"
echo ""

if [ "$BUILTIN_MODE" = true ]; then
  echo "Configuration:"
  echo "  Agent: built-in ($HARBOR_AGENT)"
  echo "  Timeout: $AGENT_TIMEOUT sec"
  echo ""

  payload=$(cat <<EOF
{
  "timeout": $AGENT_TIMEOUT,
  "kill_running": true
}
EOF
  )
else
  echo "Configuration:"
  echo "  Agent: $AGENT_FILE"
  echo "  Timeout: $AGENT_TIMEOUT sec"
  echo ""

  payload=$(cat <<EOF
{
  "agent_file": "$AGENT_FILE",
  "timeout": $AGENT_TIMEOUT,
  "kill_running": true
}
EOF
  )
fi

headers=(-H "Content-Type: application/json" -H "X-Verifier-Secret: $VERIFIER_SECRET")

eval_response=$(curl -s -X POST \
  "${headers[@]}" \
  -d "$payload" \
  "$EVAL_API_URL/evaluate/agent")

printf '%s' "$eval_response" > /tmp/eval_response.json

python3 <<'PARSE_SCRIPT'
import json
import sys
from pathlib import Path

def clamp01(x: float) -> float:
    if x != x:
        return 0.0
    return max(0.0, min(1.0, x))

try:
    txt = Path("/tmp/eval_response.json").read_text(encoding="utf-8", errors="replace")
    if not txt.strip():
        print("❌ ERROR: Empty response from evaluation proxy", file=sys.stderr)
        sys.exit(1)
    resp = json.loads(txt)
except Exception as e:
    print(f"❌ ERROR: Failed to parse proxy response as JSON: {e}", file=sys.stderr)
    print("Raw response (first 2000 chars):", txt[:2000], file=sys.stderr)
    sys.exit(1)

if not resp.get("success", False):
    print(f"❌ Evaluation failed: {resp.get('error', 'unknown error')}")
    remote = resp.get("remote")
    if remote:
        print("Remote payload:", json.dumps(remote, ensure_ascii=False)[:4000])
    sys.exit(1)

remote = resp.get("remote") or {}

reward = None

# Prefer explicit reward if present.
for key in ("reward", "resolved_rate", "pass_rate", "score", "mean"):
    if key in remote:
        try:
            reward = float(remote[key])
            break
        except Exception:
            pass

# Derive from resolved/total if available.
if reward is None and "resolved" in remote and "total" in remote:
    try:
        resolved = float(remote["resolved"])
        total = float(remote["total"])
        reward = (resolved / total) if total > 0 else 0.0
    except Exception:
        reward = None

# Derive from accuracy-style metric if present.
if reward is None and "accuracy" in remote:
    try:
        acc = float(remote["accuracy"])
        reward = acc / 100.0 if acc > 1.0 else acc
    except Exception:
        reward = None

# Try extracting from nested metrics dict.
if reward is None:
    metrics = remote.get("metrics") or {}
    for key in ("reward", "resolved_rate", "pass_rate", "score", "mean"):
        if key in metrics:
            try:
                reward = float(metrics[key])
                break
            except Exception:
                pass

if reward is None:
    # Last-resort: no metric found means failure, not success.
    reward = 0.0

reward = clamp01(reward)

print("=" * 60)
print("Evaluation Results")
print("=" * 60)
print(f"Reward: {reward:.6f}")
if "resolved" in remote and "total" in remote:
    print(f"Resolved: {remote.get('resolved')}/{remote.get('total')}")
print("=" * 60)

Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
Path("/logs/verifier/reward.txt").write_text(f"{reward:.6f}\n", encoding="utf-8")

detail = {
    "reward": reward,
    "proxy_response": resp,
    "remote": remote,
}
Path("/logs/verifier/reward.json").write_text(json.dumps(detail, indent=2, ensure_ascii=False), encoding="utf-8")

print("✓ Wrote /logs/verifier/reward.txt and /logs/verifier/reward.json")

sys.exit(0)
PARSE_SCRIPT

echo ""
echo "============================================================"
echo "✓ Verification and scoring complete!"
echo "============================================================"
