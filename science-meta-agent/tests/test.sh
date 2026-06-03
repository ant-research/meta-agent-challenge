#!/bin/bash
# Science Meta-Agent Verifier Script
# Checks API violations and evaluates agent via API

set -Ee -o pipefail

VERIFIER_ERR_REPORTED=0

write_verifier_diagnostics() {
    local exit_code="$1"
    mkdir -p /logs/verifier 2>/dev/null || true
    cat > /logs/verifier/diagnostics.txt <<EOF_DIAG
script=$0
working_directory=$(pwd)
pid=$$
parent_pid=$PPID
bash_version=${BASH_VERSION:-<unknown>}
exit_code=$exit_code
error_line=${VERIFIER_ERROR_LINE:-}
error_command=${VERIFIER_ERROR_COMMAND:-}
err_trap_reported=$VERIFIER_ERR_REPORTED
EOF_DIAG
}

on_verifier_error() {
    local exit_code=$?
    VERIFIER_ERR_REPORTED=1
    VERIFIER_ERROR_LINE="$1"
    VERIFIER_ERROR_COMMAND="$2"
    echo ""
    echo "❌ VERIFIER SCRIPT ERROR"
    echo "  Line: $1"
    echo "  Exit code: $exit_code"
    echo "  Command: $2"
}

on_verifier_exit() {
    local exit_code=$?
    write_verifier_diagnostics "$exit_code"
    echo ""
    echo "============================================================"
    echo "Verifier Exit Summary"
    echo "============================================================"
    echo "Exit code: $exit_code"
    if [ "$exit_code" -ne 0 ] && [ "$VERIFIER_ERR_REPORTED" -ne 1 ]; then
        echo "Failure happened without an ERR trap context."
    fi
}

trap 'on_verifier_error "$LINENO" "$BASH_COMMAND"' ERR
trap 'on_verifier_exit' EXIT

echo "============================================================"
echo "Science Meta-Agent Evaluation"
echo "============================================================"
echo ""

# Configuration
AGENT_FILE="${AGENT_FILE:-}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-43200}"  # 12 hours default
SPLIT="${SCIENCE_SPLIT:-test}"
# Hardcoded here because test.sh is only copied into the container AFTER
# the agent finishes, so the agent never sees this value.
VERIFIER_SECRET="science-verifier-secret-do-not-share"

# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------
find_agent_side_process_groups() {
    local current_pid="$$"
    local current_pgid
    current_pgid=$(ps -o pgid= -p "$$" 2>/dev/null | tr -d ' ' || true)

    local pgids=""
    for pat in "requests.post.*evaluate/agent" "curl.*evaluate/agent" "/workspace/agent.py" \
               "claude --verbose --output-format=stream-json"; do
        local pids
        pids=$(pgrep -f "$pat" 2>/dev/null || true)
        for pid in $pids; do
            [ "$pid" = "$current_pid" ] && continue
            local pgid
            pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ') || continue
            [ -n "$pgid" ] && [ "$pgid" != "$current_pgid" ] && pgids="$pgids $pgid"
        done
    done
    echo "$pgids" | tr ' ' '\n' | sort -u | grep -v '^$' | tr '\n' ' '
}

cleanup_agent_side_processes() {
    echo "Cleaning up residual agent processes..."
    local pgids
    pgids=$(find_agent_side_process_groups || true)
    if [ -z "$(echo "$pgids" | tr -d ' ')" ]; then
        echo "✓ No residual agent processes found"
        return 0
    fi
    for pgid in $pgids; do
        echo "  Killing process group $pgid"
        kill -TERM -- "-$pgid" 2>/dev/null || true
    done
    sleep 2
    pgids=$(find_agent_side_process_groups || true)
    for pgid in $pgids; do
        kill -KILL -- "-$pgid" 2>/dev/null || true
    done
    [ -n "$(echo "$pgids" | tr -d ' ')" ] && sleep 1
    echo "✓ Agent-side cleanup complete"
}

cleanup_eval_slot_until_idle() {
    local max_rounds="${1:-8}"
    local round=1
    echo "Draining evaluation API slot..."
    while [ $round -le "$max_rounds" ]; do
        local http_code
        http_code=$(curl -sS -o /tmp/eval_cleanup_resp.json -w "%{http_code}" \
            -X POST \
            -H "Content-Type: application/json" \
            -H "X-Verifier-Secret: $VERIFIER_SECRET" \
            -d '{"kill_running": true}' \
            "$EVAL_API_URL/evaluate/agent")
        if [ "$http_code" != "409" ]; then
            echo "✓ Eval slot is idle (round $round, HTTP $http_code)"
            return 0
        fi
        echo "  Slot still busy, retrying ($round/$max_rounds)..."
        sleep 3
        round=$((round + 1))
    done
    echo "⚠️  Could not drain eval slot after $max_rounds rounds"
    return 0
}

# Check if agent file exists - try multiple detection methods
if [ -z "$AGENT_FILE" ]; then
    # Fallback: Check if agent created /workspace/agent.py
    if [ -f "/workspace/agent.py" ]; then
        echo "Agent file detected at /workspace/agent.py..."
        AGENT_FILE="/workspace/agent.py"
        echo "✓ Using agent-created file: /workspace/agent.py"
        echo ""
    else
        echo "❌ ERROR: No agent file found"
        echo ""
        echo "Tried the following detection methods:"
        echo "  1. AGENT_FILE environment variable: Not set"
        echo "  2. Agent-created file at /workspace/agent.py: Not found"
        echo ""
        echo "Please ensure one of the following:"
        echo "  - Set AGENT_FILE environment variable to your agent file path"
        echo "  - Create your agent at /workspace/agent.py (recommended)"
        exit 1
    fi
fi

# Verify the file exists (sanity check)
if [ ! -f "$AGENT_FILE" ]; then
    echo "❌ ERROR: Agent file not found: $AGENT_FILE"
    echo "The file path was set but the file does not exist."
    exit 1
fi

# ============================================================================
# Step 1: Check for API usage violations (via evaluation-api)
# ============================================================================
echo "============================================================"
echo "Step 1: Checking for unauthorized API usage"
echo "============================================================"
echo ""

# Use evaluation API service
EVAL_API_URL="${EVALUATION_API_URL:-http://evaluation-api:8080}"

# Wait for API to be ready
echo "Waiting for evaluation API to be ready..."
max_retries=30
retry_count=0
while [ $retry_count -lt $max_retries ]; do
    if curl -s -f "$EVAL_API_URL/health" > /dev/null 2>&1; then
        echo "✓ Evaluation API is ready"
        break
    fi
    retry_count=$((retry_count + 1))
    echo "Waiting for API... ($retry_count/$max_retries)"
    sleep 2
done

if [ $retry_count -eq $max_retries ]; then
    echo "❌ ERROR: Evaluation API is not responding"
    exit 1
fi

# Check for API violations
echo "Checking agent code for violations..."
monitor_response=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "X-Verifier-Secret: $VERIFIER_SECRET" \
    -d "{\"agent_file\": \"$AGENT_FILE\"}" \
    "$EVAL_API_URL/monitor")

is_valid=$(echo "$monitor_response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('is_valid', False))" 2>/dev/null)

MONITOR_VIOLATION_COUNT=0
if [ "$is_valid" != "True" ]; then
    echo ""
    echo "⚠️  WARNING: Unauthorized API usage detected (evaluation will still proceed)"
    echo ""
    echo "$monitor_response" | python3 -c "import sys,json; r=json.load(sys.stdin); print('Violations:'); [print(f\"  - {v['file']}:{v['line']} - {v['description']}\") for v in r.get('violations', [])]" 2>/dev/null
    MONITOR_VIOLATION_COUNT=$(echo "$monitor_response" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('violations', [])))" 2>/dev/null)
    echo ""
    echo "Allowed APIs:"
    echo "  - LLM: Use provided TASK_MODEL_API_BASE"
    echo "  - Search: Google Custom Search API only"
    echo ""
else
    echo "✓ No unauthorized API usage detected"
fi
# Pass violation count to the parse script via temp file
echo "$MONITOR_VIOLATION_COUNT" > /tmp/monitor_violation_count.txt
echo ""

# ============================================================================
# Step 2: Evaluate agent via API
# ============================================================================
echo "============================================================"
echo "Step 2: Evaluating agent via API"
echo "============================================================"
echo ""

echo "Using evaluation API at: $EVAL_API_URL"
echo ""
echo "Configuration:"
echo "  Agent: $AGENT_FILE"
echo "  Split: $SPLIT"
echo "  Timeout: $AGENT_TIMEOUT seconds ($(echo "scale=1; $AGENT_TIMEOUT/3600" | bc) hours)"
echo ""

echo "Calling evaluation API with agent file..."

# Thorough cleanup before eval: two passes to catch processes spawned between rounds
cleanup_agent_side_processes
cleanup_eval_slot_until_idle 8
cleanup_agent_side_processes
cleanup_eval_slot_until_idle 8
echo ""

# Prepare JSON payload
payload=$(cat <<EOF
{
  "agent_file": "$AGENT_FILE",
  "split": "$SPLIT",
  "timeout": $AGENT_TIMEOUT
}
EOF
)

echo "DEBUG: Request payload: $payload"

eval_http_code=$(curl -sS -o /tmp/eval_response.json -w "%{http_code}" \
    -X POST \
    -H "Content-Type: application/json" \
    -H "X-Verifier-Secret: $VERIFIER_SECRET" \
    -d "$payload" \
    "$EVAL_API_URL/evaluate/agent")

curl_exit_code=$?
echo "DEBUG: curl exit code: $curl_exit_code, HTTP status: $eval_http_code"

if [ $curl_exit_code -ne 0 ]; then
    echo "❌ ERROR: Failed to call evaluation API (curl exit code: $curl_exit_code)"
    exit 1
fi

if [ "$eval_http_code" = "409" ]; then
    echo "❌ ERROR: Evaluation API returned 409 (slot still busy after cleanup)"
    cat /tmp/eval_response.json 2>/dev/null || true
    exit 1
fi

eval_response=$(cat /tmp/eval_response.json)
echo "DEBUG: Response length: $(echo "$eval_response" | wc -c) bytes"
echo "DEBUG: Response preview (first 500 chars): ${eval_response:0:500}"

# Parse response and write reward files
echo ""
echo "Parsing evaluation results..."

python3 <<'PARSE_SCRIPT'
import json
import sys
from pathlib import Path

print("DEBUG: Python script started", file=sys.stderr)

try:
    # Parse API response
    with open('/tmp/eval_response.json', 'r') as f:
        response_text = f.read()
    print(f"DEBUG: Response text length: {len(response_text)}", file=sys.stderr)

    if not response_text or response_text.strip() == '':
        print("❌ ERROR: Empty response from evaluation API", file=sys.stderr)
        sys.exit(1)

    print("DEBUG: Attempting JSON parse...", file=sys.stderr)
    response = json.loads(response_text)
    print(f"DEBUG: JSON parsed successfully. Keys: {list(response.keys())}", file=sys.stderr)

    if not response.get('success', False):
        print(f"❌ Evaluation failed: {response.get('error', 'Unknown error')}")
        if 'traceback' in response:
            print(f"Traceback:\n{response['traceback']}")
        print(f"DEBUG: Full error response: {json.dumps(response, indent=2)}", file=sys.stderr)
        sys.exit(1)

    print("DEBUG: Evaluation successful, extracting metrics...", file=sys.stderr)

    # Extract metrics from nested evaluation object
    evaluation = response.get('evaluation', {})
    accuracy = evaluation.get('accuracy', 0.0)
    correct = evaluation.get('correct', 0)
    total = evaluation.get('total', 0)

    print(f"DEBUG: Extracted - accuracy: {accuracy}, correct: {correct}, total: {total}", file=sys.stderr)

    # Reward is accuracy (already 0.0-1.0)
    reward = accuracy

    # Read monitor violation count
    monitor_violation_count = 0
    try:
        with open('/tmp/monitor_violation_count.txt', 'r') as f:
            monitor_violation_count = int(f.read().strip())
    except Exception:
        pass

    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Reward: {reward:.4f}")
    print(f"Correct: {correct}/{total}")
    if monitor_violation_count > 0:
        print(f"Monitor Violations: {monitor_violation_count}")
    print(f"{'='*60}\n")

    # Write reward to file
    Path('/logs/verifier').mkdir(parents=True, exist_ok=True)

    print("DEBUG: Writing reward.txt...", file=sys.stderr)
    with open('/logs/verifier/reward.txt', 'w') as f:
        f.write(f"{reward:.6f}\n")
    print(f"✓ Reward written to /logs/verifier/reward.txt")

    # Write detailed results as JSON (including monitor violations)
    print("DEBUG: Writing reward.json...", file=sys.stderr)
    result_json = {
        'reward': reward,
        'accuracy': accuracy,
        'accuracy_percent': accuracy * 100,
        'correct': correct,
        'total': total,
        'monitor_violations': monitor_violation_count,
        'evaluation_response': response
    }

    with open('/logs/verifier/reward.json', 'w') as f:
        json.dump(result_json, f, indent=2)
    print(f"✓ Detailed results written to /logs/verifier/reward.json")

    print(f"DEBUG: Python script completed successfully", file=sys.stderr)
    sys.exit(0)

except json.JSONDecodeError as e:
    print(f"\n❌ Failed to parse evaluation response: {e}", file=sys.stderr)
    print(f"Raw response (first 1000 chars): {response_text[:1000]}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Scoring failed: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
PARSE_SCRIPT

parse_exit_code=$?
echo "DEBUG: Python parse script exit code: $parse_exit_code"

if [ $parse_exit_code -ne 0 ]; then
    echo ""
    echo "❌ VERIFICATION FAILED: Scoring error (Python script exited with code $parse_exit_code)"
    exit 1
fi

echo ""
echo "============================================================"
echo "✓ Verification and scoring complete!"
echo "✓ Agent evaluated successfully"
echo "============================================================"
