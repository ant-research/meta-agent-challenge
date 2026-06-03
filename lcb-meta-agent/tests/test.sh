#!/bin/bash
# LiveCodeBench Meta-Agent Verifier Script
# Checks API violations and evaluates agent via API

set -Ee -o pipefail

VERIFIER_ERR_REPORTED=0
VERIFIER_PARENT_CMD=""
VERIFIER_SHELL_ARGV=""
VERIFIER_ERROR_LINE=""
VERIFIER_ERROR_COMMAND=""

write_verifier_diagnostics() {
    local exit_code="$1"

    mkdir -p /logs/verifier 2>/dev/null || true
    cat > /logs/verifier/diagnostics.txt <<EOF_DIAG
script=$0
shell_argv=${VERIFIER_SHELL_ARGV:-<unavailable>}
working_directory=$(pwd)
pid=$$
parent_pid=$PPID
parent_command=${VERIFIER_PARENT_CMD:-<unavailable>}
bash_version=${BASH_VERSION:-<unknown>}
exit_code=$exit_code
error_line=${VERIFIER_ERROR_LINE:-}
error_command=${VERIFIER_ERROR_COMMAND:-}
err_trap_reported=$VERIFIER_ERR_REPORTED
EOF_DIAG
}

print_verifier_invocation_context() {
    local parent_cmd shell_cmd

    parent_cmd=$(ps -o args= -p "$PPID" 2>/dev/null || true)
    shell_cmd=$(tr '\0' ' ' < "/proc/$$/cmdline" 2>/dev/null || true)
    VERIFIER_PARENT_CMD="$parent_cmd"
    VERIFIER_SHELL_ARGV="$shell_cmd"

    echo "============================================================"
    echo "Verifier Invocation Context"
    echo "============================================================"
    echo "Script: $0"
    echo "Shell argv: ${shell_cmd:-<unavailable>}"
    echo "Working directory: $(pwd)"
    echo "PID: $$"
    echo "Parent PID: $PPID"
    echo "Parent command: ${parent_cmd:-<unavailable>}"
    echo "BASH_VERSION: ${BASH_VERSION:-<unknown>}"
    echo ""
}

on_verifier_error() {
    local exit_code=$?
    local line_no="$1"
    local command="$2"

    VERIFIER_ERR_REPORTED=1
    VERIFIER_ERROR_LINE="$line_no"
    VERIFIER_ERROR_COMMAND="$command"

    echo ""
    echo "❌ VERIFIER SCRIPT ERROR"
    echo "  Line: $line_no"
    echo "  Exit code: $exit_code"
    echo "  Command: $command"
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
        echo "This usually means the shell exited before reaching a trapped command."
    fi
}

trap 'on_verifier_error "$LINENO" "$BASH_COMMAND"' ERR
trap 'on_verifier_exit' EXIT

print_verifier_invocation_context

echo "============================================================"
echo "LiveCodeBench Meta-Agent Evaluation"
echo "============================================================"
echo ""

AGENT_FILE="${AGENT_FILE:-}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-43200}"
SPLIT="${LCB_SPLIT:-test}"
CASE_TIMEOUT="${LCB_CASE_TIMEOUT:-10}"
VERIFIER_SECRET="lcb-verifier-secret-do-not-share"
EVAL_API_URL="${EVALUATION_API_URL:-http://evaluation-api:8080}"

# Check if agent file exists - try multiple detection methods
if [ -z "$AGENT_FILE" ]; then
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

monitor_response=$(curl -s -X POST \
    -H "X-Verifier-Secret: $VERIFIER_SECRET" \
    "$EVAL_API_URL/monitor")

monitor_success=$(echo "$monitor_response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('success', False))" 2>/dev/null)

MONITOR_VIOLATION_COUNT=0
if [ "$monitor_success" != "True" ]; then
    echo "$monitor_response" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('stdout','')); print(r.get('stderr',''))" 2>/dev/null
    MONITOR_VIOLATION_COUNT=$(echo "$monitor_response" | python3 -c "
import sys, json
r = json.load(sys.stdin)
stdout = r.get('stdout', '')
count = sum(1 for line in stdout.splitlines() if line.strip().startswith('- ') and 'Detected' in line)
print(max(count, 1))
" 2>/dev/null)
    echo ""
    echo "⚠️  WARNING: Unauthorized API usage detected (evaluation will still proceed)"
    echo "Only the provided API endpoint may be used."
    echo ""
else
    echo "$monitor_response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('stdout',''))" 2>/dev/null
    echo ""
    echo "✓ No unauthorized API usage detected"
fi
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
echo "  Case timeout: $CASE_TIMEOUT seconds"
echo ""

parse_cleanup_state() {
    python3 <<'PY'
import json
from pathlib import Path

try:
    data = json.loads(Path('/tmp/eval_cleanup_response.json').read_text(encoding='utf-8'))
except Exception:
    print("false\tfalse\t\t\t")
    raise SystemExit(0)

prev = data.get('previous_state') or {}
fields = [
    "true" if data.get('success', False) else "false",
    "true" if data.get('killed', False) else "false",
    str(prev.get('run_id', '')),
    str(prev.get('split', '')),
    str(prev.get('timeout', '')),
]
print("\t".join(fields))
PY
}

parse_busy_state() {
    python3 <<'PY'
import json
from pathlib import Path

try:
    data = json.loads(Path('/tmp/eval_response.json').read_text(encoding='utf-8'))
except Exception:
    print("\t\t")
    raise SystemExit(0)

state = data.get('state') or {}
fields = [
    str(state.get('run_id', '')),
    str(state.get('split', '')),
    str(state.get('timeout', '')),
]
print("\t".join(fields))
PY
}

find_agent_side_process_groups() {
    local current_pid current_pgid

    current_pid="$$"
    current_pgid=$(ps -o pgid= -p "$$" 2>/dev/null | tr -d ' ' || true)

    python3 - "$current_pid" "$current_pgid" <<'PY'
import subprocess
import sys

current_pid = sys.argv[1]
current_pgid = sys.argv[2]

try:
    ps_output = subprocess.run(
        ["ps", "-eo", "pid=,pgid=,args="],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
except Exception:
    raise SystemExit(0)


def is_agent_residual(args: str) -> bool:
    is_claude = "claude --verbose --output-format=stream-json" in args
    is_tail = "tail -n +1 -F /logs/agent/claude-code.txt" in args
    is_codex = "codex exec " in args or "/logs/agent/codex.txt" in args
    is_gemini = "gemini --yolo " in args or "/logs/agent/gemini-cli.txt" in args
    has_eval_endpoint = "/evaluate/agent" in args
    has_eval_split = (
        "'split': 'eval'" in args
        or '"split": "eval"' in args
        or "'split':'eval'" in args
        or '"split":"eval"' in args
        or "split=eval" in args
        or "split = eval" in args
    )
    is_eval_debugger = has_eval_endpoint and (
        "kill_running" in args
        or "first_k" in args
        or has_eval_split
        or ("requests.post(" in args and "eval" in args)
    )
    return is_claude or is_tail or is_codex or is_gemini or is_eval_debugger


rows_by_pgid: dict[str, list[tuple[str, str]]] = {}
for raw_line in ps_output:
    line = raw_line.strip()
    if not line:
        continue
    parts = line.split(None, 2)
    if len(parts) != 3:
        continue
    pid, pgid, args = parts
    if pid == current_pid or pgid == current_pgid:
        continue
    if not is_agent_residual(args):
        continue
    rows_by_pgid.setdefault(pgid, []).append((pid, args))

for pgid in sorted(rows_by_pgid, key=lambda value: int(value)):
    for pid, args in rows_by_pgid[pgid]:
        print(f"{pgid}\t{pid}\t{args}")
PY
}

cleanup_agent_side_processes() {
    local reason="$1"
    local residual_rows remaining_rows
    local -a residual_pgids=()

    echo ""
    echo "Cleaning up residual agent-side processes: $reason"

    residual_rows=$(find_agent_side_process_groups || true)
    if [ -z "$residual_rows" ]; then
        echo "✓ No residual agent-side processes detected"
        return 0
    fi

    while IFS=$'\t' read -r pgid pid args; do
        [ -z "$pgid" ] && continue
        residual_pgids+=("$pgid")
        echo "DEBUG: Residual process candidate: pgid=$pgid pid=$pid args=$args"
    done <<< "$residual_rows"

    mapfile -t residual_pgids < <(printf '%s\n' "${residual_pgids[@]}" | awk 'NF && !seen[$0]++')

    for pgid in "${residual_pgids[@]}"; do
        kill -TERM -- "-$pgid" 2>/dev/null || true
        echo "DEBUG: Sent SIGTERM to residual process group $pgid"
    done
    sleep 2

    remaining_rows=$(find_agent_side_process_groups || true)
    if [ -n "$remaining_rows" ]; then
        echo "DEBUG: Some residual agent-side processes survived SIGTERM; escalating to SIGKILL"
        mapfile -t residual_pgids < <(printf '%s\n' "$remaining_rows" | awk -F '\t' 'NF >= 2 {print $1}' | awk 'NF && !seen[$0]++')
        for pgid in "${residual_pgids[@]}"; do
            kill -KILL -- "-$pgid" 2>/dev/null || true
            echo "DEBUG: Sent SIGKILL to residual process group $pgid"
        done
        sleep 1
    fi

    remaining_rows=$(find_agent_side_process_groups || true)
    if [ -n "$remaining_rows" ]; then
        echo "⚠️  WARNING: Residual agent-side processes remain after cleanup:"
        while IFS=$'\t' read -r pgid pid args; do
            [ -z "$pgid" ] && continue
            echo "  pgid=$pgid pid=$pid args=$args"
        done <<< "$remaining_rows"
    else
        echo "✓ Residual agent-side processes cleared"
    fi
}

cleanup_eval_slot_until_idle() {
    local reason="$1"
    local max_rounds="$2"
    local round=1
    local cleanup_payload='{"kill_running": true}'

    echo ""
    echo "Cleaning up residual evaluation(s): $reason"

    while [ $round -le "$max_rounds" ]; do
        echo "DEBUG: Cleanup round $round/$max_rounds"

        cleanup_http_code=$(curl -sS -o /tmp/eval_cleanup_response.json -w "%{http_code}" \
            -X POST \
            -H "Content-Type: application/json" \
            -H "X-Verifier-Secret: $VERIFIER_SECRET" \
            -d "$cleanup_payload" \
            "$EVAL_API_URL/evaluate/agent")

        cleanup_curl_exit_code=$?
        echo "DEBUG: Cleanup curl exit code: $cleanup_curl_exit_code"
        echo "DEBUG: Cleanup HTTP status: $cleanup_http_code"

        if [ $cleanup_curl_exit_code -ne 0 ]; then
            echo "❌ ERROR: Failed to send cleanup request (curl exit code: $cleanup_curl_exit_code)"
            exit 1
        fi

        cleanup_response=$(cat /tmp/eval_cleanup_response.json)
        echo "DEBUG: Cleanup response length: $(echo "$cleanup_response" | wc -c) bytes"
        echo "DEBUG: Cleanup response preview (first 500 chars): ${cleanup_response:0:500}"

        if [ "$cleanup_http_code" -lt 200 ] || [ "$cleanup_http_code" -ge 300 ]; then
            echo "⚠️  WARNING: Cleanup request returned HTTP $cleanup_http_code"
        fi

        cleanup_state=$(parse_cleanup_state)
        IFS=$'\t' read -r cleanup_success cleanup_killed cleanup_run_id cleanup_split cleanup_timeout <<< "$cleanup_state"
        echo "DEBUG: Cleanup parsed state: success=$cleanup_success killed=$cleanup_killed previous_run_id=${cleanup_run_id:-<none>} previous_split=${cleanup_split:-<none>} previous_timeout=${cleanup_timeout:-<none>}"

        if [ "$cleanup_killed" != "true" ] && [ -z "$cleanup_run_id" ]; then
            echo "✓ Evaluation slot appears idle after cleanup"
            return 0
        fi

        echo "DEBUG: Cleanup removed an active evaluation; waiting 2s before re-checking..."
        sleep 2
        round=$((round + 1))
    done

    echo "⚠️  WARNING: Cleanup hit max rounds ($max_rounds); verifier will still try to start evaluation"
    return 0
}

INITIAL_CLEANUP_MAX_ROUNDS=8
POST_409_CLEANUP_MAX_ROUNDS=8
cleanup_agent_side_processes "before verifier run"
cleanup_eval_slot_until_idle "before verifier run" "$INITIAL_CLEANUP_MAX_ROUNDS"

echo "Calling evaluation API with agent file..."

payload=$(cat <<EOF_JSON
{
  "agent_file": "$AGENT_FILE",
  "split": "$SPLIT",
  "timeout": $AGENT_TIMEOUT,
  "case_timeout": $CASE_TIMEOUT
}
EOF_JSON
)

echo "DEBUG: Request payload: $payload"

MAX_EVAL_ATTEMPTS=5
EVAL_RETRY_SLEEP_SEC=3
attempt=1
while true; do
    eval_http_code=$(curl -sS -o /tmp/eval_response.json -w "%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -H "X-Verifier-Secret: $VERIFIER_SECRET" \
        -d "$payload" \
        "$EVAL_API_URL/evaluate/agent")

    curl_exit_code=$?
    echo "DEBUG: Evaluation curl exit code: $curl_exit_code"
    echo "DEBUG: Evaluation HTTP status: $eval_http_code (attempt $attempt/$MAX_EVAL_ATTEMPTS)"

    if [ $curl_exit_code -ne 0 ]; then
        echo "❌ ERROR: Failed to call evaluation API (curl exit code: $curl_exit_code)"
        exit 1
    fi

    eval_response=$(cat /tmp/eval_response.json)

    if [ "$eval_http_code" = "409" ]; then
        busy_state=$(parse_busy_state)
        IFS=$'\t' read -r busy_run_id busy_split busy_timeout <<< "$busy_state"
        echo "DEBUG: Busy state from 409: run_id=${busy_run_id:-<none>} split=${busy_split:-<none>} timeout=${busy_timeout:-<none>}"
    fi

    if [ "$eval_http_code" != "409" ] || [ $attempt -ge $MAX_EVAL_ATTEMPTS ]; then
        break
    fi

    echo "⚠️  Evaluation API is still busy after cleanup; running cleanup again before retry..."
    cleanup_agent_side_processes "after verifier received HTTP 409 on attempt $attempt"
    cleanup_eval_slot_until_idle "after verifier received HTTP 409 on attempt $attempt" "$POST_409_CLEANUP_MAX_ROUNDS"
    echo "⚠️  Retrying evaluation API in ${EVAL_RETRY_SLEEP_SEC}s..."
    sleep "$EVAL_RETRY_SLEEP_SEC"
    attempt=$((attempt + 1))
done

echo "DEBUG: Response length: $(echo "$eval_response" | wc -c) bytes"
echo "DEBUG: Response preview (first 500 chars): ${eval_response:0:500}"

echo ""
echo "Parsing evaluation results..."

printf '%s' "$eval_response" > /tmp/eval_response.json

python3 <<'PARSE_SCRIPT'
import json
import sys
from pathlib import Path

print("DEBUG: Python script started", file=sys.stderr)

try:
    with open('/tmp/eval_response.json', 'r', encoding='utf-8') as f:
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
        if 'agent_output' in response and response['agent_output']:
            agent_out = response['agent_output']
            if len(agent_out) > 10000:
                print(f"\n--- Agent Output (last 10000 of {len(agent_out)} chars) ---")
                print(agent_out[-10000:])
            else:
                print(f"\n--- Agent Output ({len(agent_out)} chars) ---")
                print(agent_out)
            print("--- End Agent Output ---")
        if 'agent_stderr' in response and response['agent_stderr']:
            agent_err = response['agent_stderr']
            if len(agent_err) > 5000:
                print(f"\n--- Agent Stderr (last 5000 of {len(agent_err)} chars) ---")
                print(agent_err[-5000:])
            else:
                print(f"\n--- Agent Stderr ({len(agent_err)} chars) ---")
                print(agent_err)
            print("--- End Agent Stderr ---")
        print(f"DEBUG: Error response keys: {list(response.keys())}", file=sys.stderr)
        sys.exit(1)

    print("DEBUG: Evaluation successful, extracting metrics...", file=sys.stderr)

    accuracy = float(response.get('accuracy', 0.0))
    correct = int(response.get('correct', 0))
    total = int(response.get('total', 0))
    covered = int(response.get('covered', 0))
    coverage = float(response.get('coverage', 0.0))

    print(
        f"DEBUG: Extracted - accuracy: {accuracy}, correct: {correct}, total: {total}, "
        f"covered: {covered}, coverage: {coverage}",
        file=sys.stderr,
    )

    reward = accuracy / 100.0

    monitor_violation_count = 0
    try:
        with open('/tmp/monitor_violation_count.txt', 'r', encoding='utf-8') as f:
            monitor_violation_count = int(f.read().strip())
    except Exception:
        pass

    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Reward: {reward:.4f}")
    print(f"Correct: {correct}/{total}")
    print(f"Covered: {covered}/{total} ({coverage:.2f}%)")
    if monitor_violation_count > 0:
        print(f"Monitor Violations: {monitor_violation_count}")
    print(f"{'='*60}\n")

    Path('/logs/verifier').mkdir(parents=True, exist_ok=True)

    print("DEBUG: Writing reward.txt...", file=sys.stderr)
    with open('/logs/verifier/reward.txt', 'w', encoding='utf-8') as f:
        f.write(f"{reward:.6f}\n")
    print("✓ Reward written to /logs/verifier/reward.txt")

    print("DEBUG: Writing reward.json...", file=sys.stderr)
    result_json = {
        'reward': reward,
        'accuracy_percent': accuracy,
        'correct': correct,
        'covered': covered,
        'coverage_percent': coverage,
        'total': total,
        'monitor_violations': monitor_violation_count,
        'evaluation_response': response,
    }

    with open('/logs/verifier/reward.json', 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)
    print("✓ Detailed results written to /logs/verifier/reward.json")

    if 'detailed_results' in response:
        with open('/logs/verifier/detailed_results.json', 'w', encoding='utf-8') as f:
            json.dump(response['detailed_results'], f, ensure_ascii=False, indent=2)

    print("DEBUG: Python script completed successfully", file=sys.stderr)
    sys.exit(0)

except json.JSONDecodeError as e:
    print(f"\n❌ Failed to parse evaluation response: {e}", file=sys.stderr)
    print(f"Raw response (first 1000 chars): {response_text[:1000]}", file=sys.stderr)
    print(
        f"DEBUG: JSONDecodeError at position {e.pos if hasattr(e, 'pos') else 'unknown'}",
        file=sys.stderr,
    )
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Scoring failed: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    print(f"DEBUG: Exception type: {type(e).__name__}", file=sys.stderr)
    sys.exit(1)
PARSE_SCRIPT

parse_exit_code=$?
echo "DEBUG: Python parse script exit code: $parse_exit_code"

if [ $parse_exit_code -ne 0 ]; then
    echo ""
    echo "❌ VERIFICATION FAILED: Scoring error (Python script exited with code $parse_exit_code)"
    echo "DEBUG: Check stderr output above for detailed error messages"
    exit 1
fi

echo ""
echo "============================================================"
echo "✓ Verification and scoring complete!"
echo "✓ Agent evaluated successfully"
echo "============================================================"
